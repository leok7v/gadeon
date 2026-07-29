import Accelerate
import Foundation

// Qwen3-VL vision tower (the Bonsai-27B mmproj GGUF) on the CPU via
// Accelerate GEMMs: pixels -> patch embed (the two temporal conv kernels
// summed -- for a still image llama.cpp adds their outputs, and conv is
// linear) -> merge-block reorder -> learned position embed -> N uniform
// pre-LN blocks with vision M-RoPE attention -> post-LN -> the 2x2 merger
// MLP into the LM embedding space. Every geometry number comes from the
// GGUF metadata or a tensor shape; nothing is hardcoded per model.
// Transcribed from llama.cpp clip_graph_qwen3vl + ggml's GGML_ROPE_TYPE_
// VISION path, and gated byte-for-byte against a numpy reference by
// `gadeon-cli --vit`.
// This mmproj carries NO deepstack layers (is_deepstack_layers all false,
// no deepstack tensors), so the merged output is the whole story.

public struct ViTConfig {
    public let imageSize: Int
    public let patchSize: Int
    public let merge: Int
    public let embd: Int
    public let ff: Int
    public let layers: Int
    public let heads: Int
    public let eps: Float
    public let mean: [Float]
    public let std: [Float]
    public let projDim: Int

    public var side: Int { imageSize / patchSize }
    public var patches: Int { side * side }
    public var mergedTokens: Int { patches / (merge * merge) }
    public var headDim: Int { embd / heads }
    public var patchDim: Int { patchSize * patchSize * 3 }

    // Force-read (crash, don't fake): a missing key means a broken mmproj.
    // merge falls back to a SHAPE derivation (mm.0 input / embd), so even
    // that is the file's own geometry, not a constant.
    init(_ g: GGUF) {
        func i(_ k: String) -> Int { g.int("clip.vision." + k)! }
        imageSize = i("image_size")
        patchSize = i("patch_size")
        embd = i("embedding_length")
        ff = i("feed_forward_length")
        layers = i("block_count")
        heads = i("attention.head_count")
        eps = Float(g.double("clip.vision.attention.layer_norm_epsilon")!)
        mean = g.doubles("clip.vision.image_mean")!.map { Float($0) }
        std = g.doubles("clip.vision.image_std")!.map { Float($0) }
        let mm0In = g.tensor("mm.0.weight").dims[0]
        let m2 = g.int("clip.vision.spatial_merge_size") ?? mm0In / i("embedding_length")
        merge = g.int("clip.vision.spatial_merge_size") != nil
            ? m2 : Int(Double(m2).squareRoot().rounded())
        projDim = g.tensor("mm.2.weight").dims[1]
    }

    // Header-only probe (mmap + KV parse, no weight materialization), for
    // capability checks and grid sizing without the ~1GB tower load.
    public init(mmprojPath: String) throws {
        self.init(try GGUF(path: mmprojPath))
    }
}

// Weight matrices are stored TRANSPOSED at load ([in][out]) so every
// product is a plain row-major vDSP_mmul, no per-call transpose.
private struct ViTLayer {
    let ln1w: [Float], ln1b: [Float]
    let qkvW: [Float], qkvB: [Float]
    let outW: [Float], outB: [Float]
    let ln2w: [Float], ln2b: [Float]
    let upW: [Float], upB: [Float]
    let downW: [Float], downB: [Float]

    init(_ g: GGUF, _ il: Int, _ e: Int, _ ff: Int) {
        func t(_ s: String) -> [Float] {
            Dense.floats(g.tensor("v.blk.\(il).\(s)"))
        }
        ln1w = t("ln1.weight"); ln1b = t("ln1.bias")
        qkvW = ViT.transposed(t("attn_qkv.weight"), 3 * e, e)
        qkvB = t("attn_qkv.bias")
        outW = ViT.transposed(t("attn_out.weight"), e, e)
        outB = t("attn_out.bias")
        ln2w = t("ln2.weight"); ln2b = t("ln2.bias")
        upW = ViT.transposed(t("ffn_up.weight"), ff, e)
        upB = t("ffn_up.bias")
        downW = ViT.transposed(t("ffn_down.weight"), e, ff)
        downB = t("ffn_down.bias")
    }
}

public final class ViT {
    public let cfg: ViTConfig
    private let patchW: [Float]      // summed dual conv, [patchDim][embd]
    private let patchB: [Float]
    private let posEmbd: [Float]     // [patches][embd], raster row order
    private let postW: [Float], postB: [Float]
    private let mm0W: [Float], mm0B: [Float]
    private let mm2W: [Float], mm2B: [Float]
    private let blocks: [ViTLayer]
    // Sequence order is 2x2 merge-block order; mergeOrder[s] is the raster
    // patch index at sequence slot s (llama.cpp's window reorder).
    private let mergeOrder: [Int]
    // Vision M-RoPE cos/sin per sequence slot: headDim/2 angle pairs, the
    // first half keyed by the patch row, the second by the column, each with
    // inv_freq 10000^(-2j/headDim); dims (d, d+headDim/2) rotate together.
    private let ropeCos: [Float]
    private let ropeSin: [Float]

    public init(path: String) throws {
        let g = try GGUF(path: path)
        cfg = ViTConfig(g)
        let w0 = Dense.floats(g.tensor("v.patch_embd.weight"))
        let w1 = Dense.floats(g.tensor("v.patch_embd.weight.1"))
        var sum = w0
        vDSP_vadd(w0, 1, w1, 1, &sum, 1, vDSP_Length(w0.count))
        let pd = cfg.patchSize * cfg.patchSize * 3
        patchW = ViT.transposed(sum, cfg.embd, pd)
        patchB = Dense.floats(g.tensor("v.patch_embd.bias"))
        posEmbd = Dense.floats(g.tensor("v.position_embd.weight"))
        postW = Dense.floats(g.tensor("v.post_ln.weight"))
        postB = Dense.floats(g.tensor("v.post_ln.bias"))
        let mmIn = cfg.embd * cfg.merge * cfg.merge
        mm0W = ViT.transposed(Dense.floats(g.tensor("mm.0.weight")),
                              mmIn, mmIn)
        mm0B = Dense.floats(g.tensor("mm.0.bias"))
        mm2W = ViT.transposed(Dense.floats(g.tensor("mm.2.weight")),
                              cfg.projDim, mmIn)
        mm2B = Dense.floats(g.tensor("mm.2.bias"))
        let e = cfg.embd, f = cfg.ff
        blocks = (0..<cfg.layers).map { il in ViTLayer(g, il, e, f) }
        mergeOrder = ViT.mergeOrderTable(cfg)
        let tables = ViT.ropeTables(cfg, mergeOrder)
        ropeCos = tables.cos
        ropeSin = tables.sin
    }

    // The tables below are STATIC so the GPU tower (MetalViT) builds its
    // buffers from the exact same code -- the two engines cannot drift on
    // layout, and `gadeon-cli --vit` cross-gates their forwards.

    // Sequence order is 2x2 merge-block order; result[s] is the raster patch
    // index at sequence slot s (llama.cpp's window reorder).
    static func mergeOrderTable(_ cfg: ViTConfig) -> [Int] {
        let side = cfg.side
        let m = cfg.merge
        var order: [Int] = []
        order.reserveCapacity(cfg.patches)
        for wi in 0..<(side / m) {
            for wj in 0..<(side / m) {
                for dy in 0..<m {
                    for dx in 0..<m {
                        order.append((m * wi + dy) * side + (m * wj + dx))
                    }
                }
            }
        }
        return order
    }

    // Vision M-RoPE cos/sin per sequence slot: headDim/2 angle pairs, the
    // first half keyed by the patch row, the second by the column, each with
    // inv_freq 10000^(-2j/headDim); dims (d, d+headDim/2) rotate together.
    static func ropeTables(_ cfg: ViTConfig, _ order: [Int])
        -> (cos: [Float], sin: [Float]) {
        let side = cfg.side
        let half = cfg.headDim / 2
        let quarter = half / 2
        var cosT = [Float](repeating: 0, count: cfg.patches * half)
        var sinT = [Float](repeating: 0, count: cfg.patches * half)
        for s in 0..<cfg.patches {
            let y = Float(order[s] / side)
            let x = Float(order[s] % side)
            for j in 0..<half {
                let pos = j < quarter ? y : x
                let f = pow(10000.0, -Float(j % quarter) / Float(quarter))
                cosT[s * half + j] = cos(pos * f)
                sinT[s * half + j] = sin(pos * f)
            }
        }
        return (cosT, sinT)
    }

    // Patch rows in merge-block order with ggml's im2col layout (channel-
    // major within a patch, the two temporal copies collapsed).
    static func patchRows(_ px: [Float], _ cfg: ViTConfig,
                          _ order: [Int]) -> [Float] {
        let n = cfg.patches, pd = cfg.patchDim
        let p = cfg.patchSize, side = cfg.side, img = cfg.imageSize
        var rows = [Float](repeating: 0, count: n * pd)
        for s in 0..<n {
            let py = (order[s] / side) * p
            let pxx = (order[s] % side) * p
            for c in 0..<3 {
                for ky in 0..<p {
                    for kx in 0..<p {
                        rows[s * pd + c * p * p + ky * p + kx] =
                            px[((py + ky) * img + pxx + kx) * 3 + c]
                    }
                }
            }
        }
        return rows
    }

    // Normalized pixels ([imageSize * imageSize * 3] HWC, already
    // (v - mean) / std) -> merged features [mergedTokens * projDim] f32.
    public func forward(pixels: [Float]) -> [Float] {
        var x = patchEmbed(pixels)
        for l in blocks { block(l, &x) }
        layerNorm(&x, postW, postB)
        return merger(x)
    }

    // Patch rows in merge-block order (the shared im2col), one GEMM against
    // the summed conv kernel, then bias + the reordered learned position
    // embedding.
    private func patchEmbed(_ px: [Float]) -> [Float] {
        let n = cfg.patches, pd = cfg.patchDim, e = cfg.embd
        let rows = ViT.patchRows(px, cfg, mergeOrder)
        var out = [Float](repeating: 0, count: n * e)
        matmul(rows, patchW, n, e, pd, &out)
        for s in 0..<n {
            let raster = mergeOrder[s]
            for c in 0..<e {
                out[s * e + c] += patchB[c] + posEmbd[raster * e + c]
            }
        }
        return out
    }

    private func block(_ l: ViTLayer, _ x: inout [Float]) {
        let n = cfg.patches, e = cfg.embd
        var h = x
        layerNorm(&h, l.ln1w, l.ln1b)
        var qkv = [Float](repeating: 0, count: n * 3 * e)
        matmul(h, l.qkvW, n, 3 * e, e, &qkv)
        addBias(&qkv, l.qkvB, n)
        let ctx = attention(qkv)
        var attnOut = [Float](repeating: 0, count: n * e)
        matmul(ctx, l.outW, n, e, e, &attnOut)
        addBias(&attnOut, l.outB, n)
        vDSP_vadd(x, 1, attnOut, 1, &x, 1, vDSP_Length(n * e))
        h = x
        layerNorm(&h, l.ln2w, l.ln2b)
        var up = [Float](repeating: 0, count: n * cfg.ff)
        matmul(h, l.upW, n, cfg.ff, e, &up)
        addBias(&up, l.upB, n)
        gelu(&up)
        var down = [Float](repeating: 0, count: n * e)
        matmul(up, l.downW, n, e, cfg.ff, &down)
        addBias(&down, l.downB, n)
        vDSP_vadd(x, 1, down, 1, &x, 1, vDSP_Length(n * e))
    }

    // Per-head attention over qkv [n][3*embd] (q | k | v, head-major within
    // each): M-RoPE q and k, scores softmax, context back at the head's
    // slot so the out projection consumes [n][embd] directly.
    private func attention(_ qkv: [Float]) -> [Float] {
        let n = cfg.patches, e = cfg.embd, d = cfg.headDim
        let scale = 1 / Float(d).squareRoot()
        var ctx = [Float](repeating: 0, count: n * e)
        for head in 0..<cfg.heads {
            var q = [Float](repeating: 0, count: n * d)
            var k = [Float](repeating: 0, count: n * d)
            var v = [Float](repeating: 0, count: n * d)
            for s in 0..<n {
                let base = s * 3 * e + head * d
                for i in 0..<d {
                    q[s * d + i] = qkv[base + i]
                    k[s * d + i] = qkv[base + e + i]
                    v[s * d + i] = qkv[base + 2 * e + i]
                }
                rope(&q, s); rope(&k, s)
            }
            var kT = [Float](repeating: 0, count: n * d)
            vDSP_mtrans(k, 1, &kT, 1, vDSP_Length(d), vDSP_Length(n))
            var scores = [Float](repeating: 0, count: n * n)
            matmul(q, kT, n, n, d, &scores)
            var sc = scale
            vDSP_vsmul(scores, 1, &sc, &scores, 1, vDSP_Length(n * n))
            softmaxRows(&scores, n)
            var o = [Float](repeating: 0, count: n * d)
            matmul(scores, v, n, d, n, &o)
            for s in 0..<n {
                for i in 0..<d { ctx[s * e + head * d + i] = o[s * d + i] }
            }
        }
        return ctx
    }

    // Rotate row s of a [n][headDim] buffer in place: pair (j, j + half).
    private func rope(_ b: inout [Float], _ s: Int) {
        let d = cfg.headDim, half = d / 2
        for j in 0..<half {
            let c = ropeCos[s * half + j]
            let sn = ropeSin[s * half + j]
            let x0 = b[s * d + j]
            let x1 = b[s * d + j + half]
            b[s * d + j] = x0 * c - x1 * sn
            b[s * d + j + half] = x0 * sn + x1 * c
        }
    }

    // The 2x2 merger: 4 consecutive merge-ordered rows concat (a plain
    // reshape), mm.0 -> GELU -> mm.2 into the LM embedding space.
    private func merger(_ x: [Float]) -> [Float] {
        let m2 = cfg.merge * cfg.merge
        let n = cfg.patches / m2
        let inDim = cfg.embd * m2
        var h = [Float](repeating: 0, count: n * inDim)
        matmul(x, mm0W, n, inDim, inDim, &h)
        addBias(&h, mm0B, n)
        gelu(&h)
        var out = [Float](repeating: 0, count: n * cfg.projDim)
        matmul(h, mm2W, n, cfg.projDim, inDim, &out)
        addBias(&out, mm2B, n)
        return out
    }

    // ---- primitives -----------------------------------------------------

    // ggml [out][in] row-major -> [in][out], so products are plain mmul.
    static func transposed(_ a: [Float], _ rows: Int, _ cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: a.count)
        vDSP_mtrans(a, 1, &out, 1, vDSP_Length(cols), vDSP_Length(rows))
        return out
    }

    // C[m][n] = A[m][k] @ B[k][n], all row-major.
    private func matmul(_ a: [Float], _ b: [Float], _ m: Int, _ n: Int,
                        _ k: Int, _ c: inout [Float]) {
        vDSP_mmul(a, 1, b, 1, &c, 1,
                  vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
    }

    private func addBias(_ x: inout [Float], _ b: [Float], _ rows: Int) {
        let n = b.count
        for r in 0..<rows {
            b.withUnsafeBufferPointer { bp in
                x.withUnsafeMutableBufferPointer { xp in
                    vDSP_vadd(xp.baseAddress! + r * n, 1, bp.baseAddress!, 1,
                              xp.baseAddress! + r * n, 1, vDSP_Length(n))
                }
            }
        }
    }

    private func layerNorm(_ x: inout [Float], _ w: [Float], _ b: [Float]) {
        let e = w.count
        let rows = x.count / e
        for r in 0..<rows {
            var mean: Float = 0
            var sq: Float = 0
            x.withUnsafeBufferPointer { xp in
                vDSP_meanv(xp.baseAddress! + r * e, 1, &mean, vDSP_Length(e))
                vDSP_measqv(xp.baseAddress! + r * e, 1, &sq, vDSP_Length(e))
            }
            let inv = 1 / (sq - mean * mean + cfg.eps).squareRoot()
            for i in 0..<e {
                x[r * e + i] = (x[r * e + i] - mean) * inv * w[i] + b[i]
            }
        }
    }

    // ggml's GELU: the tanh approximation, vectorized via vvtanhf.
    private func gelu(_ x: inout [Float]) {
        let n = x.count
        var t = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let v = x[i]
            t[i] = 0.7978845608 * (v + 0.044715 * v * v * v)
        }
        var cnt = Int32(n)
        vvtanhf(&t, t, &cnt)
        for i in 0..<n { x[i] = 0.5 * x[i] * (1 + t[i]) }
    }

    private func softmaxRows(_ x: inout [Float], _ n: Int) {
        for r in 0..<n {
            var mx: Float = 0
            var sum: Float = 0
            x.withUnsafeMutableBufferPointer { xp in
                let row = xp.baseAddress! + r * n
                vDSP_maxv(row, 1, &mx, vDSP_Length(n))
                mx = -mx
                vDSP_vsadd(row, 1, &mx, row, 1, vDSP_Length(n))
                var cnt = Int32(n)
                vvexpf(row, row, &cnt)
                vDSP_sve(row, 1, &sum, vDSP_Length(n))
                var inv = 1 / sum
                vDSP_vsmul(row, 1, &inv, row, 1, vDSP_Length(n))
            }
        }
    }
}

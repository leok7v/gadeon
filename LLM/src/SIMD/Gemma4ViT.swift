import Accelerate
import Foundation

// The gemma-4 vision tower on the CPU, mirroring Gemma4VisionModel.forward in
// transformers 5.14.1. It is NOT shaped like the Qwen ViT next door: there is
// no conv patch embed, no square grid, no LayerNorm and no 2x2 merger. It is
// far closer to gemma's own TEXT block -- sandwich RMS norms, a gated MLP,
// q/k norms and a scale-free v_norm -- which is why it reuses GK rather than
// ViT's primitives.
//
// Native resolution is the structural difference. The processor hands over a
// FIXED number of patches padded to a ceiling, each with an explicit (x, y)
// and (-1, -1) marking padding; pooling then averages over a k x k grid of
// those coordinates and DROPS the slots no real patch reached. So the soft
// token count follows the aspect ratio -- measured 252 to 273 on the fixture
// set against a configured 280 that is a ceiling no real image reaches.
//
// Weights are dequantized per layer inside the forward rather than held: the
// tower is 0.167 B parameters, which is 672 MB held as f32 and about 40 MB
// held one layer at a time.

public struct Gemma4VisionConfig {
    public let embd: Int
    public let layers: Int
    public let patchSize: Int
    public let poolKernel: Int
    public let maxSoftTokens: Int
    let headDim: Int
    let heads: Int
    let ff: Int
    let ropeTheta: Float
    let eps: Float
    let posTable: Int
    let activation: GemmaActivation

    public var patchDim: Int { patchSize * patchSize * 3 }

    init(_ g: GGUF) throws {
        func i(_ k: String) -> Int { g.int("gemma4.vision." + k)! }
        embd = i("embedding_length")
        layers = i("block_count")
        patchSize = i("patch_size")
        poolKernel = i("pool_kernel")
        maxSoftTokens = i("max_soft_tokens")
        ropeTheta = Float(g.double("gemma4.vision.rope_theta")!)
        eps = Float(g.double("gemma4.attention.layer_norm_rms_epsilon")!)
        // Head geometry from the tensors: q_norm is [head_dim], and the q
        // projection's output width divided by that is the head count.
        headDim = g.tensor("v.blk.0.attn_q_norm.weight").dims[0]
        heads = g.tensor("v.blk.0.attn_q.weight").dims[1] / headDim
        ff = g.tensor("v.blk.0.ffn_gate.weight").dims[1]
        posTable = g.tensor("v.position_embd.weight").dims[1]
        activation = try GemmaActivation.read(g, "gemma4.vision.activation")
    }
}

// How an image is marked up in a turn. The processor writes
// `boi + imageToken * N + eoi`, where N is the tower's ACTUAL soft-token
// count for this image -- not a constant, since it follows the aspect ratio.

public struct Gemma4VisionWire: Sendable {
    public let boi: Int32
    public let eoi: Int32
    public let token: Int32

    init(_ g: GGUF) throws {
        boi = Int32(try requireInt(g, "gemma4.boi_token_id",
                                   "an image turn has no opening marker"))
        eoi = Int32(try requireInt(g, "gemma4.eoi_token_id",
                                   "an image turn has no closing marker"))
        token = Int32(try requireInt(g, "gemma4.image_token_id",
                                     "an image has no placeholder to expand"))
    }
}

// A video reuses the IMAGE markers and the vision tower; what is its own is
// the placeholder id, how many frames to sample, and each frame's smaller
// soft-token budget.

public struct Gemma4VideoWire: Sendable {
    public let token: Int32
    public let frames: Int
    public let softTokensPerFrame: Int

    init(_ g: GGUF) throws {
        token = Int32(try requireInt(g, "gemma4.video_token_id",
                                     "a video has no placeholder to expand"))
        frames = try requireInt(g, "gemma4.video.num_frames",
                                "the frame count is unknown")
        softTokensPerFrame = try requireInt(
            g, "gemma4.video.max_soft_tokens",
            "the per-frame soft-token budget is unknown")
    }
}

public final class Gemma4ViT {
    public let cfg: Gemma4VisionConfig
    private let gguf: GGUF
    private let posEmbd: [Float]        // [2][posTable][embd]

    private let srq: [String: SRQ]

    public init(_ model: Gemma4Model) throws {
        gguf = model.gguf
        cfg = try Gemma4VisionConfig(model.gguf)
        posEmbd = Dense.floats(model.gguf.tensor("v.position_embd.weight"))
        var scales: [String: SRQ] = [:]
        for il in 0..<cfg.layers {
            for tag in ["attn_q", "attn_k", "attn_v", "attn_out",
                        "ffn_gate", "ffn_up", "ffn_down"] {
                let name = "v.blk.\(il).\(tag).weight"
                scales[name] = SRQ(model.gguf, name)
            }
        }
        srq = scales
    }

    // One QUANTIZED linear with the trained activation clamp. Routing every
    // quantized projection through here is what keeps a site from skipping it.
    private func linear(_ name: String, _ x: [Float], _ n: Int,
                        _ inDim: Int, _ outDim: Int) -> [Float] {
        let s = srq[name] ?? SRQ.none
        var h = x
        SRQ.apply(&h, s.input)
        var out = [Float](repeating: 0, count: n * outDim)
        vDSP_mmul(h, 1, matrix(name, inDim, outDim), 1, &out, 1,
                  vDSP_Length(n), vDSP_Length(outDim), vDSP_Length(inDim))
        SRQ.apply(&out, s.output)
        return out
    }

    // One image: `pixels` is [patches * patchDim] in [0,1] straight from the
    // processor, `pos` the per-patch (x, y) with -1 marking padding. Returns
    // the tower output and its projection into the LM space, both stripped to
    // the slots real patches reached.
    public func forward(pixels: [Float], pos: [(Int, Int)])
        -> (tower: [Float], proj: [Float], count: Int) {
        let n = pos.count
        let e = cfg.embd
        var x = patchEmbed(pixels, pos, n)
        let padded = pos.map { p in p.0 < 0 && p.1 < 0 }
        let tables = ropeTables(pos)
        for il in 0..<cfg.layers { block(il, &x, n, padded, tables) }
        let pooled = pool(x, pos, padded, n)
        var tower = pooled.rows
        let scale = Float(e).squareRoot()
        for i in 0..<tower.count { tower[i] *= scale }
        return (tower, project(tower, pooled.count), pooled.count)
    }

    // 2 * (p - 0.5) is applied in model code rather than by the processor, so
    // the [0,1] pixels arrive un-normalized and this is the only place it
    // happens. Padding patches keep whatever the projection gives them and are
    // zeroed later by the pooler; only their POSITION embedding is dropped.
    func patchEmbed(_ pixels: [Float], _ pos: [(Int, Int)],
                            _ n: Int) -> [Float] {
        let e = cfg.embd, pd = cfg.patchDim
        var scaled = [Float](repeating: 0, count: n * pd)
        for i in 0..<(n * pd) { scaled[i] = 2 * (pixels[i] - 0.5) }
        var out = [Float](repeating: 0, count: n * e)
        let w = matrix("v.patch_embd.weight", pd, e)
        vDSP_mmul(scaled, 1, w, 1, &out, 1,
                  vDSP_Length(n), vDSP_Length(e), vDSP_Length(pd))
        for s in 0..<n {
            if pos[s].0 >= 0 || pos[s].1 >= 0 {
                let xi = max(pos[s].0, 0), yi = max(pos[s].1, 0)
                let xb = xi * e
                let yb = (cfg.posTable + yi) * e
                for c in 0..<e {
                    out[s * e + c] += posEmbd[xb + c] + posEmbd[yb + c]
                }
            }
        }
        return out
    }

    // cos/sin per patch, 64 wide: the first 32 channels rotate by the patch's
    // x and the second 32 by its y, each half pairing (j, j + 16). Built here
    // rather than per layer because every layer shares them.
    func ropeTables(_ pos: [(Int, Int)])
        -> (cos: [Float], sin: [Float]) {
        let d = cfg.headDim
        let per = d / 2                    // 32 rotated channels per axis
        let half = per / 2                 // 16 real inverse frequencies
        var c = [Float](repeating: 0, count: pos.count * d)
        var s = [Float](repeating: 0, count: pos.count * d)
        for p in 0..<pos.count {
            for axis in 0..<2 {
                let coord = Float(max(axis == 0 ? pos[p].0 : pos[p].1, 0))
                for j in 0..<half {
                    let f = powf(cfg.ropeTheta,
                                 -2 * Float(j) / Float(per))
                    let a = coord * f
                    let base = p * d + axis * per + j
                    c[base] = cosf(a)
                    s[base] = sinf(a)
                    c[base + half] = cosf(a)
                    s[base + half] = sinf(a)
                }
            }
        }
        return (c, s)
    }

    private func block(_ il: Int, _ x: inout [Float], _ n: Int,
                       _ padded: [Bool],
                       _ rope: (cos: [Float], sin: [Float])) {
        let e = cfg.embd
        func w(_ s: String) -> [Float] {
            Dense.floats(gguf.tensor("v.blk.\(il).\(s)"))
        }
        var h = x
        rmsRows(&h, e, n, w("ln1.weight"))
        var attn = attention(il, h, n, padded, rope)
        rmsRows(&attn, e, n, w("post_attn_norm.weight"))
        vDSP_vadd(x, 1, attn, 1, &x, 1, vDSP_Length(n * e))

        h = x
        rmsRows(&h, e, n, w("ln2.weight"))
        var ff = mlp(il, h, n)
        rmsRows(&ff, e, n, w("post_ffn_norm.weight"))
        vDSP_vadd(x, 1, ff, 1, &x, 1, vDSP_Length(n * e))
    }

    private func mlp(_ il: Int, _ h: [Float], _ n: Int) -> [Float] {
        let e = cfg.embd, f = cfg.ff
        var gate = linear("v.blk.\(il).ffn_gate.weight", h, n, e, f)
        let up = linear("v.blk.\(il).ffn_up.weight", h, n, e, f)
        let act = cfg.activation
        for i in 0..<(n * f) { gate[i] = act.apply(gate[i]) * up[i] }
        return linear("v.blk.\(il).ffn_down.weight", gate, n, f, e)
    }

    // Bidirectional, scaling 1.0 (q_norm is the query's only normalization),
    // and padding patches are masked OUT of every row rather than merely
    // contributing zero.
    private func attention(_ il: Int, _ h: [Float], _ n: Int,
                           _ padded: [Bool],
                           _ rope: (cos: [Float], sin: [Float])) -> [Float] {
        let e = cfg.embd, d = cfg.headDim, nH = cfg.heads
        var q = linear("v.blk.\(il).attn_q.weight", h, n, e, nH * d)
        var k = linear("v.blk.\(il).attn_k.weight", h, n, e, nH * d)
        var v = linear("v.blk.\(il).attn_v.weight", h, n, e, nH * d)
        let qn = Dense.floats(gguf.tensor("v.blk.\(il).attn_q_norm.weight"))
        let kn = Dense.floats(gguf.tensor("v.blk.\(il).attn_k_norm.weight"))
        GK.rmsnormRows(&q, d: d, rows: n * nH, w: qn, eps: cfg.eps)
        GK.rmsnormRows(&k, d: d, rows: n * nH, w: kn, eps: cfg.eps)
        // v_norm is scale-free and leaves no tensor, exactly as in the text
        // tower.
        GK.rmsnormRowsNoWeight(&v, d: d, rows: n * nH, eps: cfg.eps)
        applyRope(&q, n, rope)
        applyRope(&k, n, rope)

        var ctx = [Float](repeating: 0, count: n * e)
        // Heads write disjoint column slices of ctx and read only q/k/v, so
        // the fill is race-free. Serial this is minutes on a 2520-patch
        // image, which is the whole forward.
        ctx.withUnsafeMutableBufferPointer { cb in
            q.withUnsafeBufferPointer { qb in
                k.withUnsafeBufferPointer { kb in
                    v.withUnsafeBufferPointer { vb in
                        nonisolated(unsafe) let cp = cb.baseAddress!
                        nonisolated(unsafe) let qp = qb.baseAddress!
                        nonisolated(unsafe) let kp = kb.baseAddress!
                        nonisolated(unsafe) let vp = vb.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: nH) {
                            head in
                            var scores = [Float](repeating: 0, count: n)
                            for i in 0..<n {
                                for j in 0..<n {
                                    var dot: Float = 0
                                    let qo = (i * nH + head) * d
                                    let ko = (j * nH + head) * d
                                    for c in 0..<d {
                                        dot += qp[qo + c] * kp[ko + c]
                                    }
                                    scores[j] = padded[j]
                                        ? -Float.greatestFiniteMagnitude : dot
                                }
                                GK.softmaxInPlace(&scores, n)
                                let oo = i * e + head * d
                                for j in 0..<n where scores[j] != 0 {
                                    let vo = (j * nH + head) * d
                                    let s = scores[j]
                                    for c in 0..<d {
                                        cp[oo + c] += s * vp[vo + c]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return linear("v.blk.\(il).attn_out.weight", ctx, n, nH * d, e)
    }

    private func applyRope(_ t: inout [Float], _ n: Int,
                           _ rope: (cos: [Float], sin: [Float])) {
        let d = cfg.headDim, nH = cfg.heads
        let per = d / 2, half = per / 2
        for p in 0..<n {
            for head in 0..<nH {
                let b = (p * nH + head) * d
                for axis in 0..<2 {
                    let o = axis * per
                    for j in 0..<half {
                        let c = rope.cos[p * d + o + j]
                        let s = rope.sin[p * d + o + j]
                        let a = t[b + o + j]
                        let bb = t[b + o + j + half]
                        t[b + o + j] = a * c - bb * s
                        t[b + o + j + half] = a * s + bb * c
                    }
                }
            }
        }
    }

    // Average over a k x k grid of PATCH COORDINATES, not of sequence slots:
    // the divisor is the fixed k^2 rather than the number of patches that
    // landed in the cell, so a padding patch contributes zero without
    // diluting its neighbours. A slot no patch reached is dropped, which is
    // where the variable soft-token count comes from.
    func pool(_ x: [Float], _ pos: [(Int, Int)], _ padded: [Bool],
                      _ n: Int) -> (rows: [Float], count: Int) {
        let e = cfg.embd
        let k = cfg.poolKernel
        let slots = n / (k * k)
        var maxX = 0
        for p in pos { maxX = max(maxX, max(p.0, 0)) }
        let cols = (maxX + 1) / k
        var acc = [Float](repeating: 0, count: slots * e)
        var used = [Bool](repeating: false, count: slots)
        let inv = 1 / Float(k * k)
        for s in 0..<n {
            let xi = max(pos[s].0, 0) / k
            let yi = max(pos[s].1, 0) / k
            let slot = xi + cols * yi
            if slot < slots {
                used[slot] = true
                if !padded[s] {
                    for c in 0..<e { acc[slot * e + c] += x[s * e + c] * inv }
                }
            }
        }
        var rows: [Float] = []
        rows.reserveCapacity(slots * e)
        var count = 0
        for slot in 0..<slots where used[slot] {
            rows.append(contentsOf: acc[(slot * e)..<((slot + 1) * e)])
            count += 1
        }
        return (rows, count)
    }

    // embed_vision: a SCALE-FREE norm then a plain projection into the LM
    // width, so the tower's four-orders-of-magnitude output range is tamed
    // before the text stack ever sees it.
    func project(_ tower: [Float], _ rows: Int) -> [Float] {
        let e = cfg.embd
        let w = gguf.tensor("mm.vision.weight")
        let outDim = w.dims[1]
        var h = tower
        GK.rmsnormRowsNoWeight(&h, d: e, rows: rows, eps: cfg.eps)
        var out = [Float](repeating: 0, count: rows * outDim)
        vDSP_mmul(h, 1, matrix("mm.vision.weight", e, outDim), 1, &out, 1,
                  vDSP_Length(rows), vDSP_Length(outDim), vDSP_Length(e))
        return out
    }

    // A weight as [in][out] row-major, so every product is a plain mmul.
    // ggml stores [out][in]; dequantizing here rather than at load keeps the
    // resident cost to one layer.
    private func matrix(_ name: String, _ inDim: Int,
                        _ outDim: Int) -> [Float] {
        Dense.transposedFloats(gguf.tensor(name), inDim, outDim)
    }

    private func rmsRows(_ x: inout [Float], _ d: Int, _ rows: Int,
                         _ w: [Float]) {
        GK.rmsnormRows(&x, d: d, rows: rows, w: w, eps: cfg.eps)
    }
}

extension Gemma4ViT {
    // The stages the GPU tower shares. Pooling and the projection are one
    // pass over at most a few hundred rows, and the patch embedding is index
    // gymnastics over a lookup table -- keeping both here means the
    // drop-empty-slot rule and the (x, y) lookup have ONE definition rather
    // than two that can drift.
    func embedForGPU(pixels: [Float], pos: [(Int, Int)]) -> [Float] {
        patchEmbed(pixels, pos, pos.count)
    }

    func ropeForGPU(pos: [(Int, Int)]) -> (cos: [Float], sin: [Float]) {
        ropeTables(pos)
    }

    func poolAndProject(_ x: [Float], pos: [(Int, Int)], padded: [Bool])
        -> (tower: [Float], proj: [Float], count: Int) {
        let pooled = pool(x, pos, padded, pos.count)
        var tower = pooled.rows
        let scale = Float(cfg.embd).squareRoot()
        for i in 0..<tower.count { tower[i] *= scale }
        return (tower, project(tower, pooled.count), pooled.count)
    }
}

// The Qwen3-VL vision tower (mmproj GGUF) on the GPU: the same op sequence
// as the CPU ViT (SIMD/ViT.swift), which stays the oracle -- `gadeon-cli
// --vit` cross-gates the two forwards on identical pixels. Weights are
// dequanted ONCE at load into resident f16 MTLBuffers (half the f32
// footprint; activations stay f32, GEMMs accumulate f32) and kept for the
// engine's lifetime, so an image turn pays neither the reload nor the
// dequant. Matrices stay in ggml's native [out][in] row-major -- the f16
// GEMM kernel reads W[m,k] directly, so the CPU path's load-time transposes
// do not exist here. All geometry comes from the mmproj (ViTConfig); the
// merge order, rope tables, and im2col layout are the CPU ViT's own statics,
// so the two engines cannot drift on layout. One command buffer per tile.
import Accelerate
import Foundation
import Metal

public final class MetalViT {
    public let cfg: ViTConfig
    // Holds the mmap alive: ctx wraps it bytesNoCopy, and Dense reads
    // tensors out of it during init.
    private let gguf: GGUF
    private let ctx: MetalContext

    private struct Layer {
        let ln1w, ln1b, qkvW, qkvB, outW, outB: MTLBuffer
        let ln2w, ln2b, upW, upB, downW, downB: MTLBuffer
    }
    private let blocks: [Layer]
    private let patchW: MTLBuffer      // summed dual temporal conv, half
    private let posBias: MTLBuffer     // patch bias + reordered pos embed
    private let postW, postB: MTLBuffer
    private let mm0W, mm0B, mm2W, mm2B: MTLBuffer
    private let ropeCos, ropeSin: MTLBuffer
    private let mergeOrder: [Int]
    // Per-tile activation scratch, sized once from the tower's grid.
    private let bRows, bX, bNorm, bQkv, bCtx, bTmp, bFF, bOut: MTLBuffer

    public init(path: String) throws {
        let g = try GGUF(path: path)
        let c = ViTConfig(g)
        let context = try MetalContext(g)
        try context.prewarm()
        let dev = context.device
        func f32(_ name: String) -> MTLBuffer {
            context.makeF32(Dense.floats(g.tensor(name)))
        }
        func f16(_ name: String) -> MTLBuffer {
            MetalViT.halfBuf(dev, Dense.floats(g.tensor(name)))
        }
        // The two temporal conv kernels sum for a still image (llama.cpp
        // adds their outputs, and conv is linear).
        let w0 = Dense.floats(g.tensor("v.patch_embd.weight"))
        let w1 = Dense.floats(g.tensor("v.patch_embd.weight.1"))
        var sum = w0
        vDSP_vadd(w0, 1, w1, 1, &sum, 1, vDSP_Length(w0.count))
        let order = ViT.mergeOrderTable(c)
        let pb = Dense.floats(g.tensor("v.patch_embd.bias"))
        let pe = Dense.floats(g.tensor("v.position_embd.weight"))
        var combined = [Float](repeating: 0, count: c.patches * c.embd)
        for s in 0 ..< c.patches {
            for i in 0 ..< c.embd {
                combined[s * c.embd + i] = pb[i] + pe[order[s] * c.embd + i]
            }
        }
        let tables = ViT.ropeTables(c, order)
        gguf = g
        cfg = c
        ctx = context
        patchW = MetalViT.halfBuf(dev, sum)
        posBias = context.makeF32(combined)
        postW = f32("v.post_ln.weight")
        postB = f32("v.post_ln.bias")
        mm0W = f16("mm.0.weight")
        mm0B = f32("mm.0.bias")
        mm2W = f16("mm.2.weight")
        mm2B = f32("mm.2.bias")
        blocks = (0 ..< c.layers).map { il in
            Layer(ln1w: f32("v.blk.\(il).ln1.weight"),
                  ln1b: f32("v.blk.\(il).ln1.bias"),
                  qkvW: f16("v.blk.\(il).attn_qkv.weight"),
                  qkvB: f32("v.blk.\(il).attn_qkv.bias"),
                  outW: f16("v.blk.\(il).attn_out.weight"),
                  outB: f32("v.blk.\(il).attn_out.bias"),
                  ln2w: f32("v.blk.\(il).ln2.weight"),
                  ln2b: f32("v.blk.\(il).ln2.bias"),
                  upW: f16("v.blk.\(il).ffn_up.weight"),
                  upB: f32("v.blk.\(il).ffn_up.bias"),
                  downW: f16("v.blk.\(il).ffn_down.weight"),
                  downB: f32("v.blk.\(il).ffn_down.bias"))
        }
        ropeCos = context.makeF32(tables.cos)
        ropeSin = context.makeF32(tables.sin)
        mergeOrder = order
        let n = c.patches
        bRows = context.makeF32(n * c.patchDim)
        bX = context.makeF32(n * c.embd)
        bNorm = context.makeF32(n * c.embd)
        bQkv = context.makeF32(n * 3 * c.embd)
        bCtx = context.makeF32(n * c.embd)
        bTmp = context.makeF32(n * c.embd)
        bFF = context.makeF32(n * c.ff)
        bOut = context.makeF32(c.mergedTokens * c.projDim)
    }

    // Normalized pixels ([imageSize * imageSize * 3] HWC, already
    // (v - mean) / std) -> merged features [mergedTokens * projDim] f32 --
    // the CPU ViT.forward contract.
    public func forward(pixels: [Float]) -> [Float] {
        let n = cfg.patches
        let embd = cfg.embd
        let hd = cfg.headDim
        let scale = 1 / Float(hd).squareRoot()
        let rows = ViT.patchRows(pixels, cfg, mergeOrder)
        rows.withUnsafeBytes { src in
            _ = memcpy(bRows.contents(), src.baseAddress!, src.count)
        }
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        let f = MetalEnc(ctx: ctx, e: enc)
        f.f16Gemm(patchW, X: bRows, out: bX, K: cfg.patchDim, M: embd, N: n)
        f.add(x: bX, y: posBias, n: n * embd)
        for l in blocks {
            f.layerNorm(x: bX, w: l.ln1w, b: l.ln1b, y: bNorm, n: embd,
                        rows: n, eps: cfg.eps)
            f.f16Gemm(l.qkvW, X: bNorm, out: bQkv, K: embd, M: 3 * embd, N: n)
            f.addBiasRows(x: bQkv, bias: l.qkvB, m: 3 * embd, rows: n)
            f.visionRope(x: bQkv, cos: ropeCos, sin: ropeSin,
                         rowStride: 3 * embd, off: 0, headDim: hd,
                         nHead: cfg.heads, N: n)
            f.visionRope(x: bQkv, cos: ropeCos, sin: ropeSin,
                         rowStride: 3 * embd, off: embd, headDim: hd,
                         nHead: cfg.heads, N: n)
            f.visionAttn(qkv: bQkv, out: bCtx, n: n, embd: embd, hd: hd,
                         nHead: cfg.heads, scale: scale)
            f.f16Gemm(l.outW, X: bCtx, out: bTmp, K: embd, M: embd, N: n)
            f.addBiasRows(x: bTmp, bias: l.outB, m: embd, rows: n)
            f.add(x: bX, y: bTmp, n: n * embd)
            f.layerNorm(x: bX, w: l.ln2w, b: l.ln2b, y: bNorm, n: embd,
                        rows: n, eps: cfg.eps)
            f.f16Gemm(l.upW, X: bNorm, out: bFF, K: embd, M: cfg.ff, N: n)
            f.addBiasRows(x: bFF, bias: l.upB, m: cfg.ff, rows: n)
            f.gelu(x: bFF, n: n * cfg.ff)
            f.f16Gemm(l.downW, X: bFF, out: bTmp, K: cfg.ff, M: embd, N: n)
            f.addBiasRows(x: bTmp, bias: l.downB, m: embd, rows: n)
            f.add(x: bX, y: bTmp, n: n * embd)
        }
        f.layerNorm(x: bX, w: postW, b: postB, y: bNorm, n: embd, rows: n,
                    eps: cfg.eps)
        // The 2x2 merger's concat is a pure reshape: 4 consecutive
        // merge-ordered rows of bNorm ARE the contiguous [n/4, 4e] input.
        let m2 = cfg.merge * cfg.merge
        let mIn = embd * m2
        let mTok = n / m2
        f.f16Gemm(mm0W, X: bNorm, out: bTmp, K: mIn, M: mIn, N: mTok)
        f.addBiasRows(x: bTmp, bias: mm0B, m: mIn, rows: mTok)
        f.gelu(x: bTmp, n: mTok * mIn)
        f.f16Gemm(mm2W, X: bTmp, out: bOut, K: mIn, M: cfg.projDim, N: mTok)
        f.addBiasRows(x: bOut, bias: mm2B, m: cfg.projDim, rows: mTok)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return Array(bOut.f32(mTok * cfg.projDim))
    }

    // f32 -> half into a fresh shared buffer, one vectorized vImage pass
    // (the load-time weight conversion; a scalar loop over ~250M params is
    // seconds, this is milliseconds).
    private static func halfBuf(_ dev: MTLDevice, _ v: [Float]) -> MTLBuffer {
        let buf = dev.makeBuffer(length: v.count * 2,
                                 options: .storageModeShared)!
        v.withUnsafeBufferPointer { src in
            var s = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: src.baseAddress!),
                height: 1, width: vImagePixelCount(v.count),
                rowBytes: v.count * 4)
            var d = vImage_Buffer(data: buf.contents(), height: 1,
                                  width: vImagePixelCount(v.count),
                                  rowBytes: v.count * 2)
            _ = vImageConvert_PlanarFtoPlanar16F(&s, &d, 0)
        }
        return buf
    }
}

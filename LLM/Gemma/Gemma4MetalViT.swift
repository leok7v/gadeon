// The gemma-4 vision tower on the GPU, mirroring Gemma4ViT op for op with the
// CPU tower as the oracle. It reuses the TEXT tower's norm kernels rather than
// the vit_* ones next door, because gemma's encoder block IS gemma's text
// block: sandwich RMS norms over BF16 weights, a gated MLP, q/k norms and the
// scale-free v_norm.
//
// Weights stay QUANTIZED in the mapped file and expand in registers inside the
// GEMM, the same way the text tower reads its own: dequantizing this tower to
// resident f16 would cost 336 MB for 0.167 B parameters and buy nothing.
//
// The patch embedding, the 2-D rope tables, pooling and the projection are the
// CPU tower's own methods: all four are index gymnastics over a lookup table
// or one pass over a few hundred rows, so sharing them keeps the (x, y)
// lookup and the drop-empty-slot rule defined ONCE rather than twice.
import Foundation
import Metal

public final class Gemma4MetalViT {
    public let cfg: Gemma4VisionConfig
    private let cpu: Gemma4ViT
    private let ctx: MetalContext

    private let model: Gemma4Model
    private let map: UnsafeRawPointer
    private let scales: [String: SRQ]

    // The context is the TEXT engine's: this tower's weights live in the same
    // mapped file, and a context of its own would cut a second set of windows
    // over the identical bytes.

    public init(_ model: Gemma4Model, ctx: MetalContext) throws {
        let g = model.gguf
        cfg = try Gemma4VisionConfig(g)
        cpu = try Gemma4ViT(model)
        self.ctx = ctx
        self.model = model
        map = g.map
        var found: [String: SRQ] = [:]
        for il in 0..<cfg.layers {
            for tag in ["attn_q", "attn_k", "attn_v", "attn_out",
                        "ffn_gate", "ffn_up", "ffn_down"] {
                let name = "v.blk.\(il).\(tag).weight"
                found[name] = SRQ(g, name)
            }
        }
        scales = found
    }

    private func srq(_ w: GGUFTensor) -> SRQ { scales[w.name] ?? SRQ.none }

    private func off(_ t: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(t.base - map))
    }

    // Same contract as Gemma4ViT.forward: [0,1] pixels straight from the
    // processor plus the per-patch (x, y), returning the tower output and its
    // projection, both stripped to the slots real patches reached.
    public func forward(pixels: [Float], pos: [(Int, Int)])
        -> (tower: [Float], proj: [Float], count: Int) {
        let n = pos.count
        let e = cfg.embd
        let hd = cfg.headDim
        let nH = cfg.heads
        let wide = nH * hd
        let padded = pos.map { p in p.0 < 0 && p.1 < 0 }
        let rope = cpu.ropeForGPU(pos: pos)

        let bX = ctx.makeF32(cpu.embedForGPU(pixels: pixels, pos: pos))
        let bNorm = ctx.makeF32(n * e)
        let bTmp = ctx.makeF32(n * e)
        let bQ = ctx.makeF32(n * wide)
        let bK = ctx.makeF32(n * wide)
        let bV = ctx.makeF32(n * wide)
        let bCtx = ctx.makeF32(n * wide)
        let bFF = ctx.makeF32(n * cfg.ff)
        let bUp = ctx.makeF32(n * cfg.ff)
        let bCos = ctx.makeF32(rope.cos)
        let bSin = ctx.makeF32(rope.sin)
        let bMask = ctx.makeF32(padded.map { p in p ? Float(1) : Float(0) })
        let bClamp = ctx.makeF32(n * max(e, cfg.ff))

        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        let f = MetalEnc(ctx: ctx, e: enc)
        for il in 0..<cfg.layers {
            func t(_ n: String) -> GGUFTensor {
                model.gguf.tensor("v.blk.\(il).\(n)")
            }
            f.rmsnormBatchBF16(x: bX, weightOff: off(t("ln1.weight")),
                               y: bNorm, n: e, rows: n, eps: cfg.eps)
            f.linear(t("attn_q.weight"), X: bNorm, out: bQ,
                     off: off(t("attn_q.weight")), N: n,
                     srq: srq(t("attn_q.weight")), scratch: bClamp)
            f.linear(t("attn_k.weight"), X: bNorm, out: bK,
                     off: off(t("attn_k.weight")), N: n,
                     srq: srq(t("attn_k.weight")), scratch: bClamp)
            f.linear(t("attn_v.weight"), X: bNorm, out: bV,
                     off: off(t("attn_v.weight")), N: n,
                     srq: srq(t("attn_v.weight")), scratch: bClamp)
            f.rmsnormRowsBF16(x: bQ, xoff: 0, d: hd, rows: n * nH,
                              weightOff: off(t("attn_q_norm.weight")),
                              eps: cfg.eps)
            f.rmsnormRowsBF16(x: bK, xoff: 0, d: hd, rows: n * nH,
                              weightOff: off(t("attn_k_norm.weight")),
                              eps: cfg.eps)
            f.rmsnormRowsNoWeight(x: bV, xoff: 0, d: hd, rows: n * nH,
                                  eps: cfg.eps)
            f.gemmaVisionRope(x: bQ, cos: bCos, sin: bSin, rowStride: wide,
                              headDim: hd, nHead: nH, N: n)
            f.gemmaVisionRope(x: bK, cos: bCos, sin: bSin, rowStride: wide,
                              headDim: hd, nHead: nH, N: n)
            f.gemmaVisionAttn(q: bQ, k: bK, v: bV, mask: bMask, out: bCtx,
                              n: n, hd: hd, nHead: nH)
            f.linear(t("attn_out.weight"), X: bCtx, out: bTmp,
                     off: off(t("attn_out.weight")), N: n,
                     srq: srq(t("attn_out.weight")), scratch: bClamp)
            f.rmsnormBatchBF16(x: bTmp,
                               weightOff: off(t("post_attn_norm.weight")),
                               y: bNorm, n: e, rows: n, eps: cfg.eps)
            f.add(x: bX, y: bNorm, n: n * e)

            f.rmsnormBatchBF16(x: bX, weightOff: off(t("ln2.weight")),
                               y: bNorm, n: e, rows: n, eps: cfg.eps)
            f.linear(t("ffn_gate.weight"), X: bNorm, out: bFF,
                     off: off(t("ffn_gate.weight")), N: n,
                     srq: srq(t("ffn_gate.weight")), scratch: bClamp)
            f.linear(t("ffn_up.weight"), X: bNorm, out: bUp,
                     off: off(t("ffn_up.weight")), N: n,
                     srq: srq(t("ffn_up.weight")), scratch: bClamp)
            f.activateMul(cfg.activation, a: bFF, b: bUp, n: n * cfg.ff)
            f.linear(t("ffn_down.weight"), X: bFF, out: bTmp,
                     off: off(t("ffn_down.weight")), N: n,
                     srq: srq(t("ffn_down.weight")), scratch: bClamp)
            f.rmsnormBatchBF16(x: bTmp,
                               weightOff: off(t("post_ffn_norm.weight")),
                               y: bNorm, n: e, rows: n, eps: cfg.eps)
            f.add(x: bX, y: bNorm, n: n * e)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fatalError("metal gemma vit: \(err)") }
        return cpu.poolAndProject(Array(bX.f32(n * e)), pos: pos,
                                  padded: padded)
    }

}

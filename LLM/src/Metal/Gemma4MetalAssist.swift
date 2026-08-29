import Foundation
import Metal

final class Gemma4MetalAssist {
    private let w: Gemma4Assist
    private let cfg: Gemma4Config
    private let ctx: MetalContext
    private let map: UnsafeRawPointer
    private let tokEmbd: GGUFTensor
    private var scales: [String: SRQ] = [:]

    private struct Norms {
        let attn, postAttn, ffn, postFfn, qNorm: WeightRef
    }
    private var norms: [Norms] = []
    private let outNormOff: WeightRef

    private let bCat, bx, bNormed, bContrib: MTLBuffer
    private let bQ, bAttnOut, bGateNull: MTLBuffer
    private let bFfnGate, bFfnUp, bClamp: MTLBuffer
    let bLogits: MTLBuffer
    let bBack: MTLBuffer

    let slidingSource: Int
    let fullSource: Int

    init(_ model: Gemma4Model, _ head: Gemma4Assist, ctx: MetalContext) {
        w = head
        cfg = model.cfg
        self.ctx = ctx
        map = model.gguf.map
        tokEmbd = model.tokEmbd
        var lastSliding = -1
        var lastFull = -1
        for il in 0..<cfg.nLayer where !cfg.isShared(il) {
            if cfg.isFull(il) { lastFull = il } else { lastSliding = il }
        }
        slidingSource = lastSliding
        fullSource = lastFull
        let maxHd = max(cfg.headDimSliding, cfg.headDimFull)
        let maxFF = head.layers.map { L in L.nFF }.max() ?? 0
        bCat = ctx.makeF32(2 * head.backbone)
        bx = ctx.makeF32(head.nEmbd)
        bNormed = ctx.makeF32(head.nEmbd)
        bContrib = ctx.makeF32(head.nEmbd)
        bQ = ctx.makeF32(maxHd * cfg.nHead)
        bAttnOut = ctx.makeF32(maxHd * cfg.nHead)
        bGateNull = ctx.makeF32(maxHd * cfg.nHead)
        bFfnGate = ctx.makeF32(maxFF)
        bFfnUp = ctx.makeF32(maxFF)
        bClamp = ctx.makeF32(max(maxFF, head.nEmbd))
        bLogits = ctx.makeF32(cfg.nVocab)
        bBack = ctx.makeF32(head.backbone)
        let g = model.gguf
        for L in head.layers {
            for t in [L.wq, L.wo, L.ffnGate, L.ffnUp, L.ffnDown] {
                scales[t.name] = SRQ(g, t.name)
            }
        }
        scales[head.output.name] = SRQ(g, head.output.name)
        scales[head.preProj.name] = SRQ(g, head.preProj.name)
        scales[head.postProj.name] = SRQ(g, head.postProj.name)
        let context = ctx
        func normOff(_ t: GGUFTensor) -> WeightRef {
            precondition(t.type == .bf16,
                         "assist norm \(t.name) is \(t.type), expected bf16")
            return context.window(UInt64(t.base - g.map))
        }
        outNormOff = normOff(head.outputNorm)
        for L in head.layers {
            norms.append(Norms(attn: normOff(L.attnNorm),
                               postAttn: normOff(L.postAttnNorm),
                               ffn: normOff(L.ffnNorm),
                               postFfn: normOff(L.postFfnNorm),
                               qNorm: normOff(L.qNorm)))
        }
    }

    private func off(_ t: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(t.base - map))
    }

    private func srq(_ t: GGUFTensor) -> SRQ { scales[t.name] ?? SRQ.none }

    var backboneWidth: Int { w.backbone }

    func seed(_ token: Int, hidden: UnsafePointer<Float>) {
        let dst = bCat.f32(2 * w.backbone).baseAddress!
        GQ.gather(tokEmbd, row: token, from: 0, count: w.backbone, into: dst)
        for i in 0..<w.backbone { dst[i] *= cfg.embedScale }
        memcpy(dst + w.backbone, hidden,
               w.backbone * MemoryLayout<Float>.stride)
    }

    func encodeStep(_ f: MetalEnc, pos: Int, pools: [Int: MetalKVPool]) {
        f.linear(w.preProj, x: bCat, out: bx, off: off(w.preProj),
                 srq: srq(w.preProj), scratch: bClamp)
        for il in 0..<w.nLayer { layer(f, il, pos: pos, pools: pools) }
        f.rmsnormBF16(x: bx, weightOff: outNormOff, out: bNormed,
                      n: w.nEmbd, eps: cfg.eps)
        f.linear(w.output, x: bNormed, out: bLogits, off: off(w.output),
                 srq: srq(w.output), scratch: bClamp)
        f.linear(w.postProj, x: bNormed, out: bBack, off: off(w.postProj),
                 srq: srq(w.postProj), scratch: bClamp)
    }

    private func layer(_ f: MetalEnc, _ il: Int, pos: Int,
                       pools: [Int: MetalKVPool]) {
        let L = w.layers[il]
        let isFull = w.isFull(il)
        let hd = isFull ? cfg.headDimFull : cfg.headDimSliding
        let nKV = isFull ? cfg.nHeadKVFull : cfg.nHeadKV
        let rot = isFull ? cfg.rotatedPairsFull : cfg.rotatedPairsSliding
        let base = isFull ? cfg.ropeBaseFull : cfg.ropeBaseSliding
        let pool = pools[isFull ? fullSource : slidingSource]!
        f.rmsnormBF16(x: bx, weightOff: norms[il].attn, out: bNormed,
                      n: w.nEmbd, eps: cfg.eps)
        f.linear(L.wq, x: bNormed, out: bQ, off: off(L.wq), srq: srq(L.wq),
                 scratch: bClamp)
        f.rmsnormRowsBF16(x: bQ, xoff: 0, d: hd, rows: cfg.nHead,
                          weightOff: norms[il].qNorm, eps: cfg.eps)
        f.ropeGemma(x: bQ, headDim: hd, nHead: cfg.nHead, rotated: rot,
                    base: base, pos: pos)
        let lo = isFull ? 0 : max(0, pos - cfg.slidingWindow + 1)
        f.attnPaged(q: bQ, kAddr: pool.kAddr, vAddr: pool.vAddr,
                    pages: pool.residentPages, gate: bGateNull,
                    out: bAttnOut, hd: hd, nH: cfg.nHead, nKV: nKV,
                    T: pos + 1, kvDim: hd * nKV, P: pool.P, scale: 1,
                    gated: 0, lo: lo)
        f.linear(L.wo, x: bAttnOut, out: bContrib, off: off(L.wo),
                 srq: srq(L.wo), scratch: bClamp)
        f.rmsnormBF16(x: bContrib, weightOff: norms[il].postAttn,
                      out: bNormed, n: w.nEmbd, eps: cfg.eps)
        f.add(x: bx, y: bNormed, n: w.nEmbd)
        f.rmsnormBF16(x: bx, weightOff: norms[il].ffn, out: bNormed,
                      n: w.nEmbd, eps: cfg.eps)
        f.linear(L.ffnGate, x: bNormed, out: bFfnGate, off: off(L.ffnGate),
                 srq: srq(L.ffnGate), scratch: bClamp)
        f.linear(L.ffnUp, x: bNormed, out: bFfnUp, off: off(L.ffnUp),
                 srq: srq(L.ffnUp), scratch: bClamp)
        f.activateMul(cfg.activation, a: bFfnGate, b: bFfnUp, n: L.nFF)
        f.linear(L.ffnDown, x: bFfnGate, out: bContrib, off: off(L.ffnDown),
                 srq: srq(L.ffnDown), scratch: bClamp)
        f.rmsnormBF16(x: bContrib, weightOff: norms[il].postFfn,
                      out: bNormed, n: w.nEmbd, eps: cfg.eps)
        f.add(x: bx, y: bNormed, n: w.nEmbd)
        f.scaleInPlace(x: bx, n: w.nEmbd, s: L.layerScalar)
    }

    func logits() -> [Float] { Array(bLogits.f32(cfg.nVocab)) }

    func backbone() -> [Float] { Array(bBack.f32(w.backbone)) }
}

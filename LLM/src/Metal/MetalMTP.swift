// The nextn drafter block on the GPU, with its own KV pool: folds a token
// onto the base hidden that predicted it and leaves h_nextn for the tied
// lm_head. Mirrors llama.cpp qwen35.cpp graph_mtp.
import Foundation
import Metal

final class MetalMTP {
    private let w: BonsaiMTP
    private let cfg: BonsaiConfig
    private let ctx: MetalContext
    private let map: UnsafeRawPointer
    private let tokEmbd: GGUFTensor
    private let head: GGUFTensor
    private let pool: MetalKVPool

    private let bEmbed, bCat, bCur, bNormed, bContrib: MTLBuffer
    private let bQFull, bQ, bGate, bK1, bV1, bAttnOut: MTLBuffer
    private let bFfnGate, bFfnUp: MTLBuffer
    let bHidden: MTLBuffer
    let bLogits: MTLBuffer

    init(_ model: BonsaiModel, _ w: BonsaiMTP, ctx: MetalContext, pageP: Int) {
        self.w = w
        self.ctx = ctx
        cfg = model.cfg
        map = model.gguf.map
        tokEmbd = model.tokEmbd
        head = model.output
        let c = model.cfg
        pool = MetalKVPool(device: ctx.device, P: pageP,
                           kvDim: c.headDim * c.nHeadKV)
        bEmbed = ctx.makeF32(c.nEmbd)
        bCat = ctx.makeF32(2 * c.nEmbd)
        bCur = ctx.makeF32(c.nEmbd)
        bNormed = ctx.makeF32(c.nEmbd)
        bContrib = ctx.makeF32(c.nEmbd)
        bQFull = ctx.makeF32(c.headDim * 2 * c.nHead)
        bQ = ctx.makeF32(c.headDim * c.nHead)
        bGate = ctx.makeF32(c.headDim * c.nHead)
        bK1 = ctx.makeF32(c.headDim * c.nHeadKV)
        bV1 = ctx.makeF32(c.headDim * c.nHeadKV)
        bAttnOut = ctx.makeF32(c.headDim * c.nHead)
        bFfnGate = ctx.makeF32(c.nFF)
        bFfnUp = ctx.makeF32(c.nFF)
        bHidden = ctx.makeF32(c.nEmbd)
        bLogits = ctx.makeF32(c.nVocab)
    }

    var cached: Int { pool.len }

    func reset() { pool.truncate(to: 0) }

    func truncate(to n: Int) { pool.truncate(to: n) }

    private func off(_ t: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(t.base - map))
    }

    func encodeStep(_ f: MetalEnc, token: Int, hidden: MTLBuffer,
                    hiddenOff: Int, ropePos: Int, head wantHead: Bool) {
        let c = cfg
        let rowBytes = GGUF.rowByteCount(tokEmbd.type, c.nEmbd)
        f.dequantRow(weightOff: off(tokEmbd) + UInt64(token * rowBytes),
                     out: bEmbed, n: c.nEmbd, type: tokEmbd.type)
        f.rmsnorm(x: bEmbed, weightOff: off(w.enorm), out: bCat, n: c.nEmbd,
                  eps: c.eps)
        f.rmsnorm(x: hidden, weightOff: off(w.hnorm), out: bCat, n: c.nEmbd,
                  eps: c.eps, xOff: hiddenOff,
                  outOff: c.nEmbd * MemoryLayout<Float>.stride)
        f.gemv(w.ehProj, x: bCat, out: bCur, off: off(w.ehProj))
        f.rmsnorm(x: bCur, weightOff: off(w.attnNorm), out: bNormed,
                  n: c.nEmbd, eps: c.eps)
        encodeAttn(f, ropePos: ropePos)
        f.add(x: bCur, y: bContrib, n: c.nEmbd)
        f.rmsnorm(x: bCur, weightOff: off(w.attnPostNorm), out: bNormed,
                  n: c.nEmbd, eps: c.eps)
        f.gemv(w.ffnGate, x: bNormed, out: bFfnGate, off: off(w.ffnGate))
        f.gemv(w.ffnUp, x: bNormed, out: bFfnUp, off: off(w.ffnUp))
        f.siluMul(a: bFfnGate, b: bFfnUp, n: c.nFF)
        f.gemv(w.ffnDown, x: bFfnGate, out: bContrib, off: off(w.ffnDown))
        f.add(x: bCur, y: bContrib, n: c.nEmbd)
        f.rmsnorm(x: bCur, weightOff: off(w.headNorm), out: bHidden,
                  n: c.nEmbd, eps: c.eps)
        if wantHead {
            f.gemv(head, x: bHidden, out: bLogits, off: off(head))
        }
    }

    private func encodeAttn(_ f: MetalEnc, ropePos: Int) {
        let c = cfg
        f.gemv(w.wq, x: bNormed, out: bQFull, off: off(w.wq))
        f.splitQGate(qFull: bQFull, q: bQ, gate: bGate, hd: c.headDim,
                     nH: c.nHead)
        f.gemv(w.wk, x: bNormed, out: bK1, off: off(w.wk))
        f.gemv(w.wv, x: bNormed, out: bV1, off: off(w.wv))
        f.rmsnormRows(x: bQ, xoff: 0, d: c.headDim, rows: c.nHead,
                      weightOff: off(w.qNorm), eps: c.eps)
        f.rmsnormRows(x: bK1, xoff: 0, d: c.headDim, rows: c.nHeadKV,
                      weightOff: off(w.kNorm), eps: c.eps)
        f.rope(x: bQ, headDim: c.headDim, nHead: c.nHead, nRot: c.nRot,
               base: c.ropeBase, pos: ropePos)
        f.rope(x: bK1, headDim: c.headDim, nHead: c.nHeadKV, nRot: c.nRot,
               base: c.ropeBase, pos: ropePos)
        let kvDim = c.headDim * c.nHeadKV
        let tail = pool.tailForAppend()
        f.kvAppend(kCur: bK1, vCur: bV1, K: tail.k, V: tail.v, kvDim: kvDim,
                   pos: tail.slot)
        pool.commitAppend()
        pool.refreshTable()
        f.attnPaged(q: bQ, kAddr: pool.kAddr, vAddr: pool.vAddr,
                    pages: pool.residentPages, gate: bGate, out: bAttnOut,
                    hd: c.headDim, nH: c.nHead, nKV: c.nHeadKV, T: pool.len,
                    kvDim: kvDim, P: pool.P,
                    scale: 1 / Float(c.headDim).squareRoot(), gated: 1)
        f.gemv(w.wo, x: bAttnOut, out: bContrib, off: off(w.wo))
    }
}

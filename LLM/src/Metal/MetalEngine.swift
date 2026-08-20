// The qwen35 forward loop on the GPU: the same 64-layer GGGA stack as the SIMD
// BonsaiEngine (embed -> per layer attn_norm -> GDN or attention -> residual ->
// post_norm -> FFN -> residual -> final norm -> lm_head), token by token,
// threading per-layer conv/rec/KV state in resident MTLBuffers. All kernels for
// one token are encoded on ONE command buffer (Metal hazard-tracks the
// dependent steps, so activation scratch is reused across layers), committed
// once, and only the logits are read back for the CPU sampler. Numerically it
// mirrors GDN.step / Attn.step / Kern.* exactly; the SIMD engine is the oracle.
import Foundation
import Metal

// Lock-backed stop flag (mirrors the CoreML Engine's StopSignal): the app raises
// it from the main actor while a synchronous Metal forward holds the ChatSession
// actor with no suspension point, and the prefill loops + decode gate poll it.
final class MetalStopSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var raised = false
    func raise() { lock.lock(); raised = true; lock.unlock() }
    func clear() { lock.lock(); raised = false; lock.unlock() }
    var raisedNow: Bool {
        lock.lock()
        defer { lock.unlock() }
        return raised
    }
}

// iOS aborts GPU work submitted while the app is backgrounded (there is no
// ANE-style background grace for Metal), so the Metal engine parks before every
// command-buffer commit until the app is active again. ONLY the GPU path polls
// this -- the CoreML/ANE Engine never commits a Metal buffer, so it keeps
// running in the background untouched. The app raises it from scenePhase on iOS;
// on macOS nothing raises it (background GPU is allowed there), so the wait is a
// no-op. Lock-backed like MetalStopSignal (Swift 6 strict concurrency forbids a
// bare mutable global).
public final class BackgroundGate: @unchecked Sendable {
    public static let shared = BackgroundGate()
    private let lock = NSLock()
    private var backgrounded = false

    // Set from the app's scenePhase (iOS): true when not .active.
    public func setBackgrounded(_ v: Bool) {
        lock.lock(); backgrounded = v; lock.unlock()
    }

    private var isBackgrounded: Bool {
        lock.lock(); defer { lock.unlock() }; return backgrounded
    }

    // Park the calling thread while the app is backgrounded, so no GPU submit
    // fires there. A 1s poll (mirrors the reference im.ai gate); returns at once
    // in the foreground, and always at once on macOS (the flag is never raised).
    func waitForForeground() {
        while isBackgrounded { Thread.sleep(forTimeInterval: 1) }
    }
}

public final class MetalEngine {
    let model: BonsaiModel
    let cfg: BonsaiConfig
    let ctx: MetalContext
    private let map: UnsafeRawPointer
    // Attention KV page size in positions (lazy pos-major page pool; context is
    // bounded only by memory, not a fixed ring).
    let pageP: Int

    public private(set) var pos = 0
    // Added to the sequence index for RoPE angles only (causality stays on
    // raw positions): 0 for text; a vision prefill compresses each image's
    // M-RoPE span, and text after it ropes at the compressed scalar --
    // mirroring the CoreML Engine's ropeShift.
    private(set) var ropeShift = 0
    var sampler: Sampler?
    // Lock-backed stop flag, mirroring the CoreML Engine. The synchronous Metal
    // forward blocks its caller with no suspension point, so AsyncStream/Task
    // cancellation does not reliably reach it; this nonisolated flag is the
    // reliable lever. The app raises it (requestStop) while a forward is in
    // flight; prefill polls it between chunks and the ChatSession decode loop
    // polls it via backend.shouldStop(). Cleared at the start of each extend.
    private let stopSignal = MetalStopSignal()
    public func requestStop() { stopSignal.raise() }
    public func shouldStop() -> Bool { stopSignal.raisedNow }

    // Resident per-layer state.
    private var gdnConv: [Int: MTLBuffer] = [:]   // [(dConv-1)*convDim]
    private var gdnRec: [Int: MTLBuffer] = [:]    // [nV*dS*dS]
    private var kvPool: [Int: MetalKVPool] = [:]  // lazy paged KV per attn layer

    private var mtp: MetalMTP?
    private var specN = 2
    private var bSpecPrev: MTLBuffer?
    private var specBatch: BatchScratch?
    private var specLogits: MTLBuffer?
    private var specIds: MTLBuffer?
    private var specPick: MTLBuffer?
    // The GDN rollback ring: `recSlots` copies of every recurrent layer's
    // state, `recSlot` naming the live one. A verify pass writes the state
    // after each token into the next slots, so accepting m is `recSlot += m`
    // and rejecting costs nothing at all. [gdn-ring]
    private var recSlots = 1
    private var recSlot = 0

    // Reusable activation scratch (sized to the largest layer need).
    private let bx, bNormed, bContrib: MTLBuffer
    private let bQkv, bConvOut, bZ, bO: MTLBuffer
    private let bBetaPre, bAlphaPre, bBeta, bG: MTLBuffer
    private let bQFull, bQ, bGate, bK1, bV1, bAttnOut: MTLBuffer
    private let bFfnGate, bFfnUp: MTLBuffer
    private let bLogits: MTLBuffer

    public init(_ model: BonsaiModel, pageP: Int = 512) throws {
        self.model = model
        cfg = model.cfg
        map = model.gguf.map
        self.pageP = pageP
        ctx = try MetalContext(model.gguf)
        try ctx.prewarm()
        let c = cfg
        bx = ctx.makeF32(c.nEmbd)
        bNormed = ctx.makeF32(c.nEmbd)
        bContrib = ctx.makeF32(c.nEmbd)
        bQkv = ctx.makeF32(c.convDim)
        bConvOut = ctx.makeF32(c.convDim)
        bZ = ctx.makeF32(c.valueDim)
        bO = ctx.makeF32(c.valueDim)
        bBetaPre = ctx.makeF32(c.nVHead)
        bAlphaPre = ctx.makeF32(c.nVHead)
        bBeta = ctx.makeF32(c.nVHead)
        bG = ctx.makeF32(c.nVHead)
        bQFull = ctx.makeF32(c.headDim * 2 * c.nHead)
        bQ = ctx.makeF32(c.headDim * c.nHead)
        bGate = ctx.makeF32(c.headDim * c.nHead)
        bK1 = ctx.makeF32(c.headDim * c.nHeadKV)
        bV1 = ctx.makeF32(c.headDim * c.nHeadKV)
        bAttnOut = ctx.makeF32(c.headDim * c.nHead)
        bFfnGate = ctx.makeF32(c.nFF)
        bFfnUp = ctx.makeF32(c.nFF)
        bLogits = ctx.makeF32(c.nVocab)
        for il in 0..<c.nLayer {
            if c.isRecurrent(il) {
                gdnConv[il] = ctx.makeF32(c.convDim * (c.dConv - 1))
                gdnRec[il] = ctx.makeF32(c.nVHead * c.dState * c.dState)
            } else {
                kvPool[il] = MetalKVPool(device: ctx.device, P: pageP,
                                         kvDim: c.headDim * c.nHeadKV)
            }
        }
    }

    // Fresh conversation: clear GDN conv/rec + drop the KV pages + position.
    public func reset() {
        pos = 0
        ropeShift = 0
        stopSignal.clear()
        for (_, b) in gdnConv { memset(b.contents(), 0, b.length) }
        for (_, b) in gdnRec { memset(b.contents(), 0, b.length) }
        for (_, p) in kvPool { p.truncate(to: 0) }
        recSlot = 0
        mtp?.reset()
    }

    private func off(_ t: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(t.base - map))
    }

    // Prefill `ids` onto the CURRENT state, returning the next-token prediction.
    // Only the LAST token needs logits; the rest just advance state (hidden only,
    // no lm_head).
    // Prefill `ids` onto the CURRENT state; only the LAST token runs the lm_head
    // (the rest advance state hidden-only).
    // Batched-GEMM prefill via the simdgroup-matrix q2_0_gemm_mm: MEASURED
    // pp 36.4 t/s on an idle M3 over the 512-token bench prompt vs
    // token-by-token's 7.8 t/s
    // (4.7x). ON by default now that the matrix-unit GEMM beats N GEMVs; disable
    // with LLM_BATCHED_PREFILL=0 for an A/B. (The earlier scalar 2D-tile
    // q2_0_gemm was 3.3x SLOWER -- the win is entirely the matrix-unit kernel.)
    static let batchedPrefill =
        ProcessInfo.processInfo.environment["LLM_BATCHED_PREFILL"] != "0"
    // The batched forward is built on the simdgroup-matrix GEMM, so it exists
    // only where the GPU has matrix units. An A13 (Apple6) falls back to the
    // token-by-token path: far slower to read a prompt, but every kernel it
    // encodes is plain SIMD, which is what makes the small ternary models
    // runnable on those phones at all.
    var batched: Bool { MetalEngine.batchedPrefill && ctx.matrixUnits }
    // Prefill chunk (tokens per batched forward). Each chunk re-streams all
    // weights once, so bigger = fewer streams; capped by activation memory + the
    // serial-over-N GDN scan. Tunable for the sweep; default set from it.
    static let prefillChunk =
        Int(ProcessInfo.processInfo.environment["LLM_PREFILL_CHUNK"] ?? "") ?? 512
    // Diagnostic: LLM_SKIP=ffn,scan,proj,attn omits a kernel group from the
    // batched forward AND the seq=1 decode forward, so a METAL_TIMING gpu=
    // drop attributes that group's cost (a per-kernel bisection; the forward
    // is numerically wrong while skipping). Decode needs it as much as
    // prefill: past a few hundred cached positions attention is the term that
    // grows with context, and nothing else says by how much.
    static let skip: Set<String> = Set(
        (ProcessInfo.processInfo.environment["LLM_SKIP"] ?? "")
            .split(separator: ",").map(String.init))

    public func extend(_ ids: [Int32]) -> Int32 {
        // A prior turn's Stop must not kill this one; the app raises it again if
        // the user stops during THIS prefill (prefillBatch / the loop poll it).
        stopSignal.clear()
        var out: Int32 = 0
        if ids.count > 1 && batched {
            // Prompt prefill: the batched forward streams each weight once per
            // chunk instead of once per token.
            out = prefillBatch(ids)
        } else {
            // Token-by-token: only the last token runs the lm_head. The stop
            // gate is in the loop condition (single-exit); MetalBackend.extend
            // then throws EngineError.stopped so a prefill Stop rolls the turn
            // back, matching the CoreML path.
            var i = 0
            while i < ids.count && !stopSignal.raisedNow {
                if i == ids.count - 1 {
                    out = pick(forwardLogits(token: Int(ids[i]), pos: pos))
                } else {
                    forward(token: Int(ids[i]), pos: pos)
                }
                pos += 1
                i += 1
            }
        }
        return out
    }

    public func chunkCost(_ ids: [Int32], from first: Int,
                          want: (Int, Int32, UnsafePointer<Float>) -> Void) {
        let c = cfg
        let n = ids.count
        reset()
        let b = BatchScratch(ctx: ctx, cfg: c, N: n)
        let idsBuf = ctx.device.makeBuffer(length: n * 4,
                                           options: .storageModeShared)!
        idsBuf.contents().withMemoryRebound(to: Int32.self, capacity: n) { p in
            for k in 0..<n { p[k] = ids[k] }
        }
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        let f = MetalEnc(ctx: ctx, e: e)
        f.embedBatch(ids: idsBuf, weightOff: off(model.tokEmbd), out: b.x,
                     nEmbd: c.nEmbd, N: n, type: model.tokEmbd.type)
        encodeChunk(f, b, N: n, basePos: 0, pos3: nil, head: false)
        e.endEncoding()
        commitTimed(cb, "ppl.trunk")
        let sub = 64
        let hidden = ctx.makeF32(sub * c.nEmbd)
        let logits = ctx.makeF32(sub * c.nVocab)
        var at = max(first - 1, 0)
        while at < n - 1 {
            let span = min(sub, n - 1 - at)
            let src = b.normed.contents()
                .advanced(by: at * c.nEmbd * MemoryLayout<Float>.stride)
            memcpy(hidden.contents(), src,
                   span * c.nEmbd * MemoryLayout<Float>.stride)
            let hb = ctx.queue.makeCommandBuffer()!
            let he = hb.makeComputeCommandEncoder()!
            MetalEnc(ctx: ctx, e: he).gemm(model.output, X: hidden,
                                           out: logits,
                                           off: off(model.output), N: span)
            he.endEncoding()
            commitTimed(hb, "ppl.head")
            let lp = logits.contents().assumingMemoryBound(to: Float.self)
            for j in 0..<span {
                want(at + j, ids[at + j + 1], lp + j * c.nVocab)
            }
            at += span
        }
        pos += n
    }

    public func step(_ token: Int32) -> [Float] {
        let out = forwardLogits(token: Int(token), pos: pos)
        pos += 1
        return out
    }

    public func decode(_ token: Int32) -> Int32 {
        let out = pick(forwardLogits(token: Int(token), pos: pos))
        pos += 1
        return out
    }

    // Batched prefill: process the prompt in chunks of N tokens, streaming each
    // weight ONCE per chunk (the GEMM) instead of once per token. The GDN conv/
    // scan recurrences run sequentially inside their batched kernels; norms,
    // gates, silu, rope, attention are gridded over N. Numerically matches the
    // token-by-token extend (validated against the SIMD engine). Returns the
    // next-token prediction after the last id.
    func prefillBatch(_ ids: [Int32], chunk c0: Int? = nil) -> Int32 {
        let chunk = c0 ?? MetalEngine.prefillChunk
        let c = cfg
        var out: Int32 = 0
        // One scratch set + ids buffer sized to the largest chunk, reused across
        // chunks: each chunk is its own committed+waited command buffer, so
        // there is no cross-chunk hazard, and a short final chunk (N<capN) just
        // uses a prefix of each buffer.
        let capN = min(chunk, ids.count)
        let b = BatchScratch(ctx: ctx, cfg: c, N: capN)
        let idsBuf = ctx.device.makeBuffer(length: capN * 4,
                                           options: .storageModeShared)!
        var i = 0
        while i < ids.count && !stopSignal.raisedNow {
            let end = min(i + chunk, ids.count)
            let N = end - i
            let basePos = pos
            idsBuf.contents().withMemoryRebound(to: Int32.self, capacity: N) {
                p in
                for k in 0..<N { p[k] = ids[i + k] }
            }
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            let f = MetalEnc(ctx: ctx, e: e)
            f.embedBatch(ids: idsBuf, weightOff: off(model.tokEmbd),
                         out: b.x, nEmbd: c.nEmbd, N: N,
                         type: model.tokEmbd.type)
            encodeChunk(f, b, N: N, basePos: basePos, pos3: nil)
            e.endEncoding()
            commitTimed(cb, "prefillBatch")
            pos += N
            if end == ids.count { out = pick(Array(bLogits.f32(c.nVocab))) }
            i = end
        }
        return out
    }

    // The layer stack + final norm + last-token lm_head for one prefill
    // chunk whose [N, nEmbd] hidden is already in b.x. pos3 nil = text
    // (sequential 1D rope at basePos + ropeShift); non-nil = per-token 3D
    // M-RoPE positions (an image span in the chunk).
    private func encodeChunk(_ f: MetalEnc, _ b: BatchScratch, N: Int,
                             basePos: Int, pos3: MTLBuffer?,
                             head: Bool = true,
                             ring: StateRing = .inPlace) {
        let c = cfg
        for il in 0..<c.nLayer {
            let L = model.layers[il]
            f.rmsnormBatch(x: b.x, weightOff: off(L.attnNorm), y: b.normed,
                           n: c.nEmbd, rows: N, eps: c.eps)
            if L.recurrent {
                gdnBatch(f, L, il, b, N: N, ring: ring)
            } else {
                attnBatchLayer(f, L, il, b, N: N, basePos: basePos,
                               pos3: pos3)
            }
            f.add(x: b.x, y: b.contrib, n: N * c.nEmbd)
            f.rmsnormBatch(x: b.x, weightOff: off(L.attnPostNorm),
                           y: b.normed, n: c.nEmbd, rows: N, eps: c.eps)
            if !MetalEngine.skip.contains("ffn") {
                f.gemm(L.ffnGate, X: b.normed, out: b.ffnGate,
                       off: off(L.ffnGate), N: N)
                f.gemm(L.ffnUp, X: b.normed, out: b.ffnUp,
                       off: off(L.ffnUp), N: N)
                f.siluMul(a: b.ffnGate, b: b.ffnUp, n: N * c.nFF)
                f.gemm(L.ffnDown, X: b.ffnGate, out: b.contrib,
                       off: off(L.ffnDown), N: N)
            }
            f.add(x: b.x, y: b.contrib, n: N * c.nEmbd)
        }
        f.rmsnormBatch(x: b.x, weightOff: off(model.outputNorm),
                       y: b.normed, n: c.nEmbd, rows: N, eps: c.eps)
        if head {
            f.gemv(model.output, x: b.normed, out: bLogits,
                   off: off(model.output),
                   xOff: (N - 1) * c.nEmbd * MemoryLayout<Float>.stride)
        }
    }

    // Vision prefill onto the CURRENT state: `feats` are the tower's merged
    // embeddings per image (flat [mergedRows * nEmbd] f32, already in LM
    // space), replacing the <|image_pad|> rows of the chunk's hidden --
    // built CPU-side (Q2_0 embed rows dequanted, feat rows memcpy'd), so no
    // splice kernel exists. Attention ropes with the 3D M-RoPE positions
    // from the shared planner; causality stays on raw positions; ropeShift
    // carries the compressed scalar into decode, like the CoreML engine.
    public func extendVision(_ ids: [Int32], feats: [[Float]],
                             starts: [Int], gridH: Int,
                             gridW: Int) -> Int32 {
        stopSignal.clear()
        let c = cfg
        let base = pos
        let mergedRows = (feats.first?.count ?? 0) / c.nEmbd
        let spans = starts.map { s in (start: s, gh: gridH, gw: gridW) }
        let plan = Engine.visionPositionsMulti(
            ids.count, spans, startScalar: base + ropeShift)
        ropeShift = plan.next - (base + ids.count)
        let chunk = MetalEngine.prefillChunk
        let capN = min(chunk, ids.count)
        let b = BatchScratch(ctx: ctx, cfg: c, N: capN)
        let pos3Buf = ctx.device.makeBuffer(length: capN * 3 * 4,
                                            options: .storageModeShared)!
        var out: Int32 = 0
        var i = 0
        while i < ids.count && !stopSignal.raisedNow {
            let end = min(i + chunk, ids.count)
            let N = end - i
            let basePos = pos
            let xp = b.x.contents().assumingMemoryBound(to: Float.self)
            for k in 0..<N {
                let g = i + k
                var img = -1
                for (j, s) in starts.enumerated()
                where g >= s && g < s + mergedRows { img = j }
                if img >= 0 {
                    feats[img].withUnsafeBufferPointer { fp in
                        _ = memcpy(xp + k * c.nEmbd,
                                   fp.baseAddress!
                                       + (g - starts[img]) * c.nEmbd,
                                   c.nEmbd * 4)
                    }
                } else {
                    QB.dequant(model.tokEmbd, row: Int(ids[g]),
                               count: c.nEmbd, into: xp + k * c.nEmbd)
                }
            }
            let pp = pos3Buf.contents().assumingMemoryBound(to: Int32.self)
            for k in 0..<N {
                let p = plan.pos[i + k]
                pp[k * 3] = p.0
                pp[k * 3 + 1] = p.1
                pp[k * 3 + 2] = p.2
            }
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            encodeChunk(MetalEnc(ctx: ctx, e: e), b, N: N, basePos: basePos,
                        pos3: pos3Buf)
            e.endEncoding()
            commitTimed(cb, "prefillVision")
            pos += N
            if end == ids.count { out = pick(Array(bLogits.f32(c.nVocab))) }
            i = end
        }
        return out
    }

    private func gdnBatch(_ f: MetalEnc, _ L: BonsaiLayer, _ il: Int,
                          _ b: BatchScratch, N: Int,
                          ring: StateRing = .inPlace) {
        let c = cfg
        let qkv = b.qkv
        let convOut = b.convOut
        let gate = b.g
        let beta = b.beta
        if !MetalEngine.skip.contains("proj") {
            f.gemm(L.wqkv!, X: b.normed, out: qkv, off: off(L.wqkv!), N: N)
            f.gemm(L.wqkvGate!, X: b.normed, out: b.z, off: off(L.wqkvGate!),
                   N: N)
            f.gemm(L.ssmBeta!, X: b.normed, out: b.betaPre, off: off(L.ssmBeta!),
                   N: N)
            f.gemm(L.ssmAlpha!, X: b.normed, out: b.alphaPre,
                   off: off(L.ssmAlpha!), N: N)
        }
        f.gdnGate(bPre: b.betaPre, aPre: b.alphaPre, dtOff: off(L.ssmDt!),
                  aOff: off(L.ssmA!), beta: beta, g: gate, nV: c.nVHead,
                  count: N * c.nVHead)
        f.gdnConvBatch(qkvMixN: qkv, convState: gdnConv[il]!,
                       cwOff: off(L.ssmConv1d!), outN: convOut,
                       convDim: c.convDim, dConv: c.dConv, N: N, ring: ring)
        // L2-norm q and k per head for all N tokens in ONE dispatch: within a
        // token, q|k are the contiguous first 2*keyDim of convOut (= 2*nKHead
        // rows of dState based at n*convDim; v follows, untouched). Batching
        // replaces the 2*N tiny per-token dispatches that made prefill
        // CPU-bound.
        f.l2normRowsBatch(x: convOut, d: c.dState, rowsPerTok: 2 * c.nKHead,
                          tokStride: c.convDim, tokens: N, eps: c.eps)
        let qScale = 1 / Float(c.dState).squareRoot()
        if !MetalEngine.skip.contains("scan") {
            f.gdnScanBatch(convOutN: convOut, keyDim: c.keyDim,
                           valueDim: c.valueDim, convDim: c.convDim, gN: gate,
                           betaN: beta, S: gdnRec[il]!, oN: b.o, nV: c.nVHead,
                           nK: c.nKHead, dS: c.dState, qScale: qScale, N: N,
                           ring: ring)
        }
        f.rmsnormRows(x: b.o, xoff: 0, d: c.dState, rows: N * c.nVHead,
                      weightOff: off(L.ssmNorm!), eps: c.eps)
        f.mulSilu(a: b.o, b: b.z, n: N * c.valueDim)
        f.gemm(L.ssmOut!, X: b.o, out: b.contrib, off: off(L.ssmOut!), N: N)
    }

    private func attnBatchLayer(_ f: MetalEnc, _ L: BonsaiLayer, _ il: Int,
                                _ b: BatchScratch, N: Int, basePos: Int,
                                pos3: MTLBuffer? = nil) {
        let c = cfg
        let kvDim = c.headDim * c.nHeadKV
        if c.dense {
            f.gemm(L.wq!, X: b.normed, out: b.q, off: off(L.wq!), N: N)
        } else {
            f.gemm(L.wq!, X: b.normed, out: b.qFull, off: off(L.wq!), N: N)
            f.splitQGateBatch(qFullN: b.qFull, qN: b.q, gateN: b.gate,
                              hd: c.headDim, nH: c.nHead, N: N)
        }
        f.gemm(L.wk!, X: b.normed, out: b.k1, off: off(L.wk!), N: N)
        f.gemm(L.wv!, X: b.normed, out: b.v1, off: off(L.wv!), N: N)
        f.rmsnormRows(x: b.q, xoff: 0, d: c.headDim, rows: N * c.nHead,
                      weightOff: off(L.qNorm!), eps: c.eps)
        f.rmsnormRows(x: b.k1, xoff: 0, d: c.headDim, rows: N * c.nHeadKV,
                      weightOff: off(L.kNorm!), eps: c.eps)
        if let pos3 {
            f.ropeMBatch(x: b.q, pos3: pos3, headDim: c.headDim,
                         nHead: c.nHead, nRot: c.nRot, base: c.ropeBase, N: N)
            f.ropeMBatch(x: b.k1, pos3: pos3, headDim: c.headDim,
                         nHead: c.nHeadKV, nRot: c.nRot, base: c.ropeBase,
                         N: N)
        } else {
            f.ropeBatch(x: b.q, headDim: c.headDim, nHead: c.nHead,
                        nRot: c.nRot, base: c.ropeBase,
                        basePos: basePos + ropeShift, N: N)
            f.ropeBatch(x: b.k1, headDim: c.headDim, nHead: c.nHeadKV,
                        nRot: c.nRot, base: c.ropeBase,
                        basePos: basePos + ropeShift, N: N)
        }
        let pool = kvPool[il]!
        pool.appendBatch(N)
        f.kvAppendBatch(kCurN: b.k1, vCurN: b.v1, kAddr: pool.kAddr,
                        vAddr: pool.vAddr, pages: pool.residentPages,
                        kvDim: kvDim, basePos: basePos, P: pool.P, N: N)
        if !MetalEngine.skip.contains("attn") {
            f.attnBatch(qN: b.q, kAddr: pool.kAddr, vAddr: pool.vAddr,
                        pages: pool.residentPages, gateN: b.gate,
                        outN: b.attnOut, hd: c.headDim, nH: c.nHead,
                        nKV: c.nHeadKV, kvDim: kvDim, P: pool.P,
                        scale: 1 / Float(c.headDim).squareRoot(),
                        basePos: basePos, N: N, gated: c.dense ? 0 : 1)
        }
        f.gemm(L.wo!, X: b.attnOut, out: b.contrib, off: off(L.wo!), N: N)
    }

    // A restorable snapshot of the whole generation state (position + GDN
    // conv/rec + the used span of each attention KV), for the ChatSession
    // mark/rewind + park/resume seam. Mirrors BonsaiEngine.Bookmark.
    // The GDN recurrence copied whole (cannot be paged), the attention KV shared
    // as append-only page snapshots (only a partial tail copies). The mark/rewind
    // + park/resume seam; mirrors BonsaiEngine.Bookmark.
    public struct Bookmark: @unchecked Sendable {
        let pos: Int
        let ropeShift: Int
        let recSlot: Int
        let conv: [Int: [Float]]
        let rec: [Int: [Float]]
        let kv: [Int: MetalKVPool.Snapshot]
    }

    public func bookmark() -> Bookmark {
        var conv: [Int: [Float]] = [:], rec: [Int: [Float]] = [:]
        var kv: [Int: MetalKVPool.Snapshot] = [:]
        for (il, b) in gdnConv { conv[il] = Array(b.f32(b.length / 4)) }
        for (il, b) in gdnRec { rec[il] = Array(b.f32(b.length / 4)) }
        for (il, p) in kvPool { kv[il] = p.snapshot() }
        return Bookmark(pos: pos, ropeShift: ropeShift, recSlot: recSlot,
                        conv: conv, rec: rec, kv: kv)
    }

    public func restore(_ b: Bookmark) {
        pos = b.pos
        ropeShift = b.ropeShift
        recSlot = b.recSlot
        for (il, a) in b.conv { copyIn(a, gdnConv[il]!) }
        for (il, a) in b.rec { copyIn(a, gdnRec[il]!) }
        for (il, s) in b.kv { kvPool[il]!.restore(s) }
    }

    // Byte-serialize a Bookmark for on-disk state persistence (the Metal
    // counterpart of the CoreML Engine.serialize): pos + ropeShift, the GDN
    // conv/rec arrays per layer, and each attention layer's KV compacted to
    // its used length. Little-endian i64 + raw f32 runs.

    public func serialize(_ b: Bookmark) -> Data {
        var out = Data()
        MetalEngine.putInt(&out, b.pos)
        MetalEngine.putInt(&out, b.ropeShift)
        for dict in [b.conv, b.rec] {
            MetalEngine.putInt(&out, dict.count)
            for il in dict.keys.sorted() {
                MetalEngine.putInt(&out, il)
                MetalEngine.putFloats(&out, dict[il]!)
            }
        }
        MetalEngine.putInt(&out, b.kv.count)
        for il in b.kv.keys.sorted() {
            let s = b.kv[il]!
            MetalEngine.putInt(&out, il)
            MetalEngine.putInt(&out, s.len)
            MetalEngine.putFloats(&out, compactPages(s.kPages, s.len))
            MetalEngine.putFloats(&out, compactPages(s.vPages, s.len))
        }
        return out
    }

    // Rebuild a Bookmark from serialize()'s bytes; restore() then makes it
    // live. Pages are re-chunked at this engine's pageP.

    public func deserialize(_ data: Data) -> Bookmark {
        let b = [UInt8](data)
        var p = 0
        let pos = MetalEngine.getInt(b, &p)
        let ropeShift = MetalEngine.getInt(b, &p)
        var conv: [Int: [Float]] = [:], rec: [Int: [Float]] = [:]
        for _ in 0 ..< MetalEngine.getInt(b, &p) {
            let il = MetalEngine.getInt(b, &p)
            conv[il] = MetalEngine.getFloats(b, &p)
        }
        for _ in 0 ..< MetalEngine.getInt(b, &p) {
            let il = MetalEngine.getInt(b, &p)
            rec[il] = MetalEngine.getFloats(b, &p)
        }
        var kv: [Int: MetalKVPool.Snapshot] = [:]
        for _ in 0 ..< MetalEngine.getInt(b, &p) {
            let il = MetalEngine.getInt(b, &p)
            let len = MetalEngine.getInt(b, &p)
            let k = MetalEngine.getFloats(b, &p)
            let v = MetalEngine.getFloats(b, &p)
            kv[il] = MetalKVPool.Snapshot(kPages: pages(from: k, len: len),
                                          vPages: pages(from: v, len: len),
                                          len: len)
        }
        return Bookmark(pos: pos, ropeShift: ropeShift, recSlot: recSlot,
                        conv: conv, rec: rec, kv: kv)
    }

    // The used span of a page list as one contiguous [len * kvDim] array. The
    // pages are half; the SERIALIZED form stays f32 so a state file keeps one
    // meaning regardless of what the cache stores. Widening here costs one
    // pass per park / precook, never per token.
    private func compactPages(_ pages: [MTLBuffer], _ len: Int) -> [Float] {
        let kvDim = cfg.headDim * cfg.nHeadKV
        var out = [Float](repeating: 0, count: len * kvDim)
        var r = 0
        while r < len {
            let span = min(pageP - r % pageP, len - r)
            let src = pages[r / pageP].contents()
                .assumingMemoryBound(to: Float16.self) + (r % pageP) * kvDim
            out.withUnsafeMutableBufferPointer { ob in
                for i in 0 ..< span * kvDim {
                    ob[r * kvDim + i] = Float(src[i])
                }
            }
            r += span
        }
        return out
    }

    // Re-chunk a contiguous [len * kvDim] array into pageP-sized half pages
    // (the tail page zero-padded), the shape Snapshot/restore expect.
    private func pages(from flat: [Float], len: Int) -> [MTLBuffer] {
        let kvDim = cfg.headDim * cfg.nHeadKV
        var out: [MTLBuffer] = []
        var r = 0
        while r < len {
            let span = min(pageP, len - r)
            let page = ctx.device.makeBuffer(
                length: pageP * kvDim * MemoryLayout<Float16>.stride,
                options: .storageModeShared)!
            memset(page.contents(), 0, page.length)
            let dst = page.contents().assumingMemoryBound(to: Float16.self)
            flat.withUnsafeBufferPointer { fp in
                for i in 0 ..< span * kvDim {
                    dst[i] = Float16(fp[r * kvDim + i])
                }
            }
            out.append(page)
            r += span
        }
        return out
    }

    private static func putInt(_ out: inout Data, _ v: Int) {
        var x = Int64(v).littleEndian
        withUnsafeBytes(of: &x) { raw in out.append(contentsOf: raw) }
    }

    private static func putFloats(_ out: inout Data, _ v: [Float]) {
        putInt(&out, v.count)
        v.withUnsafeBytes { raw in out.append(contentsOf: raw) }
    }

    private static func getInt(_ b: [UInt8], _ p: inout Int) -> Int {
        var x: Int64 = 0
        withUnsafeMutableBytes(of: &x) { dst in
            for i in 0 ..< 8 { dst[i] = b[p + i] }
        }
        p += 8
        return Int(Int64(littleEndian: x))
    }

    private static func getFloats(_ b: [UInt8], _ p: inout Int) -> [Float] {
        let n = getInt(b, &p)
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBytes { dst in
            for i in 0 ..< n * 4 { dst[i] = b[p + i] }
        }
        p += n * 4
        return out
    }

    private func copyIn(_ a: [Float], _ b: MTLBuffer) {
        a.withUnsafeBytes { raw in
            _ = memcpy(b.contents(), raw.baseAddress!, raw.count)
        }
    }

    // A greedy turn already has its answer from argmax_rows, so the row never
    // crosses to the host; a sampled one needs the whole distribution.
    private func pickRow(_ row: UnsafePointer<Float>, gpu: Int32) -> Int32 {
        var out = gpu
        if sampler != nil {
            out = pick(Array(UnsafeBufferPointer(start: row,
                                                 count: cfg.nVocab)))
        }
        return out
    }

    func pick(_ logits: [Float]) -> Int32 {
        var out: Int32
        if sampler != nil {
            var work = logits
            let picked = sampler!.sample(&work)
            sampler!.accept(picked)
            out = picked
        } else {
            out = Int32(argmax(logits))
        }
        return out
    }

    func argmax(_ v: [Float]) -> Int {
        var bi = 0; var bv = -Float.greatestFiniteMagnitude
        for i in 0..<v.count where v[i] > bv { bv = v[i]; bi = i }
        return bi
    }

    // One token at absolute position `pos`. Encodes every kernel onto a single
    // command buffer, runs it, and returns the buffer holding the post-final-
    // norm hidden (bNormed). Logits are a separate pass so only sampled tokens
    // pay the lm_head.
    // Debug knob: cap the number of layers to bisect a GPU hang (0 = embed +
    // final norm only). Full stack when .max.
    var maxLayers = Int.max
    // Env-gated per-token GPU-vs-wall timing to localize the perf bottleneck.
    static let timing = ProcessInfo.processInfo.environment["METAL_TIMING"] != nil

    // Encode one token's whole forward (embed -> layers -> final norm) onto the
    // shared encoder, leaving the post-final-norm hidden in bNormed. No commit --
    // the caller optionally appends the lm_head in the SAME command buffer so a
    // decoded token costs ONE dispatch stream + ONE GPU sync, not two.
    private func encodeForward(_ f: MetalEnc, token: Int, pos: Int) {
        let c = cfg
        let rowBytes = GGUF.rowByteCount(model.tokEmbd.type, c.nEmbd)
        f.dequantRow(weightOff: off(model.tokEmbd) + UInt64(token * rowBytes),
                     out: bx, n: c.nEmbd, type: model.tokEmbd.type)
        for il in 0..<min(c.nLayer, maxLayers) {
            encodeLayer(f, il, pos: pos)
        }
        f.rmsnorm(x: bx, weightOff: off(model.outputNorm), out: bNormed,
                  n: c.nEmbd, eps: c.eps)
    }

    private var imatrix: [String: MTLBuffer] = [:]
    private var imatrixWidth: [String: Int] = [:]

    public func imatrixSums() -> [String: [Float]] {
        var out: [String: [Float]] = [:]
        for (k, b) in imatrix { out[k] = Array(b.f32(imatrixWidth[k]!)) }
        return out
    }

    public func collectImatrix() {
        let c = cfg
        for il in 0..<c.nLayer {
            var sites = ["l\(il).in": c.nEmbd, "l\(il).post": c.nEmbd,
                         "l\(il).ffn_down": c.nFF]
            if c.isRecurrent(il) {
                sites["l\(il).ssm_out"] = c.valueDim
            } else {
                sites["l\(il).attn_out"] = c.headDim * c.nHead
            }
            for (name, n) in sites {
                let b = ctx.makeF32(n)
                memset(b.contents(), 0, b.length)
                imatrix[name] = b
                imatrixWidth[name] = n
            }
        }
    }

    private func tally(_ f: MetalEnc, _ key: String, _ src: MTLBuffer,
                       _ n: Int) {
        if let dst = imatrix[key] { f.accumSq(dst: dst, src: src, n: n) }
        if let h = hessian[key] { f.accumOuter(h: h, src: src, n: n) }
    }

    private var hessian: [String: MTLBuffer] = [:]

    static let wideHessian =
        ProcessInfo.processInfo.environment["LLM_HESS_WIDE"] == "1"

    public func collectHessians(from lo: Int, upto hi: Int) {
        let c = cfg
        for il in lo..<min(hi, c.nLayer) {
            var sites = ["in": c.nEmbd, "post": c.nEmbd]
            if MetalEngine.wideHessian { sites["ffn_down"] = c.nFF }
            if c.isRecurrent(il) {
                sites["ssm_out"] = c.valueDim
            } else {
                sites["attn_out"] = c.headDim * c.nHead
            }
            for (site, n) in sites {
                let b = ctx.device.makeBuffer(
                    length: n * n * MemoryLayout<Float>.stride,
                    options: .storageModeShared)!
                memset(b.contents(), 0, b.length)
                hessian["l\(il).\(site)"] = b
            }
        }
    }

    public func hessianNames() -> [String] { Array(hessian.keys).sorted() }

    public func hessianBytes(_ name: String) -> Data {
        let b = hessian[name]!
        return Data(bytes: b.contents(), count: b.length)
    }

    private func encodeLayer(_ f: MetalEnc, _ il: Int, pos: Int) {
        let c = cfg
        let L = model.layers[il]
        f.rmsnorm(x: bx, weightOff: off(L.attnNorm), out: bNormed,
                  n: c.nEmbd, eps: c.eps)
        tally(f, "l\(il).in", bNormed, c.nEmbd)
        if L.recurrent {
            gdnLayer(f, L, il)
        } else {
            attnLayer(f, L, il, pos: pos)
        }
        f.add(x: bx, y: bContrib, n: c.nEmbd)
        f.rmsnorm(x: bx, weightOff: off(L.attnPostNorm), out: bNormed,
                  n: c.nEmbd, eps: c.eps)
        tally(f, "l\(il).post", bNormed, c.nEmbd)
        if !MetalEngine.skip.contains("ffn") {
            f.gemv(L.ffnGate, x: bNormed, out: bFfnGate, off: off(L.ffnGate))
            f.gemv(L.ffnUp, x: bNormed, out: bFfnUp, off: off(L.ffnUp))
            f.siluMul(a: bFfnGate, b: bFfnUp, n: c.nFF)
            tally(f, "l\(il).ffn_down", bFfnGate, c.nFF)
            f.gemv(L.ffnDown, x: bFfnGate, out: bContrib, off: off(L.ffnDown))
        }
        f.add(x: bx, y: bContrib, n: c.nEmbd)
    }

    public func tapLayers(token: Int) -> [[Float]] {
        let c = cfg
        let rowBytes = GGUF.rowByteCount(model.tokEmbd.type, c.nEmbd)
        var out: [[Float]] = []
        let cb0 = ctx.queue.makeCommandBuffer()!
        let e0 = cb0.makeComputeCommandEncoder()!
        MetalEnc(ctx: ctx, e: e0).dequantRow(
            weightOff: off(model.tokEmbd) + UInt64(token * rowBytes),
            out: bx, n: c.nEmbd, type: model.tokEmbd.type)
        e0.endEncoding()
        commitTimed(cb0, "tap.embed")
        out.append(Array(bx.f32(c.nEmbd)))
        for il in 0..<c.nLayer {
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            encodeLayer(MetalEnc(ctx: ctx, e: e), il, pos: pos)
            e.endEncoding()
            commitTimed(cb, "tap.l\(il)")
            out.append(Array(bx.f32(c.nEmbd)))
        }
        pos += 1
        return out
    }

    // One token, hidden only (bNormed). Its own command buffer + sync -- used by
    // the op-by-op self-test; the runtime uses the fused forwardLogits.
    @discardableResult
    func forward(token: Int, pos: Int) -> MTLBuffer {
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        encodeForward(MetalEnc(ctx: ctx, e: e), token: token, pos: pos)
        e.endEncoding()
        commitTimed(cb, "forward")
        return bNormed
    }

    // One token forward WITH the lm_head folded into the same command buffer, so
    // the whole step is one commit + one wait, and returns the logits.
    func forwardLogits(token: Int, pos: Int) -> [Float] {
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        let f = MetalEnc(ctx: ctx, e: e)
        encodeForward(f, token: token, pos: pos)
        f.gemv(model.output, x: bNormed, out: bLogits, off: off(model.output))
        e.endEncoding()
        commitTimed(cb, "forward+head")
        return Array(bLogits.f32(cfg.nVocab))
    }

    private func commitTimed(_ cb: MTLCommandBuffer, _ tag: String) {
        // Hold the encoded buffer until the app is foreground: iOS aborts a GPU
        // submit made in the background. Every runtime decode / prefill commit
        // funnels through here, so this one gate covers them all.
        BackgroundGate.shared.waitForForeground()
        let t0 = Date()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fatalError("metal \(tag): \(err)") }
        if MetalEngine.timing {
            let wall = Date().timeIntervalSince(t0) * 1000
            let gpu = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            FileHandle.standardError.write(Data(
                "\(tag) gpu=\(Int(gpu))ms wall=\(Int(wall))ms\n".utf8))
        }
    }

    private func gdnLayer(_ f: MetalEnc, _ L: BonsaiLayer, _ il: Int) {
        let c = cfg
        f.gemv(L.wqkv!, x: bNormed, out: bQkv, off: off(L.wqkv!))
        f.gemv(L.wqkvGate!, x: bNormed, out: bZ, off: off(L.wqkvGate!))
        f.gemv(L.ssmBeta!, x: bNormed, out: bBetaPre, off: off(L.ssmBeta!))
        f.gemv(L.ssmAlpha!, x: bNormed, out: bAlphaPre, off: off(L.ssmAlpha!))
        f.gdnGate(bPre: bBetaPre, aPre: bAlphaPre, dtOff: off(L.ssmDt!),
                  aOff: off(L.ssmA!), beta: bBeta, g: bG, nV: c.nVHead,
                  count: c.nVHead)
        f.gdnConv(qkvMix: bQkv, convState: gdnConv[il]!,
                  cwOff: off(L.ssmConv1d!), out: bConvOut,
                  convDim: c.convDim, dConv: c.dConv)
        f.l2normRows(x: bConvOut, xoff: 0, d: c.dState, rows: c.nKHead,
                     eps: c.eps)
        f.l2normRows(x: bConvOut, xoff: UInt32(c.keyDim), d: c.dState,
                     rows: c.nKHead, eps: c.eps)
        let qScale = 1 / Float(c.dState).squareRoot()
        f.gdnScan(convOut: bConvOut, keyDim: c.keyDim, g: bG, beta: bBeta,
                  S: gdnRec[il]!, o: bO, nV: c.nVHead, nK: c.nKHead,
                  dS: c.dState, qScale: qScale)
        f.rmsnormRows(x: bO, xoff: 0, d: c.dState, rows: c.nVHead,
                      weightOff: off(L.ssmNorm!), eps: c.eps)
        f.mulSilu(a: bO, b: bZ, n: c.valueDim)
        tally(f, "l\(il).ssm_out", bO, c.valueDim)
        f.gemv(L.ssmOut!, x: bO, out: bContrib, off: off(L.ssmOut!))
    }

    private func attnLayer(_ f: MetalEnc, _ L: BonsaiLayer, _ il: Int,
                           pos: Int) {
        let c = cfg
        // Dense qwen3 projects q straight into bQ; the hybrid fuses q|gate in
        // wq and splits it (the gate feeds the attention output gate).
        if c.dense {
            f.gemv(L.wq!, x: bNormed, out: bQ, off: off(L.wq!))
        } else {
            f.gemv(L.wq!, x: bNormed, out: bQFull, off: off(L.wq!))
            f.splitQGate(qFull: bQFull, q: bQ, gate: bGate, hd: c.headDim,
                         nH: c.nHead)
        }
        f.gemv(L.wk!, x: bNormed, out: bK1, off: off(L.wk!))
        f.gemv(L.wv!, x: bNormed, out: bV1, off: off(L.wv!))
        f.rmsnormRows(x: bQ, xoff: 0, d: c.headDim, rows: c.nHead,
                      weightOff: off(L.qNorm!), eps: c.eps)
        f.rmsnormRows(x: bK1, xoff: 0, d: c.headDim, rows: c.nHeadKV,
                      weightOff: off(L.kNorm!), eps: c.eps)
        f.rope(x: bQ, headDim: c.headDim, nHead: c.nHead, nRot: c.nRot,
               base: c.ropeBase, pos: pos + ropeShift)
        f.rope(x: bK1, headDim: c.headDim, nHead: c.nHeadKV, nRot: c.nRot,
               base: c.ropeBase, pos: pos + ropeShift)
        let kvDim = c.headDim * c.nHeadKV
        let pool = kvPool[il]!
        // append this position into the tail page (slot = pos % P), then read
        // the whole pool via the bindless page table.
        let tail = pool.tailForAppend()
        f.kvAppend(kCur: bK1, vCur: bV1, K: tail.k, V: tail.v,
                   kvDim: kvDim, pos: tail.slot)
        pool.commitAppend()
        pool.refreshTable()
        if !MetalEngine.skip.contains("attn") {
            f.attnPaged(q: bQ, kAddr: pool.kAddr, vAddr: pool.vAddr,
                        pages: pool.residentPages, gate: bGate, out: bAttnOut,
                        hd: c.headDim, nH: c.nHead, nKV: c.nHeadKV,
                        T: pool.len, kvDim: kvDim, P: pool.P,
                        scale: 1 / Float(c.headDim).squareRoot(),
                        gated: c.dense ? 0 : 1)
        }
        tally(f, "l\(il).attn_out", bAttnOut, c.headDim * c.nHead)
        f.gemv(L.wo!, x: bAttnOut, out: bContrib, off: off(L.wo!))
    }

    public private(set) var specCycles = 0
    public private(set) var specCommitted = 0
    public private(set) var specDrafted = 0
    public private(set) var specAccepted = 0

    public var mtpReady: Bool { mtp != nil }

    public func loadMTP(drafts: Int = 2) {
        if let w = model.mtp, mtp == nil, drafts > 0 {
            let c = cfg
            let width = drafts + 1
            specN = drafts
            mtp = MetalMTP(model, w, ctx: ctx, pageP: pageP)
            bSpecPrev = ctx.makeF32(c.nEmbd)
            specBatch = BatchScratch(ctx: ctx, cfg: c, N: width)
            specLogits = ctx.makeF32(width * c.nVocab)
            specIds = ctx.device.makeBuffer(
                length: width * 4, options: .storageModeShared)
            specPick = ctx.makeU32(width)
            recSlots = width + 1
            recSlot = 0
            for il in 0..<c.nLayer where c.isRecurrent(il) {
                gdnConv[il] = ctx.makeF32(
                    recSlots * c.convDim * (c.dConv - 1))
                gdnRec[il] = ctx.makeF32(
                    recSlots * c.nVHead * c.dState * c.dState)
            }
            Diag.shared.report("[mtp] ON: metal drafter, n=\(drafts), "
                + "\(recSlots) state slots")
        }
    }

    // Feeds `token` plainly and keeps the hidden it produced, which is the
    // seed the first draft of the next cycle folds its token onto.
    public func specPrime(_ token: Int32) -> Int32 {
        let out = pick(forwardLogits(token: Int(token), pos: pos))
        if let prev = bSpecPrev {
            memcpy(prev.contents(), bNormed.contents(),
                   cfg.nEmbd * MemoryLayout<Float>.stride)
        }
        pos += 1
        return out
    }

    // One draft/verify/accept cycle: returns every token it committed, the
    // last of which is the bonus that seeds the next cycle.
    public func specDecode(_ token: Int32) -> [Int32] {
        var out: [Int32] = []
        if let d = mtp, let prev = bSpecPrev,
           let b = specBatch, let lg = specLogits, let idsBuf = specIds {
            out = specCycle(d, prev, b, lg, idsBuf, token)
        } else {
            out = [specPrime(token)]
        }
        return out
    }

    private func specCycle(_ d: MetalMTP,
                           _ prev: MTLBuffer, _ b: BatchScratch,
                           _ lg: MTLBuffer, _ idsBuf: MTLBuffer,
                           _ token: Int32) -> [Int32] {
        let c = cfg
        let p0 = pos
        let mtpBase = d.cached
        var fed: [Int32] = [token]
        var drafts: [Int32] = []
        var i = 0
        while i < specN {
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            let f = MetalEnc(ctx: ctx, e: e)
            d.encodeStep(f, token: Int(fed[i]),
                         hidden: i == 0 ? prev : d.bHidden, hiddenOff: 0,
                         ropePos: p0 + i, head: true)
            f.argmaxRows(x: d.bLogits, out: specPick!, n: c.nVocab, rows: 1)
            e.endEncoding()
            commitTimed(cb, "spec.draft")
            let dt = specPick!.contents()
                .assumingMemoryBound(to: Int32.self)[0]
            drafts.append(dt)
            fed.append(dt)
            i += 1
        }
        let width = fed.count
        idsBuf.contents().withMemoryRebound(to: Int32.self, capacity: width) {
            p in
            for k in 0..<width { p[k] = fed[k] }
        }
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        let f = MetalEnc(ctx: ctx, e: e)
        f.embedBatch(ids: idsBuf, weightOff: off(model.tokEmbd), out: b.x,
                     nEmbd: c.nEmbd, N: width, type: model.tokEmbd.type)
        encodeChunk(f, b, N: width, basePos: p0, pos3: nil, head: false,
                    ring: StateRing(slot0: recSlot, slots: recSlots))
        f.gemm(model.output, X: b.normed, out: lg, off: off(model.output),
               N: width)
        f.argmaxRows(x: lg, out: specPick!, n: c.nVocab, rows: width)
        e.endEncoding()
        commitTimed(cb, "spec.verify")
        let lp = lg.contents().assumingMemoryBound(to: Float.self)
        let gp = specPick!.contents().assumingMemoryBound(to: Int32.self)
        var accepted = 0
        var bonus: Int32 = 0
        var scanning = true
        while scanning {
            let r = pickRow(lp + accepted * c.nVocab, gpu: gp[accepted])
            if accepted < drafts.count && r == drafts[accepted] {
                accepted += 1
            } else {
                bonus = r
                scanning = false
            }
        }
        let m = accepted + 1
        recSlot = (recSlot + m) % recSlots
        if m < width {
            for (_, p) in kvPool { p.truncate(to: p0 + m) }
        }
        pos = p0 + m
        maintainMTP(d, prev, b, fed, from: mtpBase, at: p0, count: m)
        memcpy(prev.contents(),
               b.normed.contents().advanced(
                   by: (m - 1) * c.nEmbd * MemoryLayout<Float>.stride),
               c.nEmbd * MemoryLayout<Float>.stride)
        specCycles += 1
        specCommitted += m
        specDrafted += drafts.count
        specAccepted += accepted
        var out = Array(drafts.prefix(accepted))
        out.append(bonus)
        return out
    }

    // Rebuilds the drafter's KV over the committed tokens from the BASE
    // hiddens, replacing what drafting wrote from the drafter's own.
    private func maintainMTP(_ d: MetalMTP, _ prev: MTLBuffer,
                             _ b: BatchScratch, _ fed: [Int32], from base: Int,
                             at p0: Int, count m: Int) {
        let c = cfg
        d.truncate(to: base)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        let f = MetalEnc(ctx: ctx, e: e)
        var j = 0
        while j < m {
            let stride = MemoryLayout<Float>.stride
            d.encodeStep(f, token: Int(fed[j]),
                         hidden: j == 0 ? prev : b.normed,
                         hiddenOff: j == 0 ? 0 : (j - 1) * c.nEmbd * stride,
                         ropePos: p0 + j, head: false)
            j += 1
        }
        e.endEncoding()
        commitTimed(cb, "spec.maintain")
    }

    // logits = output @ hidden  (tied Q2_0 lm_head), returned to the CPU sampler.
    public func logits(_ hidden: MTLBuffer) -> [Float] {
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        MetalEnc(ctx: ctx, e: e).gemv(model.output, x: hidden, out: bLogits,
                                      off: off(model.output))
        e.endEncoding()
        let t0 = Date()
        cb.commit()
        cb.waitUntilCompleted()
        if MetalEngine.timing {
            let wall = Date().timeIntervalSince(t0) * 1000
            let gpu = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            FileHandle.standardError.write(Data(
                "logits gpu=\(Int(gpu))ms wall=\(Int(wall))ms\n".utf8))
        }
        return Array(bLogits.f32(cfg.nVocab))
    }
}


// Per-chunk batched activation buffers for prefillBatch (token-major [N, dim]).
private struct BatchScratch {
    let x, normed, contrib: MTLBuffer
    let qkv, z, betaPre, alphaPre, beta, g, convOut, o: MTLBuffer
    let qFull, q, gate, k1, v1, attnOut: MTLBuffer
    let ffnGate, ffnUp: MTLBuffer

    init(ctx: MetalContext, cfg c: BonsaiConfig, N: Int) {
        x = ctx.makeF32(N * c.nEmbd)
        normed = ctx.makeF32(N * c.nEmbd)
        contrib = ctx.makeF32(N * c.nEmbd)
        qkv = ctx.makeF32(N * c.convDim)
        z = ctx.makeF32(N * c.valueDim)
        betaPre = ctx.makeF32(N * c.nVHead)
        alphaPre = ctx.makeF32(N * c.nVHead)
        beta = ctx.makeF32(N * c.nVHead)
        g = ctx.makeF32(N * c.nVHead)
        convOut = ctx.makeF32(N * c.convDim)
        o = ctx.makeF32(N * c.valueDim)
        qFull = ctx.makeF32(N * c.headDim * 2 * c.nHead)
        q = ctx.makeF32(N * c.headDim * c.nHead)
        gate = ctx.makeF32(N * c.headDim * c.nHead)
        k1 = ctx.makeF32(N * c.headDim * c.nHeadKV)
        v1 = ctx.makeF32(N * c.headDim * c.nHeadKV)
        attnOut = ctx.makeF32(N * c.headDim * c.nHead)
        ffnGate = ctx.makeF32(N * c.nFF)
        ffnUp = ctx.makeF32(N * c.nFF)
    }
}

// Bring-up validation for the Metal backend, run before the full engine exists.
// Each check dispatches one GPU kernel and diffs its output against the
// pure-Swift SIMD reference on identical inputs, so a kernel is proven in
// isolation before it is wired into the forward loop. Driven by
// `gadeon-cli <model.gguf> --metal-selftest`.
import Foundation
import Metal

public enum MetalSelfTest {
    // Deterministic pseudo-random f32 in [-1, 1): a hashed LCG over the index so
    // the vector is reproducible run to run (no Date/random seeds).
    private static func fill(_ n: Int, seed: UInt64) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        var s = seed &+ 0x9E37_79B9_7F4A_7C15
        for i in 0..<n {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let bits = UInt32(truncatingIfNeeded: s >> 33)
            out[i] = Float(bits) / Float(UInt32.max) * 2 - 1
        }
        return out
    }

    private static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        var m: Float = 0
        for i in 0..<min(a.count, b.count) { m = max(m, abs(a[i] - b[i])) }
        return m
    }

    // Load the model, then run each isolated kernel check and return a report.
    public static func run(ggufPath: String) throws -> String {
        let model = try BonsaiModel(path: ggufPath)
        let ctx = try MetalContext(model.gguf)
        func step(_ s: String) { print(s); fflush(stdout) }
        // VERIFY the threadgroup budget headroom: the flash attention now asks
        // for a CONSTANT 5*hd+8 floats, so it must sit well under the device
        // limit regardless of context length (old code scaled with T and
        // capped near 8K).
        let hd = model.cfg.headDim
        step("device.maxThreadgroupMemoryLength="
            + "\(ctx.device.maxThreadgroupMemoryLength)  "
            + "attn tgmem=\((5 * hd + 8) * 4)B (hd=\(hd), constant in T)")
        step("weights in \(ctx.windowMB.count) windows \(ctx.windowMB) MB, "
            + "GPU caps one at "
            + "\(ctx.device.maxBufferLength / 1_048_576) MB")
        step(try checkGemv(model, ctx))
        step(try checkGemm(model, ctx))
        step(try checkDequant(model, ctx))
        step(try checkForward(model))
        step(try checkDecode(model))
        step(checkBookmark(model))
        step(try checkBatchedPrefill(model))
        step(try checkLongContext(model))
        step(try checkMetalPaged(model))
        step(try checkMetalBookmark(model))
        step(try checkCeiling(model))
        return "done"
    }

    // Force the Metal page pool to span many pages cheaply by using a tiny page
    // size (P=4): 17 positions cross 5 pages, exercising the bindless multi-page
    // gather + page-boundary allocation. Must still match the SIMD engine (P=512).
    // Batched prefill vs SIMD token-by-token: prefill 10 ids in chunks of 4
    // (multi-chunk + partial last chunk), then greedily decode 8 -- the whole
    // token path must match the SIMD engine.
    private static func checkBatchedPrefill(_ model: BonsaiModel) throws
        -> String {
        let ids: [Int32] = [9707, 11, 264, 1879, 374, 13, 264, 2100, 1110, 320]
        let n = 8
        let mEng = try MetalEngine(model)
        let sEng = BonsaiEngine(model)
        mEng.reset(); sEng.reset()
        var mCur = mEng.prefillBatch(ids, chunk: 4)
        var sCur = sEng.extend(ids)
        var mSeq: [Int32] = [], sSeq: [Int32] = []
        for _ in 0..<n {
            mSeq.append(mCur); sSeq.append(sCur)
            mCur = mEng.decode(mCur); sCur = sEng.decode(sCur)
        }
        var match = 0
        for i in 0..<n where mSeq[i] == sSeq[i] { match += 1 }
        return "batched prefill(chunk 4) \(match)/\(n) match  "
            + (match == n ? "PASS" : "FAIL") + "\n  metal=\(mSeq)\n  simd =\(sSeq)"
    }

    // Long-context parity: prefill 150 tokens (chunk 48) so the flash attention
    // crosses many TK=32 key tiles, spreads them across all 4 simdgroup stripes
    // WITH wraparound (>128 keys), and hits partial final tiles -- none of which
    // the short (<32-position) checks above exercise. Then greedily decode; the
    // whole path must still match the SIMD engine. Ids are an arbitrary in-vocab
    // pattern (both engines see the same feed, so parity is all that matters).
    private static func checkLongContext(_ model: BonsaiModel) throws -> String {
        let count = 150
        var ids: [Int32] = []
        for i in 0..<count { ids.append(Int32(100 + (i * 37) % 900)) }
        let n = 6
        let mEng = try MetalEngine(model)
        let sEng = BonsaiEngine(model)
        mEng.reset(); sEng.reset()
        var mCur = mEng.prefillBatch(ids, chunk: 48)
        var sCur = sEng.extend(ids)
        var mSeq: [Int32] = [], sSeq: [Int32] = []
        for _ in 0..<n {
            mSeq.append(mCur); sSeq.append(sCur)
            mCur = mEng.decode(mCur); sCur = sEng.decode(sCur)
        }
        var match = 0
        for i in 0..<n where mSeq[i] == sSeq[i] { match += 1 }
        return "long-context (150 prefill, chunk 48) \(match)/\(n) match  "
            + (match == n ? "PASS" : "FAIL")
            + "\n  metal=\(mSeq)\n  simd =\(sSeq)"
    }

    // Prove the ~8K threadgroup-memory ceiling is actually gone: prefill PAST
    // the old cliff (8160 f32 scores) on the Metal engine. Pre-fix, attn_batch
    // requested (basePos+N) floats and violated setThreadgroupMemoryLength here
    // (validation asserts; otherwise a command-buffer fault -> fatalError). The
    // flash path asks for a constant 5*hd+8 floats, so it sails past. No SIMD
    // parity at this length (too slow in pure Swift) -- COMPLETING the run is
    // the check; a regression would abort the process, not print FAIL.
    private static func checkCeiling(_ model: BonsaiModel) throws -> String {
        let count = 9000                       // > the old ~8160 f32-scores cap
        var ids: [Int32] = []
        for i in 0..<count { ids.append(Int32(100 + (i * 37) % 900)) }
        let mEng = try MetalEngine(model)
        mEng.reset()
        let tok = mEng.prefillBatch(ids, chunk: 512)   // chunks cross the cliff
        _ = mEng.decode(tok)                            // decode T=9001 > cliff
        return "ceiling: prefilled \(count) positions + decoded past the old "
            + "~8160 cap  PASS"
    }

    private static func checkMetalPaged(_ model: BonsaiModel) throws -> String {
        let ids: [Int32] = [9707, 11, 264, 1879, 374]
        let n = 12
        let mEng = try MetalEngine(model, pageP: 4)
        let sEng = BonsaiEngine(model)
        mEng.reset(); sEng.reset()
        var mSeq: [Int32] = [], sSeq: [Int32] = []
        var mCur = mEng.extend(ids), sCur = sEng.extend(ids)
        for _ in 0..<n {
            mSeq.append(mCur); sSeq.append(sCur)
            mCur = mEng.decode(mCur); sCur = sEng.decode(sCur)
        }
        var match = 0
        for i in 0..<n where mSeq[i] == sSeq[i] { match += 1 }
        return "metal paged P=4 \(match)/\(n) match  \(match == n ? "PASS" : "FAIL")"
    }

    // Metal paged bookmark/restore roundtrip with a tiny page size, so the mark
    // lands mid-page and the tail-page duplication + multi-page truncate are
    // exercised: mark, decode a path, restore, replay -- must be identical.
    private static func checkMetalBookmark(_ model: BonsaiModel) throws -> String {
        let e = try MetalEngine(model, pageP: 4)
        e.reset()
        _ = e.extend([9707, 11, 264, 1879, 374, 13])   // mark mid-page (len 6)
        let mark = e.bookmark()
        func path() -> [Int32] {
            var out: [Int32] = []
            var cur = e.extend([264, 1879])
            for _ in 0..<5 { out.append(cur); cur = e.decode(cur) }
            return out
        }
        let a = path()
        e.restore(mark)
        let b = path()
        return "metal bookmark replay  \(a == b ? "identical" : "DIVERGED")"
            + "  \(a == b ? "PASS" : "FAIL")"
    }

    // Paged bookmark/restore roundtrip on the SIMD engine: mark mid-page, decode
    // a path, restore to the mark, replay the SAME feed -- the two token paths
    // must be identical. Exercises KVCache page snapshot/restore/truncate + the
    // whole-copy GDN recurrence rollback (the multi-turn think-strip flow).
    private static func checkBookmark(_ model: BonsaiModel) -> String {
        let e = BonsaiEngine(model)
        e.reset()
        _ = e.extend([9707, 11, 264])          // prime (mark lands mid-page)
        let mark = e.bookmark()
        func path() -> [Int32] {
            var out: [Int32] = []
            var cur = e.extend([1879, 374])
            for _ in 0..<5 { out.append(cur); cur = e.decode(cur) }
            return out
        }
        let a = path()
        e.restore(mark)
        let b = path()
        let tag = a == b ? "PASS" : "FAIL"
        return "bookmark restore replay  \(a == b ? "identical" : "DIVERGED")"
            + "  \(tag)\n  a=\(a)\n  b=\(b)"
    }

    // One token from a reset state: Metal forward hidden + its logits argmax vs
    // SIMD (both from the SAME single forward -- no re-forward, which would
    // pollute the GDN recurrent state). Isolates any per-op divergence on the
    // first token (GDN sees a zero state, attention a length-1 KV).
    private static func checkForward(_ model: BonsaiModel) throws -> String {
        let mEng = try MetalEngine(model)
        let sEng = BonsaiEngine(model)
        let tok = 9707
        mEng.reset()
        let mhBuf = mEng.forward(token: tok, pos: 0)
        let mh = Array(mhBuf.f32(model.cfg.nEmbd))
        let ml = mEng.argmax(mEng.logits(mhBuf))
        sEng.reset()
        let sh = sEng.forward(token: tok, pos: 0)
        let sl = sEng.argmax(sEng.logits(sh))
        let dh = maxAbsDiff(mh, sh)
        let tag = dh < 5e-2 && ml == sl ? "PASS" : "FAIL"
        return "forward tok=\(tok)  hiddenMaxDiff=\(dh)  "
            + "argmax metal=\(ml) simd=\(sl)  \(tag)"
    }

    // End-to-end greedy parity: prefill a seed, then greedily decode 12 tokens
    // on each engine and compare the full sequences. Exercises the GDN
    // recurrence + growing-KV attention over many steps -- the real parity test.
    private static func checkDecode(_ model: BonsaiModel) throws -> String {
        let ids: [Int32] = [9707, 11, 264, 1879, 374]
        let n = 12
        let mEng = try MetalEngine(model)
        let sEng = BonsaiEngine(model)
        mEng.reset(); sEng.reset()
        var mSeq: [Int32] = []
        var sSeq: [Int32] = []
        var mCur = mEng.extend(ids)
        var sCur = sEng.extend(ids)
        for _ in 0..<n {
            mSeq.append(mCur); sSeq.append(sCur)
            mCur = mEng.decode(mCur)
            sCur = sEng.decode(sCur)
        }
        var match = 0
        for i in 0..<n where mSeq[i] == sSeq[i] { match += 1 }
        let tag = match == n ? "PASS" : "FAIL"
        return "decode \(match)/\(n) greedy tokens match  \(tag)\n"
            + "  metal=\(mSeq)\n  simd =\(sSeq)"
    }

    // Q2_0 batched GEMM vs per-column Q2_0.matvec on layer-0 ffn_gate (a Q2_0
    // matrix both lineages carry; the hybrid's wqkv is absent on dense qwen3).
    private static func checkGemm(_ model: BonsaiModel,
                                  _ ctx: MetalContext) throws -> String {
        let w = model.layers[0].ffnGate
        let k = w.dims[0], m = w.dims[1]
        let n = 5
        var xflat: [Float] = []
        for c in 0..<n { xflat += fill(k, seed: UInt64(c + 10)) }
        let xb = ctx.makeF32(xflat)
        let outb = ctx.makeF32(n * m)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        MetalEnc(ctx: ctx, e: e).gemm(
            w, X: xb, out: outb,
            off: ctx.window(UInt64(w.base - model.gguf.map)), N: n)
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let gpu = Array(outb.f32(n * m))
        var d: Float = 0
        var mag: Float = 0
        for c in 0..<n {
            let col = Array(xflat[c * k ..< (c + 1) * k])
            var ref = [Float](repeating: 0, count: m)
            Q2_0.matvec(w, x: col, out: &ref)
            for j in 0..<m {
                d = max(d, abs(gpu[c * m + j] - ref[j]))
                mag = max(mag, abs(ref[j]))
            }
        }
        // RELATIVE, because an absolute bound on a K-deep dot product says
        // nothing: the output grows with K and the activation scale, so the
        // same tolerance is loose on one tensor and impossible on another.
        // fp16 carries ~4.9e-4 of relative precision and the half-tile GEMM
        // stages its operands there by design, so anything under 2e-3 is the
        // format working; a real defect lands at O(1), not near epsilon.
        let rel = mag > 0 ? d / mag : d
        return "gemm [\(n),\(k)]@[\(k),\(m)]  maxAbsDiff=\(d)  rel=\(rel)  "
            + (rel < 2e-3 ? "PASS" : "FAIL")
    }

    // Q2_0 GEMV vs Q2_0.matvec on the layer-0 ffn_gate (present in both
    // lineages; dense qwen3 has no GDN in-projection).
    private static func checkGemv(_ model: BonsaiModel,
                                  _ ctx: MetalContext) throws -> String {
        let w = model.layers[0].ffnGate
        let k = w.dims[0]
        let m = w.dims[1]
        let x = fill(k, seed: 1)

        var ref = [Float](repeating: 0, count: m)
        Q2_0.matvec(w, x: x, out: &ref)

        let gpu = try gemv(ctx, weight: w, mapBase: model.gguf.map, x: x)
        let d = maxAbsDiff(ref, gpu)
        let tag = d < 1e-3 ? "PASS" : "FAIL"
        return "gemv ffn_gate[\(k),\(m)]  maxAbsDiff=\(d)  \(tag)"
    }

    // Q2_0 row dequant vs Q2_0.dequant on the token-embedding row for id 100.
    private static func checkDequant(_ model: BonsaiModel,
                                     _ ctx: MetalContext) throws -> String {
        let t = model.tokEmbd
        let k = t.dims[0]
        let id = 100
        let rowBytes = k / 128 * 34
        let row = ctx.window(
            UInt64((t.base - model.gguf.map) + id * rowBytes))

        var ref = [Float](repeating: 0, count: k)
        Q2_0.dequant(t.base + id * rowBytes, count: k, into: &ref)

        let out = ctx.makeF32(k)
        let enc = try dispatch(ctx, "q2_0_dequant_row", threads: k) { e in
            e.setBuffer(row.buf, offset: 0, index: 0)
            e.setBuffer(out, offset: 0, index: 1)
            var a = GemvArgs(woff: row.local, K: UInt32(k), M: UInt32(k))
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 2)
        }
        enc.waitUntilCompleted()
        let gpu = Array(out.f32(k))
        let d = maxAbsDiff(ref, gpu)
        let tag = d < 1e-4 ? "PASS" : "FAIL"
        return "dequant tok_embd row[\(k)]  maxAbsDiff=\(d)  \(tag)"
    }

    // One GEMV dispatch (one simdgroup per output row), returning the result.
    static func gemv(_ ctx: MetalContext, weight w: GGUFTensor,
                     mapBase: UnsafeRawPointer, x: [Float]) throws -> [Float] {
        let k = w.dims[0]
        let m = w.dims[1]
        let ref = ctx.window(UInt64(w.base - mapBase))
        let xb = ctx.makeF32(x)
        let out = ctx.makeF32(m)
        let cb = try dispatch(ctx, "q2_0_gemv", groups: m, threadsPerGroup: 32) {
            e in
            e.setBuffer(ref.buf, offset: 0, index: 0)
            e.setBuffer(xb, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            var a = GemvArgs(woff: ref.local, K: UInt32(k), M: UInt32(m))
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 3)
        }
        cb.waitUntilCompleted()
        return Array(out.f32(m))
    }

    // Encode + commit a 1-D grid of `threads` (one thread per element).
    static func dispatch(_ ctx: MetalContext, _ name: String, threads: Int,
                         _ bind: (MTLComputeCommandEncoder) -> Void) throws
        -> MTLCommandBuffer {
        let pipe = try ctx.pipeline(name)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pipe)
        bind(e)
        let w = pipe.threadExecutionWidth
        e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                          threadsPerThreadgroup:
                            MTLSize(width: w, height: 1, depth: 1))
        e.endEncoding()
        cb.commit()
        return cb
    }

    // Encode + commit `groups` threadgroups of `threadsPerGroup` threads each
    // (the GEMV wants exactly one simdgroup per output row).
    static func dispatch(_ ctx: MetalContext, _ name: String, groups: Int,
                         threadsPerGroup: Int,
                         _ bind: (MTLComputeCommandEncoder) -> Void) throws
        -> MTLCommandBuffer {
        let pipe = try ctx.pipeline(name)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pipe)
        bind(e)
        e.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup:
                MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        e.endEncoding()
        cb.commit()
        return cb
    }
}

// Matches the MSL `GemvArgs { ulong woff; uint K; uint M; }`.
struct GemvArgs { var woff: UInt64; var K: UInt32; var M: UInt32 }

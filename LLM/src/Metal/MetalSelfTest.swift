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
        step(try checkNarrow(model, ctx))
        step(try checkVisionBlocks(ctx))
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

    // ---- the vision-block mask ------------------------------------------
    // The batched attention's block mask, three ways: the scalar kernel, the
    // matrix kernel, and a plain Swift oracle -- on a chunk carrying TWO
    // blocks and text.
    //
    // WHY IT EXISTS, and this is the durable part. Every vision fixture this
    // repo owns is a SINGLE block, and a chunk used to carry at most one, so
    // both bugs the block mask has shipped -- a chunk that silently dropped
    // the mask, and a matrix kernel that read the wrong rows' bounds -- sailed
    // through every gate while the model answered nonsense about a video. A
    // gate that cannot see two blocks cannot see that class of bug at all.
    //
    // The last assertion is the one that keeps it honest: run the SAME inputs
    // with an empty block table and the answer must MOVE. Without it a kernel
    // that ignored the table entirely would score a clean pass against an
    // oracle that agreed with it.
    private static func checkVisionBlocks(_ ctx: MetalContext) throws
        -> String {
        var lines: [String] = []
        var failed = 0
        for basePos in [0, 8] {
            for window in [0, 12] {
                let r = try visionCase(ctx, basePos: basePos, window: window)
                lines.append("  " + r.0)
                if !r.1 { failed += 1 }
            }
        }
        return "vision blocks (2 blocks + text in one chunk)  "
            + (failed == 0 ? "PASS" : "FAIL (\(failed))") + "\n"
            + lines.joined(separator: "\n")
    }

    // One geometry. N=37 ends on a PARTIAL 8-row tile, so the matrix kernel's
    // padding rows past N are exercised; the two runs straddle its tile
    // boundaries at rows 8, 16 and 32.
    private static func visionCase(_ ctx: MetalContext, basePos: Int,
                                   window: Int) throws -> (String, Bool) {
        let hd = 256, nH = 2, nKV = 1, P = 8, N = 37
        let scale: Float = 0.125
        let T = basePos + N
        let fix = kvFixture(ctx, hd: hd, nH: nH, nKV: nKV, P: P, T: T, N: N)
        let rows = blockRows(N, basePos, [(2, 14), (15, 31)])
        let table = ctx.makeU32(2 * (N + 8))
        let cells = table.u32(2 * (N + 8))
        for j in 0..<N {
            cells[2 * j] = UInt32(rows[j].0)
            cells[2 * j + 1] = UInt32(rows[j].1)
        }
        let want = attendRef(fix.q, fix.k, fix.v, rows, basePos: basePos,
                             N: N, nH: nH, hd: hd, window: window,
                             scale: scale)
        func run(_ blocks: MTLBuffer?, _ matrix: Bool) -> [Float] {
            attn(ctx, fix, hd: hd, nH: nH, nKV: nKV, P: P, N: N,
                 basePos: basePos, window: window, scale: scale,
                 blocks: blocks, matrix: matrix)
        }
        let scalar = relDiff(run(table, false), want)
        // Below Apple7 there are no matrix units and no attn_batch_mm to
        // build, so the scalar arm is the whole kernel there.
        let mm = ctx.matrixUnits ? relDiff(run(table, true), want) : 0
        let blind = ctx.matrixUnits ? relDiff(run(nil, true), want)
                                    : relDiff(run(nil, false), want)
        let ok = scalar < 1e-4 && mm < 1e-2 && blind > 0.1
        return (String(format:
            "p%-2d w%-2d  scalar %.2e  matrix %.2e  no-table %.2e  %@",
            basePos, window, scalar, mm, blind, ok ? "PASS" : "FAIL"), ok)
    }

    // The vision block each chunk row belongs to, as the absolute [lo, hi)
    // every position inside it attends across; (0, 0) for text.
    private static func blockRows(_ N: Int, _ basePos: Int,
                                  _ runs: [(Int, Int)]) -> [(Int, Int)] {
        var out = [(Int, Int)](repeating: (0, 0), count: N)
        for r in runs {
            for j in r.0..<r.1 { out[j] = (basePos + r.0, basePos + r.1) }
        }
        return out
    }

    // A filled KV pool plus the f32 values it actually holds: the pages are
    // half, so the oracle has to score against the ROUNDED values or it
    // measures the storage format instead of the mask.
    private struct KVFixture {
        let pool: MetalKVPool
        let qBuf: MTLBuffer
        let gate: MTLBuffer
        let out: MTLBuffer
        let q: [Float]
        let k: [Float]
        let v: [Float]
    }

    private static func kvFixture(_ ctx: MetalContext, hd: Int, nH: Int,
                                  nKV: Int, P: Int, T: Int,
                                  N: Int) -> KVFixture {
        precondition(nKV == 1, "the k/v below are indexed as one kv head")
        let pool = MetalKVPool(device: ctx.device, P: P, kvDim: hd * nKV)
        pool.appendBatch(T)
        let src = fill(2 * T * hd, seed: 60)
        var k = [Float](repeating: 0, count: T * hd)
        var v = [Float](repeating: 0, count: T * hd)
        for t in 0..<T {
            let kp = pool.kPages[t / P].contents()
                .assumingMemoryBound(to: Float16.self) + (t % P) * hd
            let vp = pool.vPages[t / P].contents()
                .assumingMemoryBound(to: Float16.self) + (t % P) * hd
            for i in 0..<hd {
                kp[i] = Float16(src[t * hd + i])
                vp[i] = Float16(src[(T + t) * hd + i])
                k[t * hd + i] = Float(kp[i])
                v[t * hd + i] = Float(vp[i])
            }
        }
        pool.refreshTable()
        let q = fill(N * nH * hd, seed: 61)
        return KVFixture(pool: pool, qBuf: ctx.makeF32(q),
                         gate: ctx.makeF32(N * nH * hd),
                         out: ctx.makeF32(N * nH * hd), q: q, k: k, v: v)
    }

    // One batched-attention dispatch, on whichever arm `matrix` selects.
    private static func attn(_ ctx: MetalContext, _ fix: KVFixture, hd: Int,
                             nH: Int, nKV: Int, P: Int, N: Int, basePos: Int,
                             window: Int, scale: Float, blocks: MTLBuffer?,
                             matrix: Bool) -> [Float] {
        let pool = fix.pool
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        MetalEnc(ctx: ctx, e: e).attnBatch(
            qN: fix.qBuf, kAddr: pool.kAddr, vAddr: pool.vAddr,
            pages: pool.residentPages, gateN: fix.gate, outN: fix.out,
            hd: hd, nH: nH, nKV: nKV, kvDim: hd * nKV, P: P, scale: scale,
            basePos: basePos, N: N, gated: 0, window: window,
            blocks: blocks, matrix: matrix)
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        precondition(cb.error == nil, "attn_batch: \(cb.error!)")
        return Array(fix.out.f32(N * nH * hd))
    }

    // The oracle. Each row's range is [lo, hi): the window bounds it from
    // below, and a row inside a vision block reads to that block's end --
    // which stays CONTIGUOUS because the block contains the row.
    private static func attendRef(_ q: [Float], _ k: [Float], _ v: [Float],
                                  _ blocks: [(Int, Int)], basePos: Int,
                                  N: Int, nH: Int, hd: Int, window: Int,
                                  scale: Float) -> [Float] {
        var out = [Float](repeating: 0, count: N * nH * hd)
        for n in 0..<N {
            let p = basePos + n
            let t = p + 1
            let lo = window == 0 || t <= window ? 0 : t - window
            let b = blocks[n]
            let hi = p >= b.0 && p < b.1 ? max(t, b.1) : t
            for h in 0..<nH {
                let row = attendOne(q, (n * nH + h) * hd, k, v, hd, lo, hi,
                                    scale)
                for i in 0..<hd { out[(n * nH + h) * hd + i] = row[i] }
            }
        }
        return out
    }

    // One query row over [lo, hi): scores, softmax, weighted V -- the
    // arithmetic attn_flash performs, in the same order.
    private static func attendOne(_ q: [Float], _ qo: Int, _ k: [Float],
                                  _ v: [Float], _ hd: Int, _ lo: Int,
                                  _ hi: Int, _ scale: Float) -> [Float] {
        var s = [Float](repeating: 0, count: hi - lo)
        var top = -Float.greatestFiniteMagnitude
        for t in lo..<hi {
            s[t - lo] = dot(q, qo, k, t * hd, hd) * scale
            top = max(top, s[t - lo])
        }
        var sum: Float = 0
        for i in 0..<s.count {
            s[i] = expf(s[i] - top)
            sum += s[i]
        }
        var out = [Float](repeating: 0, count: hd)
        for t in lo..<hi { axpy(s[t - lo] / sum, v, t * hd, &out, hd) }
        return out
    }

    private static func dot(_ a: [Float], _ ao: Int, _ b: [Float], _ bo: Int,
                            _ n: Int) -> Float {
        var s: Float = 0
        for i in 0..<n { s += a[ao + i] * b[bo + i] }
        return s
    }

    private static func axpy(_ w: Float, _ src: [Float], _ so: Int,
                             _ dst: inout [Float], _ n: Int) {
        for i in 0..<n { dst[i] += w * src[so + i] }
    }

    // Worst absolute difference against the reference's own scale. An
    // absolute bound says nothing here: the output is an average of V rows,
    // so its magnitude follows the fixture rather than the kernel.
    private static func relDiff(_ a: [Float], _ b: [Float]) -> Float {
        var d: Float = 0
        var mag: Float = 0
        for i in 0..<min(a.count, b.count) {
            d = max(d, abs(a[i] - b[i]))
            mag = max(mag, abs(b[i]))
        }
        return mag > 0 ? d / mag : d
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
            QB.matvec(w, x: col, out: &ref)
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

    // The narrow GEMMs against N gemv calls of the same weight. The two
    // differ only in the ORDER of one reduction, so anything past fp32
    // rounding is a layout bug rather than a tolerance to widen.

    private static func checkNarrow(_ model: BonsaiModel,
                                    _ ctx: MetalContext) throws -> String {
        var lines: [String] = []
        var failed = 0
        var ran = 0
        for ty in [GGUFType.q2_e8, .q4_0] {
            if let w = widest(model, ty) {
                let k = w.dims[0], m = w.dims[1]
                let off = ctx.window(UInt64(w.base - model.gguf.map))
                for n in 2...MetalEnc.narrowMax {
                    var xflat: [Float] = []
                    for c in 0..<n { xflat += fill(k, seed: UInt64(c + 70)) }
                    let xb = ctx.makeF32(xflat)
                    let narrow = ctx.makeF32(2 * n * m)
                    let cb = ctx.queue.makeCommandBuffer()!
                    let e = cb.makeComputeCommandEncoder()!
                    let f = MetalEnc(ctx: ctx, e: e)
                    f.gemmNarrow(w, X: xb, out: narrow, off: off, N: n)
                    for c in 0..<n {
                        f.gemv(w, x: xb, out: narrow, off: off,
                               xOff: c * k * 4,
                               outOff: (n + c) * m * 4)
                    }
                    e.endEncoding()
                    cb.commit()
                    cb.waitUntilCompleted()
                    precondition(cb.error == nil, "narrow: \(cb.error!)")
                    let got = Array(narrow.f32(2 * n * m))
                    var d: Float = 0
                    var mag: Float = 0
                    for i in 0..<(n * m) {
                        d = max(d, abs(got[i] - got[n * m + i]))
                        mag = max(mag, abs(got[n * m + i]))
                    }
                    let rel = mag > 0 ? d / mag : d
                    let ok = rel < 1e-5
                    if !ok { failed += 1 }
                    ran += 1
                    lines.append(String(format:
                        "  %@ N=%d [%d,%d]  rel %.2e  %@", "\(ty)", n, k, m,
                        rel, ok ? "PASS" : "FAIL"))
                }
            }
        }
        return "narrow gemm vs gemv \(ran - failed)/\(ran)  "
            + (failed == 0 ? "PASS" : "FAIL") + "\n"
            + lines.joined(separator: "\n")
    }

    private static func widest(_ model: BonsaiModel,
                               _ ty: GGUFType) -> GGUFTensor? {
        var out: GGUFTensor? = nil
        for name in model.gguf.tensors.keys.sorted() {
            let t = model.gguf.tensors[name]!
            if t.type == ty && t.dims.count == 2 && t.dims[0] % 128 == 0
                && (out == nil || t.dims[1] > out!.dims[1]) {
                out = t
            }
        }
        return out
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
        QB.matvec(w, x: x, out: &ref)

        let gpu = try gemv(ctx, weight: w, mapBase: model.gguf.map, x: x)
        let d = maxAbsDiff(ref, gpu)
        let mag = ref.reduce(Float(0)) { peak, v in max(peak, abs(v)) }
        let rel = mag > 0 ? d / mag : d
        let tag = rel < 1e-4 ? "PASS" : "FAIL"
        return "gemv ffn_gate[\(k),\(m)]  maxAbsDiff=\(d)  rel=\(rel)  \(tag)"
    }

    // Q2_0 row dequant vs Q2_0.dequant on the token-embedding row for id 100.
    private static func checkDequant(_ model: BonsaiModel,
                                     _ ctx: MetalContext) throws -> String {
        let t = model.tokEmbd
        let k = t.dims[0]
        let id = 100
        let rowBytes = GGUF.rowByteCount(t.type, k)
        let row = ctx.window(
            UInt64((t.base - model.gguf.map) + id * rowBytes))

        var ref = [Float](repeating: 0, count: k)
        ref.withUnsafeMutableBufferPointer { rb in
            QB.dequant(t, row: id, count: k, into: rb.baseAddress!)
        }

        let out = ctx.makeF32(k)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        MetalEnc(ctx: ctx, e: e).dequantRow(weightOff: row, out: out, n: k,
                                            type: t.type)
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let gpu = Array(out.f32(k))
        let d = maxAbsDiff(ref, gpu)
        let tag = d < 1e-4 ? "PASS" : "FAIL"
        return "dequant tok_embd row[\(k)]  maxAbsDiff=\(d)  \(tag)"
    }

    static func gemv(_ ctx: MetalContext, weight w: GGUFTensor,
                     mapBase: UnsafeRawPointer, x: [Float]) throws -> [Float] {
        let m = w.dims[1]
        let xb = ctx.makeF32(x)
        let out = ctx.makeF32(m)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        MetalEnc(ctx: ctx, e: e).gemv(
            w, x: xb, out: out,
            off: ctx.window(UInt64(w.base - mapBase)))
        e.endEncoding()
        cb.commit()
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

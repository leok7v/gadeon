import Accelerate
import CoreML
import Foundation

// Paged-KV attention: a lazy pos-major page pool plus flash-style tiles whose
// un-normalized partials merge on the host with an fp32 online softmax. The
// cache is fixed-size pages of P positions, each allocated on first write.
// Port of paged_attention.py. Tiles emit o dh-major ([KVH,dh,G]) because the
// [KVH,G,dh] form lowers to CPU, so the host transposes before merging.

// kv-head count is per-model (2 for 0.8B/2B, 4 for 4B/9B); the pool carries it
// so its byte layout matches. dh (head dim) is constant across the qwen3_5
// family (256), so it stays a module constant. G (query heads per kv head)
// is per-model -- 4 for 0.8B-9B, 6 for the 27B (24 attn / 4 kv) -- so it is
// passed in from the engine's I/O-derived gqaGroup, not hardcoded here.
// SCALE (dh**-0.5) is baked into the tile mlprogram; the host merge re-applies
// it only in the seq=1 CPU decode, where it is derived from the pool's dh.

// Lazy pos-major KV page pool for one attention layer. Pages are created on
// first write; no preallocation.
final class PagePool {
    let P: Int
    let kvHeads: Int                      // kv heads (2 or 4), for the byte layout
    let dh: Int                           // head dim, from the model (I/O-derived)
    private var kPages: [MLMultiArray] = []
    private var vPages: [MLMultiArray] = []
    private(set) var length = 0
    // fp32 mirror of the written K/V, one contiguous [n,dh] per kv head,
    // grown to a WATERMARK (f32Len): rows convert from the fp16 pages once,
    // when first read, so seq=1 decode pays one row's conversion per token
    // instead of re-converting the whole history every token every layer.
    // The mirror costs 2x the fp16 pages in memory -- the space bought for
    // the O(1) per-token time. truncate() only lowers the watermark (stale
    // rows re-convert in place on the next grow), so spec-decode's
    // truncate/append cycles never churn allocations.
    private var kF32: [[Float]] = []
    private var vF32: [[Float]] = []
    private var f32Len = 0

    init(P: Int, kvHeads: Int, dh: Int) {
        self.P = P; self.kvHeads = kvHeads; self.dh = dh
    }

    // Per-head contiguous fp32 K/V covering positions [0, length); entries
    // past length*dh are stale and must not be read.
    func floats32() -> (k: [[Float]], v: [[Float]]) {
        if kF32.count != kvHeads {
            kF32 = Array(repeating: [], count: kvHeads)
            vF32 = Array(repeating: [], count: kvHeads)
            f32Len = 0
        }
        if f32Len < length {
            for h in 0 ..< kvHeads {
                growF32(h, kPages, &kF32[h])
                growF32(h, vPages, &vF32[h])
            }
            f32Len = length
        }
        return (kF32, vF32)
    }

    // Convert head h's rows [f32Len, length) from the fp16 pages into `dst`
    // at their row offsets, page span by page span.
    private func growF32(_ h: Int, _ pages: [MLMultiArray],
                         _ dst: inout [Float]) {
        let need = length * dh
        if dst.count < need {
            dst.append(contentsOf: repeatElement(0, count: need - dst.count))
        }
        var r = f32Len
        while r < length {
            let span = min(P - r % P, length - r)
            convertHalfBlock(pages[r / P], (h * P + r % P) * dh, &dst,
                             r * dh, span * dh)
            r += span
        }
    }

    var pageCount: Int { kPages.count }
    func kPage(_ i: Int) -> MLMultiArray { kPages[i] }
    func vPage(_ i: Int) -> MLMultiArray { vPages[i] }

    // Write one position's K/V ([KVH,dh] each, fp16) at the current tail,
    // starting a fresh zeroed page when the tail lands on a page boundary.
    func append(k: MLMultiArray, v: MLMultiArray) throws {
        if length % P == 0 {
            kPages.append(try zeroPage(P))
            vPages.append(try zeroPage(P))
        }
        let slot = length % P
        writePosition(kPages[kPages.count - 1], k, slot)
        writePosition(vPages[vPages.count - 1], v, slot)
        length += 1
    }

    // Bulk-load `count` positions from contiguous prefill K/V ([KVH,count,dh]
    // fp16 each): one page per P-position span, the tail page left
    // zero-padded.
    func seed(K: MLMultiArray, V: MLMultiArray, count: Int) throws {
        var i = 0
        while i < count {
            let n = min(P, count - i)
            let kp = try zeroPage(P)
            let vp = try zeroPage(P)
            copyPrefix(kp, K, count, i, n)
            copyPrefix(vp, V, count, i, n)
            kPages.append(kp)
            vPages.append(vp)
            i += P
        }
        length = count
    }


    // Scatter [KVH,dh] into page slot `slot`, head-strided by the page's P.
    private func writePosition(_ page: MLMultiArray,
                               _ src: MLMultiArray, _ slot: Int) {
        let KVH = kvHeads
        page.withUnsafeMutableBytes { dst, _ in
            src.withUnsafeBytes { s in
                for h in 0 ..< KVH {
                    memcpy(dst.baseAddress! + (h * P + slot) * dh * 2,
                           s.baseAddress! + h * dh * 2, dh * 2)
                }
            }
        }
    }

    // Copy `n` positions from src[:, start:start+n, :] ([KVH,total,dh]) into
    // the head of `page`, translating the source's `total` head stride to P.
    private func copyPrefix(_ page: MLMultiArray, _ src: MLMultiArray,
                            _ total: Int, _ start: Int, _ n: Int) {
        let KVH = kvHeads
        page.withUnsafeMutableBytes { dst, _ in
            src.withUnsafeBytes { s in
                for h in 0 ..< KVH {
                    memcpy(dst.baseAddress! + h * P * dh * 2,
                           s.baseAddress! + (h * total + start) * dh * 2,
                           n * dh * 2)
                }
            }
        }
    }

    // Copy `count` positions starting at global position `start` into fresh
    // [KVH,tileP,dh] K/V buffers -- a carry-tile key window. Because P % tileP
    // == 0 (16384 % 512), a tileP window never crosses a pool-page boundary,
    // so it reads exactly one page. Positions past `count` stay zero (the band
    // masks them). One small memcpy per head; decode's page size P is
    // unchanged.
    func tileAt(_ start: Int, _ count: Int, _ tileP: Int) throws
        -> (MLMultiArray, MLMultiArray) {
        let pi = start / P
        let off = start % P
        let k = try zeroPage(tileP)
        let v = try zeroPage(tileP)
        copyWindow(k, kPages[pi], off, count, tileP)
        copyWindow(v, vPages[pi], off, count, tileP)
        return (k, v)
    }

    // Copy `n` positions of `page` starting at slot `off` (head stride P) into
    // the head of `dst` (head stride tileP), per head.
    private func copyWindow(_ dst: MLMultiArray, _ page: MLMultiArray,
                            _ off: Int, _ n: Int, _ tileP: Int) {
        let KVH = kvHeads
        dst.withUnsafeMutableBytes { d, _ in
            page.withUnsafeBytes { s in
                for h in 0 ..< KVH {
                    memcpy(d.baseAddress! + h * tileP * dh * 2,
                           s.baseAddress! + (h * P + off) * dh * 2, n * dh * 2)
                }
            }
        }
    }

    // Roll the tail back to `newLength`, dropping any now-empty pages. Rows at
    // or past newLength keep stale bytes, but the flash band mask ignores them
    // and the next append overwrites them, so only length + page count reset.
    // Used by the engine to undo rejected speculative-decode steps.
    func truncate(to newLength: Int) {
        length = newLength
        f32Len = min(f32Len, newLength)
        let need = (newLength + P - 1) / P
        while kPages.count > need {
            kPages.removeLast()
            vPages.removeLast()
        }
    }

    // A batched verify appends its chunk's K/V to a SNAPSHOT of the committed
    // pool without disturbing it. Completed pages are append-only and
    // read-only, so the clone shares them; only the partially-filled tail page
    // is copied, since the next append writes into it. When the pool ends on a
    // page boundary the next append starts a fresh page, so no copy is needed.
    // The fp32 mirror is deliberately NOT carried over: a restored bookmark
    // pays one full re-convert on its first decode instead of every bookmark
    // holding a second fp32 copy of the whole KV.
    func clone() throws -> PagePool {
        let c = PagePool(P: P, kvHeads: kvHeads, dh: dh)
        c.kPages = kPages
        c.vPages = vPages
        c.length = length
        if length % P != 0 {
            let tail = kPages.count - 1
            c.kPages[tail] = try Self.duplicate(kPages[tail])
            c.vPages[tail] = try Self.duplicate(vPages[tail])
        }
        return c
    }

    private static func duplicate(_ a: MLMultiArray) throws -> MLMultiArray {
        let d = try MLMultiArray(shape: a.shape, dataType: .float16)
        a.withUnsafeBytes { s in
            d.withUnsafeMutableBytes { dst, _ in
                _ = memcpy(dst.baseAddress!, s.baseAddress!, s.count)
            }
        }
        return d
    }

    // Compact fp16 [KVH, length, dh] K and V for the written positions
    // (page-strided -> logical contiguous), for serializing a context to
    // storage. Empty when length == 0.
    func compact() -> (k: [UInt16], v: [UInt16]) {
        let KVH = kvHeads
        var kOut = [UInt16](repeating: 0, count: KVH * length * dh)
        var vOut = [UInt16](repeating: 0, count: KVH * length * dh)
        gatherHeads(kPages, length, &kOut)
        gatherHeads(vPages, length, &vOut)
        return (kOut, vOut)
    }

    private func gatherHeads(_ pages: [MLMultiArray], _ n: Int,
                             _ out: inout [UInt16]) {
        let KVH = kvHeads
        for h in 0 ..< KVH {
            var done = 0
            for pi in 0 ..< pages.count where done < n {
                let inPage = min(P, n - done)
                pages[pi].withUnsafeBytes { s in
                    let sp = s.bindMemory(to: UInt16.self)
                    for slot in 0 ..< inPage {
                        let src = (h * P + slot) * dh
                        let dst = (h * n + done + slot) * dh
                        for d in 0 ..< dh { out[dst + d] = sp[src + d] }
                    }
                }
                done += inPage
            }
        }
    }

    // Rebuild a pool from compact fp16 [KVH, length, dh] K and V (the inverse
    // of compact), via the bulk seed path. kvHeads pins the per-model layout.
    static func fromCompact(k: [UInt16], v: [UInt16], length: Int,
                            P: Int, kvHeads: Int, dh: Int) throws -> PagePool {
        let pool = PagePool(P: P, kvHeads: kvHeads, dh: dh)
        if length > 0 {
            try pool.seed(K: dense(k, length, kvHeads, dh),
                          V: dense(v, length, kvHeads, dh), count: length)
        }
        return pool
    }

    private static func dense(_ src: [UInt16], _ n: Int,
                              _ kvHeads: Int, _ dh: Int) throws -> MLMultiArray {
        let a = try MLMultiArray(
            shape: [NSNumber(value: kvHeads), NSNumber(value: n),
                    NSNumber(value: dh)], dataType: .float16)
        a.withUnsafeMutableBytes { d, _ in
            let dp = d.bindMemory(to: UInt16.self)
            for i in 0 ..< src.count { dp[i] = src[i] }
        }
        return a
    }

    private func zeroPage(_ P: Int) throws -> MLMultiArray {
        let a = try MLMultiArray(
            shape: [NSNumber(value: kvHeads), NSNumber(value: P),
                    NSNumber(value: dh)], dataType: .float16)
        a.withUnsafeMutableBytes { buf, _ in
            _ = buf.initializeMemory(as: UInt16.self, repeating: 0)
        }
        return a
    }
}

// vImage-convert `count` contiguous fp16 halves at `srcOffsetHalves` in `page`
// into `dst` at `dstOffset` (fp32). One vectorized half->single pass.
private func convertHalfBlock(_ page: MLMultiArray, _ srcOffsetHalves: Int,
                              _ dst: inout [Float], _ dstOffset: Int,
                              _ count: Int) {
    page.withUnsafeBytes { raw in
        let src = raw.bindMemory(to: UInt16.self)
        dst.withUnsafeMutableBufferPointer { db in
            var sBuf = vImage_Buffer(
                data: UnsafeMutableRawPointer(
                    mutating: src.baseAddress! + srcOffsetHalves),
                height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
            var dBuf = vImage_Buffer(data: db.baseAddress! + dstOffset, height: 1,
                                     width: vImagePixelCount(count),
                                     rowBytes: count * 4)
            _ = vImageConvert_Planar16FtoPlanarF(&sBuf, &dBuf, 0)
        }
    }
}

// Host (Accelerate) attention for ONE seq=1 decode query over the pool: the
// normalized [KVH,G,dh] fp32 result (SCALE applied). At seq=1 each per-page
// attention is a tiny GEMV, so the ANE tile's per-dispatch cost dwarfs the
// arithmetic; CPU fp32 is faster and at least as accurate. No causal mask --
// the newest query attends to all `length` KV positions. The GEMVs run over
// the pool's incrementally-converted fp32 mirror, so the per-token cost is
// one row's fp16->fp32 conversion plus the O(ctx) math, not an O(ctx)
// re-conversion of the whole history.
func hostFlashDecode(q: MLMultiArray, pool: PagePool, G: Int) -> [Float] {
    let n = pool.length
    let KVH = pool.kvHeads
    let dh = pool.dh
    let scale = Float(pow(Double(dh), -0.5))          // DH**-0.5, model-derived
    let HEADS = KVH * G
    var result = [Float](repeating: 0, count: HEADS * dh)
    if n == 0 { return result }
    let qf = toFloats(q)                              // [KVH,G,dh] fp32
    let (kAll, vAll) = pool.floats32()
    var scores = [Float](repeating: 0, count: n)
    for kvh in 0 ..< KVH {
        let kf = kAll[kvh]
        let vf = vAll[kvh]
        for g in 0 ..< G {
            let h = kvh * G + g
            // scores[n] = K[n,dh] . q_h[dh]  (vDSP_mmul: [n,dh]x[dh,1])
            qf.withUnsafeBufferPointer { qp in
                kf.withUnsafeBufferPointer { kp in
                    scores.withUnsafeMutableBufferPointer { sp in
                        vDSP_mmul(kp.baseAddress!, 1, qp.baseAddress! + h * dh, 1,
                                  sp.baseAddress!, 1, vDSP_Length(n), 1,
                                  vDSP_Length(dh))
                    }
                }
            }
            var mx = -Float.infinity
            for j in 0 ..< n {
                let s = scores[j] * scale; scores[j] = s; if s > mx { mx = s }
            }
            var sum: Float = 0
            for j in 0 ..< n {
                let e = Foundation.exp(scores[j] - mx); scores[j] = e; sum += e
            }
            // result_h[dh] = scores[n] . V[n,dh]  (vDSP_mmul: [1,n]x[n,dh])
            scores.withUnsafeBufferPointer { sp in
                vf.withUnsafeBufferPointer { vp in
                    result.withUnsafeMutableBufferPointer { rp in
                        vDSP_mmul(sp.baseAddress!, 1, vp.baseAddress!, 1,
                                  rp.baseAddress! + h * dh, 1, 1, vDSP_Length(dh),
                                  vDSP_Length(n))
                    }
                }
            }
            let inv = 1 / sum
            for d in 0 ..< dh { result[h * dh + d] *= inv }
        }
    }
    return result
}

// Batched-query paged flash for S query rows q ([KVH,S,G,dh] fp16) at global
// offset s0GlobalPos: the normalized [KVH,S,G,dh] result. One tile call per
// page with a per-(row, key) causal/pad band, then an fp32 online-softmax
// merge carrying the S axis. History pages come out fully valid (band
// all-zero); the tail page gets the causal band (row s sees keys with global
// pos <= s0+s). LEGACY fallback only (a set without tile_carry): the band
// mask is [1,1,P,S*G] fp16 -- ~64MB per page per layer at P=16384 -- so
// every shipped set carries tile_carry and never comes here.
func pagedFlashBatched(q: MLMultiArray, pool: PagePool, tile: MLModel,
                       s0GlobalPos: Int, sCount: Int, G: Int) throws -> [Float] {
    let P = pool.P
    let KVH = pool.kvHeads
    let dh = pool.dh
    let S = sCount
    let N = S * G                        // S*G query rows on the tile's N axis
    let npages = pool.pageCount
    var M = [Float](repeating: -.infinity, count: KVH * N)
    var L = [Float](repeating: 0, count: KVH * N)
    var O = [Float](repeating: 0, count: KVH * N * dh)
    for pi in 0 ..< npages {
        let pageStart = pi * P
        let valid = min(P, pool.length - pageStart)
        let out = try predict(tile, [
            "q": q, "K": pool.kPage(pi), "V": pool.vPage(pi),
            "mask": try bandMask(pageStart, valid, s0GlobalPos, S, P, G)])
        let mP = toFloats(out.featureValue(for: "m_out")!.multiArrayValue!)
        let lP = toFloats(out.featureValue(for: "l_out")!.multiArrayValue!)
        let oP = toFloats(out.featureValue(for: "o_out")!.multiArrayValue!)
        for kv in 0 ..< KVH {
            for n in 0 ..< N {
                let idx = kv * N + n
                let mNew = max(M[idx], mP[idx])
                let a = Foundation.exp(M[idx] - mNew)
                let b = Foundation.exp(mP[idx] - mNew)
                L[idx] = a * L[idx] + b * lP[idx]
                for d in 0 ..< dh {
                    O[idx * dh + d] = a * O[idx * dh + d]
                        + b * oP[kv * dh * N + d * N + n]  // o_out is dh-major
                }
                M[idx] = mNew
            }
        }
    }
    var result = [Float](repeating: 0, count: KVH * N * dh)
    for i in 0 ..< KVH * N {
        for d in 0 ..< dh { result[i * dh + d] = O[i * dh + d] / L[i] }
    }
    return result
}

// One IN-GRAPH-merge flash call over a single right-sized 512-key page: the
// block's own K/V [KVH,P,dh] (a fresh block at p0, no prior history), causal
// band. The carry seed (M=-1e4, L=0, O=0) makes the tile's in-graph online
// softmax return the merged running state for this one page, so there is NO
// host merge and NO 16384-oversized key axis. Final o = O/L (one host divide).
// Returns [KVH*N*dh] (kv,n row-major then dh), the same layout
// pagedFlashBatched returns.
func pagedFlashCarrySingle(q: MLMultiArray, K: MLMultiArray, V: MLMultiArray,
                           tile: MLModel, s0GlobalPos: Int, sCount: Int,
                           valid: Int, G: Int) throws -> [Float] {
    let P = K.shape[1].intValue
    let KVH = K.shape[0].intValue
    let dh = K.shape[2].intValue
    let S = sCount
    let N = S * G
    let mask = try bandMask(0, valid, s0GlobalPos, S, P, G)
    let out = try predict(tile, [
        "q": q, "K": K, "V": V, "mask": mask,
        "M_in": try seedArray([KVH, N], -1e4),
        "L_in": try seedArray([KVH, N], 0),
        "O_in": try seedArray([KVH, dh, N], 0)])
    let lP = toFloats(out.featureValue(for: "l_out")!.multiArrayValue!)
    let oP = toFloats(out.featureValue(for: "o_out")!.multiArrayValue!)  // dh-major
    var result = [Float](repeating: 0, count: KVH * N * dh)
    for kv in 0 ..< KVH {
        for n in 0 ..< N {
            let i = kv * N + n
            for d in 0 ..< dh {
                result[i * dh + d] = oP[kv * dh * N + d * N + n] / lP[i]
            }
        }
    }
    return result
}

// Full-context attention as a CHAIN of 512-key carry-tile calls with the
// online- softmax running state (M/L/O) threaded IN-GRAPH between tiles -- no
// host merge at any length. Walks the pool (prior turns + this block) in
// 512-windows; each tile updates the carried state, so context is unbounded
// and 100% on the NE. Final o = O/L (one host divide). Same [KVH*N*dh] layout
// as pagedFlashBatched.
func pagedFlashCarry(q: MLMultiArray, pool: PagePool, tile: MLModel,
                     s0GlobalPos: Int, sCount: Int, tileP: Int, G: Int)
    throws -> [Float] {
    let KVH = pool.kvHeads
    let dh = pool.dh
    let S = sCount
    let N = S * G
    let total = pool.length
    var mIn = try seedArray([KVH, N], -1e4)
    var lIn = try seedArray([KVH, N], 0)
    var oIn = try seedArray([KVH, dh, N], 0)
    var start = 0
    while start < total {
        let valid = min(tileP, total - start)
        let (K, V) = try pool.tileAt(start, valid, tileP)
        let mask = try bandMask(start, valid, s0GlobalPos, S, tileP, G)
        let out = try predict(tile, [
            "q": q, "K": K, "V": V, "mask": mask,
            "M_in": mIn, "L_in": lIn, "O_in": oIn])
        mIn = out.featureValue(for: "m_out")!.multiArrayValue!
        lIn = out.featureValue(for: "l_out")!.multiArrayValue!
        oIn = out.featureValue(for: "o_out")!.multiArrayValue!
        start += tileP
    }
    let lP = toFloats(lIn)
    let oP = toFloats(oIn)                                // dh-major
    var result = [Float](repeating: 0, count: KVH * N * dh)
    for kv in 0 ..< KVH {
        for n in 0 ..< N {
            let i = kv * N + n
            for d in 0 ..< dh {
                result[i * dh + d] = oP[kv * dh * N + d * N + n] / lP[i]
            }
        }
    }
    return result
}

// A [shape] fp16 array filled with one value (the carry-tile M/L/O seeds).
private func seedArray(_ shape: [Int], _ v: Float16) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) },
                             dataType: .float16)
    a.withUnsafeMutableBytes { buf, _ in
        let d = buf.bindMemory(to: Float16.self)
        for i in 0 ..< d.count { d[i] = v }
    }
    return a
}

// Per-(query-row s, page-local key p) causal/pad band [1,1,P,S*G] fp16 for one
// page: 0 where key global pos (pageStart+p) is at or before query global pos
// (p0+s) AND p is a written row (p < valid), else -1e4. The same value repeats
// across the G query heads of a kv head. History pages come out all-zero.
private func bandMask(_ pageStart: Int, _ valid: Int, _ p0: Int,
                      _ S: Int, _ P: Int, _ G: Int) throws -> MLMultiArray {
    let N = S * G
    let a = try MLMultiArray(
        shape: [1, 1, NSNumber(value: P), NSNumber(value: N)],
        dataType: .float16)
    a.withUnsafeMutableBytes { buf, _ in
        let d = buf.bindMemory(to: Float16.self)
        for p in 0 ..< P {
            for s in 0 ..< S {
                let ok = pageStart + p <= p0 + s && p < valid
                let val = Float16(ok ? 0 : -1e4)
                for g in 0 ..< G { d[p * N + s * G + g] = val }
            }
        }
    }
    return a
}

// Read an fp16 MLMultiArray into fp32 in logical C order, honouring strides
// (CoreML outputs may be padded, so a raw memcpy would misread).
private func toFloats(_ a: MLMultiArray) -> [Float] {
    let shape = a.shape.map { $0.intValue }
    let strides = a.strides.map { $0.intValue }
    let rank = shape.count
    let count = shape.reduce(1, *)
    var out = [Float](repeating: 0, count: count)
    // WHY: CoreML fp16 tile outputs are almost always C-contiguous -- convert
    // the whole buffer in one vectorized vImage half->single pass; fall back
    // to the strided element gather only when the layout is padded/permuted.
    var contiguous = true
    var expect = 1
    for r in stride(from: rank - 1, through: 0, by: -1) {
        if strides[r] != expect { contiguous = false }
        expect *= shape[r]
    }
    a.withUnsafeBytes { raw in
        let src = raw.bindMemory(to: UInt16.self)
        if contiguous {
            out.withUnsafeMutableBufferPointer { ob in
                var sBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: src.baseAddress!),
                    height: 1, width: vImagePixelCount(count),
                    rowBytes: count * 2)
                var dBuf = vImage_Buffer(data: ob.baseAddress!, height: 1,
                                         width: vImagePixelCount(count),
                                         rowBytes: count * 4)
                _ = vImageConvert_Planar16FtoPlanarF(&sBuf, &dBuf, 0)
            }
        } else {
            var idx = [Int](repeating: 0, count: rank)
            for flat in 0 ..< count {
                var off = 0
                for r in 0 ..< rank { off += idx[r] * strides[r] }
                out[flat] = Float(Float16(bitPattern: src[off]))
                var r = rank - 1
                idx[r] += 1
                while r > 0 && idx[r] == shape[r] {
                    idx[r] = 0
                    r -= 1
                    idx[r] += 1
                }
            }
        }
    }
    return out
}

private func predict(_ model: MLModel,
                     _ inputs: [String: MLMultiArray]) throws
    -> MLFeatureProvider {
    var dict: [String: MLFeatureValue] = [:]
    for (name, value) in inputs {
        dict[name] = MLFeatureValue(multiArray: value)
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: dict)
    return try model.prediction(from: provider)
}

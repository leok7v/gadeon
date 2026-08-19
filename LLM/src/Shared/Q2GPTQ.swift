import Accelerate
import Foundation

public enum Q2GPTQ {

    public static let damp = 0.01

    public static func inverseCholesky(_ h: [Float], _ k: Int,
                                       damp ridgeScale: Double = damp)
        -> [Float] {
        var a = [Double](repeating: 0, count: k * k)
        var trace = 0.0
        for i in 0..<k {
            let d = Double(h[i * k + i])
            trace += d > 0 ? d : 1.0
        }
        let ridge = ridgeScale * trace / Double(k)
        for i in 0..<k {
            for j in 0..<k { a[i * k + j] = Double(h[i * k + j]) }
            if a[i * k + i] <= 0 {
                for j in 0..<k where j != i {
                    a[i * k + j] = 0
                    a[j * k + i] = 0
                }
                a[i * k + i] = 1.0
            }
            a[i * k + i] += ridge
        }
        var n = __LAPACK_int(k)
        var lda = __LAPACK_int(k)
        var info = __LAPACK_int(0)
        var up = Int8(UInt8(ascii: "U"))
        a.withUnsafeMutableBufferPointer { ab in
            dpotrf_(&up, &n, ab.baseAddress!, &lda, &info)
            if info == 0 {
                dpotri_(&up, &n, ab.baseAddress!, &lda, &info)
            }
        }
        if info == 0 {
            for r in 0..<k {
                for c in (r + 1)..<k { a[r * k + c] = a[c * k + r] }
            }
            a.withUnsafeMutableBufferPointer { ab in
                dpotrf_(&up, &n, ab.baseAddress!, &lda, &info)
            }
        }
        var out = [Float](repeating: 0, count: k * k)
        for i in 0..<k where info == 0 {
            for j in i..<k { out[i * k + j] = Float(a[j * k + i]) }
        }
        return info == 0 ? out : []
    }

    public static func quantize(_ values: [Float], rows: Int, k: Int,
                                soa: Bool, opts: Q2Fit, ternary: Bool,
                                hinv: [Float]) -> ([UInt8], [Float]) {
        let nblk = k / Q2E8.qk
        var work = values
        var codes = [UInt8](repeating: 0, count: rows * k)
        var scales = [Float](repeating: 0, count: rows * nblk)
        var cuts = [Float](repeating: 0, count: rows * nblk)
        var blk = [Float](repeating: 0, count: rows * Q2E8.qk)
        var err = [Float](repeating: 0, count: rows * Q2E8.qk)
        var step = [Float](repeating: 0, count: rows)
        for b in 0..<nblk {
            let i1 = b * Q2E8.qk
            for r in 0..<rows {
                for c in 0..<Q2E8.qk {
                    blk[r * Q2E8.qk + c] = work[r * k + i1 + c]
                }
            }
            fitBlock(&blk, rows, i1, k, opts, ternary, &scales, &cuts, b, nblk)
            sweep(&blk, &err, &step, &codes, rows, k, i1, hinv, scales, cuts,
                  b, nblk, ternary, opts)
            spread(&work, err, rows, k, i1, hinv)
        }
        return pack(codes, scales, rows, k, soa, ternary, opts)
    }

    static func fitBlock(_ blk: inout [Float], _ rows: Int, _ i1: Int,
                         _ k: Int, _ opts: Q2Fit, _ ternary: Bool,
                         _ scales: inout [Float], _ cuts: inout [Float],
                         _ b: Int, _ nblk: Int) {
        let weights = opts.importance ?? []
        blk.withUnsafeMutableBufferPointer { bb in
        scales.withUnsafeMutableBufferPointer { sb in
        cuts.withUnsafeMutableBufferPointer { tb in
        weights.withUnsafeBufferPointer { wb in
                    nonisolated(unsafe) let bp = bb.baseAddress!
                    nonisolated(unsafe) let sp = sb.baseAddress!
                    nonisolated(unsafe) let tp = tb.baseAddress!
                    nonisolated(unsafe) let imp =
                        weights.count == k ? wb.baseAddress! + i1 : nil
                    DispatchQueue.concurrentPerform(iterations: rows) { r in
                        let at = bp + r * Q2E8.qk
                        if ternary {
                            let fit = Ternary.solve(at, Q2E8.qk, imp)
                            sp[r * nblk + b] = fit.scale
                            tp[r * nblk + b] = fit.cut
                        } else {
                            sp[r * nblk + b] = Q2E8.fit(at, opts, imp)
                        }
                    }
        }}}}
    }

    static func sweep(_ blk: inout [Float], _ err: inout [Float],
                      _ step: inout [Float], _ codes: inout [UInt8],
                      _ rows: Int, _ k: Int, _ i1: Int, _ hinv: [Float],
                      _ scales: [Float], _ cuts: [Float], _ b: Int,
                      _ nblk: Int, _ ternary: Bool, _ opts: Q2Fit) {
        let qk = Q2E8.qk
        for i in 0..<qk {
            let diag = hinv[(i1 + i) * k + i1 + i]
            for r in 0..<rows {
                let d = scales[r * nblk + b]
                let value = blk[r * qk + i]
                let code: UInt8
                let back: Float
                if ternary {
                    let live = abs(value) >= cuts[r * nblk + b]
                    code = live ? (value < 0 ? 0 : 2) : 1
                    back = (Float(code) - 1) * d
                } else {
                    let outer = Q2E8.book(d, opts)
                    let s = abs(d)
                    code = Q2E8.code(value, s, outer)
                    back = Q2E8.level(code, outer) * s
                }
                codes[r * k + i1 + i] = code
                step[r] = (value - back) / diag
                err[r * qk + i] = step[r]
            }
            let span = qk - i
            blk.withUnsafeMutableBufferPointer { bb in
                hinv.withUnsafeBufferPointer { hb in
                    cblas_sger(CblasRowMajor, Int32(rows), Int32(span), -1,
                               step, 1,
                               hb.baseAddress! + (i1 + i) * k + i1 + i, 1,
                               bb.baseAddress! + i, Int32(qk))
                }
            }
        }
    }

    static func spread(_ work: inout [Float], _ err: [Float], _ rows: Int,
                       _ k: Int, _ i1: Int, _ hinv: [Float]) {
        let i2 = i1 + Q2E8.qk
        let rest = k - i2
        if rest > 0 {
            work.withUnsafeMutableBufferPointer { wb in
                hinv.withUnsafeBufferPointer { hb in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(rows), Int32(rest), Int32(Q2E8.qk), -1,
                                err, Int32(Q2E8.qk),
                                hb.baseAddress! + i1 * k + i2, Int32(k), 1,
                                wb.baseAddress! + i2, Int32(k))
                }
            }
        }
    }

    static func pack(_ codes: [UInt8], _ scales: [Float], _ rows: Int,
                     _ k: Int, _ soa: Bool, _ ternary: Bool,
                     _ opts: Q2Fit) -> ([UInt8], [Float]) {
        let nblk = k / Q2E8.qk
        let rowBytes = nblk * Q2E8.blockBytes
        var packed = [UInt8](repeating: 0, count: rows * rowBytes)
        var back = [Float](repeating: 0, count: rows * k)
        codes.withUnsafeBufferPointer { cb in
        scales.withUnsafeBufferPointer { sb in
        packed.withUnsafeMutableBufferPointer { pb in
        back.withUnsafeMutableBufferPointer { bb in
                    nonisolated(unsafe) let cp = cb.baseAddress!
                    nonisolated(unsafe) let sp = sb.baseAddress!
                    nonisolated(unsafe) let dst = pb.baseAddress!
                    nonisolated(unsafe) let ref = bb.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: rows) { r in
                        for b in 0..<nblk {
                            let d = sp[r * nblk + b]
                            let at = r * k + b * Q2E8.qk
                            let half = Float16(d).bitPattern
                            let base = soa ? r * rowBytes + b * 2
                                           : r * rowBytes + b * Q2E8.blockBytes
                            dst[base] = UInt8(half & 0xFF)
                            dst[base + 1] = UInt8(half >> 8)
                            let codeAt = soa
                                ? r * rowBytes + nblk * 2 + b * 32
                                : r * rowBytes + b * Q2E8.blockBytes + 2
                            Q2E8.packBytes(cp + at, dst + codeAt)
                            let outer = Q2E8.book(d, opts)
                            let s = abs(d)
                            for i in 0..<Q2E8.qk {
                                ref[at + i] = ternary
                                    ? (Float(cp[at + i]) - 1) * d
                                    : Q2E8.level(cp[at + i], outer) * s
                            }
                        }
                    }
        }}}}
        return (packed, back)
    }

    public static func outputError(_ delta: [Float], _ want: [Float],
                                   rows: Int, k: Int, _ h: [Float]) -> Float {
        Float((quadratic(delta, rows, k, h) / max(quadratic(want, rows, k, h),
                                                  1e-30)).squareRoot())
    }

    public static func quadratic(_ w: [Float], _ rows: Int, _ k: Int,
                          _ h: [Float]) -> Double {
        var wh = [Float](repeating: 0, count: rows * k)
        w.withUnsafeBufferPointer { wb in
            h.withUnsafeBufferPointer { hb in
                wh.withUnsafeMutableBufferPointer { ob in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(rows), Int32(k), Int32(k), 1,
                                wb.baseAddress!, Int32(k),
                                hb.baseAddress!, Int32(k), 0,
                                ob.baseAddress!, Int32(k))
                }
            }
        }
        var acc = 0.0
        for i in 0..<(rows * k) { acc += Double(wh[i]) * Double(w[i]) }
        return acc
    }
}

public struct Spectrum: Sendable {
    public let k: Int
    public let condition: Double
    public let topShare: Double
    public let peakShare: Double
    public let participation: Double
}

public enum Eigen {

    public static func of(_ h: [Float], _ k: Int) -> Spectrum {
        return summarize(values(h, k), k)
    }

    static func summarize(_ ev: [Double], _ k: Int) -> Spectrum {
        let total = ev.reduce(0, +)
        let head = max(1, k / 100)
        let top = ev.suffix(head).reduce(0, +)
        let peak = ev.last ?? 0
        let floor = ev.first ?? 0
        let square = ev.reduce(0.0) { s, v in s + v * v }
        return Spectrum(
            k: k,
            condition: floor > 0 ? peak / floor : Double.infinity,
            topShare: total > 0 ? top / total : 0,
            peakShare: total > 0 ? peak / total : 0,
            participation: square > 0 ? total * total / square : 0)
    }
}

public enum LowRank {

    // The rank-r part of W, as W projected onto the top-r eigenvectors of
    // W^T W. Returned SEPARATELY so the caller quantizes only the remainder
    // and adds this back at full precision.

    public static func split(_ w: [Float], _ rows: Int, _ k: Int,
                             _ r: Int) -> ([Float], Double) {
        var g = [Float](repeating: 0, count: k * k)
        w.withUnsafeBufferPointer { wp in
            g.withUnsafeMutableBufferPointer { gp in
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            Int32(k), Int32(k), Int32(rows), 1,
                            wp.baseAddress!, Int32(k),
                            wp.baseAddress!, Int32(k), 0,
                            gp.baseAddress!, Int32(k))
            }
        }
        let (values, vectors) = basis(g, k)
        var vk = [Float](repeating: 0, count: r * k)
        for j in 0..<r {
            let row = k - r + j
            for c in 0..<k { vk[j * k + c] = Float(vectors[row * k + c]) }
        }
        var a = [Float](repeating: 0, count: rows * r)
        var lr = [Float](repeating: 0, count: rows * k)
        w.withUnsafeBufferPointer { wp in
        vk.withUnsafeBufferPointer { vp in
        a.withUnsafeMutableBufferPointer { ap in
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(rows), Int32(r), Int32(k), 1,
                        wp.baseAddress!, Int32(k),
                        vp.baseAddress!, Int32(k), 0,
                        ap.baseAddress!, Int32(r))
        }}}
        a.withUnsafeBufferPointer { ap in
        vk.withUnsafeBufferPointer { vp in
        lr.withUnsafeMutableBufferPointer { lp in
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(rows), Int32(k), Int32(r), 1,
                        ap.baseAddress!, Int32(r),
                        vp.baseAddress!, Int32(k), 0,
                        lp.baseAddress!, Int32(k))
        }}}
        return (lr, values[0..<(k - r)].reduce(0) { s, v in s + max(v, 0) })
    }

    public static let ranks = [1, 2, 4, 8, 16, 32, 64]
    nonisolated(unsafe) public static var lastSharedTotal = 0.0

    public static func basis(_ h: [Float], _ k: Int) -> ([Double], [Double]) {
        var a = [Double](repeating: 0, count: k * k)
        for i in 0..<(k * k) { a[i] = Double(h[i]) }
        var w = [Double](repeating: 0, count: k)
        var n = __LAPACK_int(k)
        var lda = __LAPACK_int(k)
        var info = __LAPACK_int(0)
        var jobz = Int8(UInt8(ascii: "V"))
        var uplo = Int8(UInt8(ascii: "U"))
        var probe = Double(0)
        var iprobe = __LAPACK_int(0)
        var lwork = __LAPACK_int(-1)
        var liwork = __LAPACK_int(-1)
        a.withUnsafeMutableBufferPointer { ab in
            w.withUnsafeMutableBufferPointer { wb in
                dsyevd_(&jobz, &uplo, &n, ab.baseAddress!, &lda,
                        wb.baseAddress!, &probe, &lwork, &iprobe, &liwork,
                        &info)
            }
        }
        var work = [Double](repeating: 0, count: max(Int(probe), 1))
        var iwork = [__LAPACK_int](repeating: 0, count: max(Int(iprobe), 1))
        lwork = __LAPACK_int(work.count)
        liwork = __LAPACK_int(iwork.count)
        a.withUnsafeMutableBufferPointer { ab in
            w.withUnsafeMutableBufferPointer { wb in
                work.withUnsafeMutableBufferPointer { kb in
                    iwork.withUnsafeMutableBufferPointer { ib in
                        dsyevd_(&jobz, &uplo, &n, ab.baseAddress!, &lda,
                                wb.baseAddress!, kb.baseAddress!, &lwork,
                                ib.baseAddress!, &liwork, &info)
                    }
                }
            }
        }
        return (w, a)
    }

    public static func shared(_ delta: [Float], _ rows: Int, _ k: Int,
                              _ values: [Double],
                              _ vectors: [Double]) -> [Double] {
        var b = [Double](repeating: 0, count: rows * k)
        var d = [Double](repeating: 0, count: rows * k)
        for i in 0..<(rows * k) { d[i] = Double(delta[i]) }
        d.withUnsafeBufferPointer { dp in
            vectors.withUnsafeBufferPointer { vp in
                b.withUnsafeMutableBufferPointer { bp in
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                Int32(rows), Int32(k), Int32(k), 1,
                                dp.baseAddress!, Int32(k),
                                vp.baseAddress!, Int32(k), 0,
                                bp.baseAddress!, Int32(k))
                }
            }
        }
        var share = [Double](repeating: 0, count: k)
        for j in 0..<k {
            var acc = 0.0
            for r in 0..<rows {
                let v = b[r * k + j]
                acc += v * v
            }
            share[j] = max(values[j], 0) * acc
        }
        lastSharedTotal = share.reduce(0, +)
        let total = lastSharedTotal
        return ranks.map { r in
            let cut = share.suffix(min(r, k)).reduce(0, +)
            return total > 0 ? max(total - cut, 0) / total : 0
        }
    }

    public static func remaining(_ delta: [Float], _ rows: Int, _ k: Int,
                                 _ values: [Double],
                                 _ vectors: [Double]) -> [Double] {
        var b = [Double](repeating: 0, count: rows * k)
        var d = [Double](repeating: 0, count: rows * k)
        for i in 0..<(rows * k) { d[i] = Double(delta[i]) }
        d.withUnsafeBufferPointer { dp in
            vectors.withUnsafeBufferPointer { vp in
                b.withUnsafeMutableBufferPointer { bp in
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                Int32(rows), Int32(k), Int32(k), 1,
                                dp.baseAddress!, Int32(k),
                                vp.baseAddress!, Int32(k), 0,
                                bp.baseAddress!, Int32(k))
                }
            }
        }
        for j in 0..<k {
            let s = max(values[j], 0).squareRoot()
            for r in 0..<rows { b[r * k + j] *= s }
        }
        var s = [Double](repeating: 0, count: k * k)
        b.withUnsafeBufferPointer { bp in
            s.withUnsafeMutableBufferPointer { sp in
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            Int32(k), Int32(k), Int32(rows), 1,
                            bp.baseAddress!, Int32(k),
                            bp.baseAddress!, Int32(k), 0,
                            sp.baseAddress!, Int32(k))
            }
        }
        var sq = [Float](repeating: 0, count: k * k)
        for i in 0..<(k * k) { sq[i] = Float(s[i]) }
        let spec = Eigen.values(sq, k)
        let total = spec.reduce(0, +)
        return ranks.map { r in
            let cut = spec.suffix(min(r, k)).reduce(0, +)
            return total > 0 ? max(total - cut, 0) / total : 0
        }
    }
}

extension Eigen {
    public static func values(_ h: [Float], _ k: Int) -> [Double] {
        var a = [Double](repeating: 0, count: k * k)
        for i in 0..<(k * k) { a[i] = Double(h[i]) }
        var w = [Double](repeating: 0, count: k)
        var n = __LAPACK_int(k)
        var lda = __LAPACK_int(k)
        var info = __LAPACK_int(0)
        var jobz = Int8(UInt8(ascii: "N"))
        var uplo = Int8(UInt8(ascii: "U"))
        var probe = Double(0)
        var iprobe = __LAPACK_int(0)
        var lwork = __LAPACK_int(-1)
        var liwork = __LAPACK_int(-1)
        a.withUnsafeMutableBufferPointer { ab in
            w.withUnsafeMutableBufferPointer { wb in
                dsyevd_(&jobz, &uplo, &n, ab.baseAddress!, &lda,
                        wb.baseAddress!, &probe, &lwork, &iprobe, &liwork,
                        &info)
            }
        }
        var work = [Double](repeating: 0, count: max(Int(probe), 1))
        var iwork = [__LAPACK_int](repeating: 0, count: max(Int(iprobe), 1))
        lwork = __LAPACK_int(work.count)
        liwork = __LAPACK_int(iwork.count)
        a.withUnsafeMutableBufferPointer { ab in
            w.withUnsafeMutableBufferPointer { wb in
                work.withUnsafeMutableBufferPointer { kb in
                    iwork.withUnsafeMutableBufferPointer { ib in
                        dsyevd_(&jobz, &uplo, &n, ab.baseAddress!, &lda,
                                wb.baseAddress!, kb.baseAddress!, &lwork,
                                ib.baseAddress!, &liwork, &info)
                    }
                }
            }
        }
        return w
    }
}

public enum Hadamard {

    public static func blockSize(_ k: Int) -> Int {
        var b = 1
        while b * 2 <= k && k % (b * 2) == 0 { b *= 2 }
        return b
    }

    public static func signs(_ k: Int) -> [Float] {
        var out = [Float](repeating: 1, count: k)
        var s = UInt64(k) &+ 0x9E37_79B9_7F4A_7C15
        for i in 0..<k {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            out[i] = (s >> 33) & 1 == 0 ? -1 : 1
        }
        return out
    }

    static func fwht(_ p: UnsafeMutablePointer<Float>, _ n: Int) {
        var len = 1
        while len < n {
            var i = 0
            while i < n {
                for j in i..<(i + len) {
                    let a = p[j]
                    let b = p[j + len]
                    p[j] = a + b
                    p[j + len] = a - b
                }
                i += len * 2
            }
            len *= 2
        }
        let s = 1 / Float(n).squareRoot()
        for i in 0..<n { p[i] *= s }
    }

    public static func rows(_ m: inout [Float], _ rows: Int, _ k: Int,
                            _ sign: [Float]) {
        let block = blockSize(k)
        m.withUnsafeMutableBufferPointer { mb in
            sign.withUnsafeBufferPointer { sb in
                nonisolated(unsafe) let p = mb.baseAddress!
                nonisolated(unsafe) let g = sb.baseAddress!
                DispatchQueue.concurrentPerform(iterations: rows) { r in
                    let row = p + r * k
                    for i in 0..<k { row[i] *= g[i] }
                    var at = 0
                    while at < k {
                        fwht(row + at, block)
                        at += block
                    }
                }
            }
        }
    }

    public static func cols(_ m: inout [Float], _ rows: Int, _ k: Int,
                           _ sign: [Float]) {
        let block = blockSize(rows)
        m.withUnsafeMutableBufferPointer { mb in
            sign.withUnsafeBufferPointer { sb in
                nonisolated(unsafe) let p = mb.baseAddress!
                nonisolated(unsafe) let g = sb.baseAddress!
                DispatchQueue.concurrentPerform(iterations: k) { c in
                    var v = [Float](repeating: 0, count: rows)
                    v.withUnsafeMutableBufferPointer { vb in
                        let q = vb.baseAddress!
                        for j in 0..<rows { q[j] = p[j * k + c] * g[j] }
                        var at = 0
                        while at < rows {
                            fwht(q + at, block)
                            at += block
                        }
                        for j in 0..<rows { p[j * k + c] = q[j] }
                    }
                }
            }
        }
    }

    static func transpose(_ m: [Float], _ n: Int) -> [Float] {
        var out = [Float](repeating: 0, count: n * n)
        vDSP_mtrans(m, 1, &out, 1, vDSP_Length(n), vDSP_Length(n))
        return out
    }

    public static func congruence(_ h: inout [Float], _ k: Int,
                                  _ sign: [Float]) {
        rows(&h, k, k, sign)
        var t = transpose(h, k)
        rows(&t, k, k, sign)
        h = transpose(t, k)
    }
}

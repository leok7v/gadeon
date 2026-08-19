import Accelerate
import Foundation

public struct E8Stats: Sendable {
    public var vectors = 0
    public var clamped = 0
    public var peak: Float = 0
    // Only the padded (E8P) GPTQ path fills these; they are what Q2_E8 packs.
    public var codes: [UInt16] = []
    public var scales: [Float] = []
}

public enum E8 {
    public static let dim = 8
    public static let shell: Float = 10
    // E8P's table reaches norm^2 12 and the +-1/4 offset can add
    // another 4.5, so its points are bounded at 16.5, not by shell.
    public static func bound(_ padded: Bool) -> Float {
        padded ? 16.5 : shell
    }

    static func norm2(_ v: SIMD8<Float>) -> Float { (v * v).sum() }

    // Conway and Sloane: the closest point of D8 is the rounded vector with
    // its parity repaired at the coordinate that rounded worst.

    static func roundD8(_ x: SIMD8<Float>) -> SIMD8<Float> {
        var r = x.rounded(.toNearestOrEven)
        var parity = 0
        for i in 0..<dim { parity += Int(r[i]) }
        if parity % 2 != 0 {
            var worst = 0
            var gap: Float = -1
            for i in 0..<dim where abs(x[i] - r[i]) > gap {
                gap = abs(x[i] - r[i])
                worst = i
            }
            r[worst] += x[worst] > r[worst] ? 1 : -1
        }
        return r
    }

    static func nearest(_ x: SIMD8<Float>) -> SIMD8<Float> {
        let half = SIMD8<Float>(repeating: 0.5)
        let a = roundD8(x)
        let b = roundD8(x - half) + half
        return norm2(x - a) <= norm2(x - b) ? a : b
    }

    // The codebook is the lattice INSIDE the shell, so a vector that decodes
    // outside it is walked back toward the origin until it lands.

    static func bounded(_ x: SIMD8<Float>, _ hit: inout Int) -> SIMD8<Float> {
        var c = nearest(x)
        var y = x
        var tries = 0
        while norm2(c) > shell && tries < 40 {
            y *= (shell / norm2(c)).squareRoot() * 0.99
            c = nearest(y)
            tries += 1
        }
        if tries > 0 { hit += 1 }
        return norm2(c) > shell ? SIMD8<Float>() : c
    }

    // mu-law companding. E8P's codebook has a HOLE at the origin, so an
    // 8-vector far below the block's fitted scale lands in it. Compressing
    // before the lattice and expanding after pulls those out; the expansion
    // folds into the decode table, so it costs nothing at inference.

    public static let reference: Float = 3

    static func compand(_ t: SIMD8<Float>, _ mu: Float) -> SIMD8<Float> {
        var out = SIMD8<Float>()
        let denom = log(1 + mu)
        for i in 0..<dim {
            let a = abs(t[i])
            let v = reference * log(1 + mu * a / reference) / denom
            out[i] = t[i] < 0 ? -v : v
        }
        return out
    }

    static func expand(_ u: SIMD8<Float>, _ mu: Float) -> SIMD8<Float> {
        var out = SIMD8<Float>()
        for i in 0..<dim {
            let a = abs(u[i])
            let v = reference * (pow(1 + mu, a / reference) - 1) / mu
            out[i] = u[i] < 0 ? -v : v
        }
        return out
    }

    static func load(_ p: UnsafePointer<Float>) -> SIMD8<Float> {
        var out = SIMD8<Float>()
        for i in 0..<dim { out[i] = p[i] }
        return out
    }

    // `code` is the packable 16-bit E8P index, or 0xFFFF for a variant that
    // has no packed form (the shell codebook, or companding, whose expanded
    // point is not a lattice point). pack+unpack costs the same as nearest --
    // `choose` dominates both -- so the index is free on the padded path.

    static func point(_ x: SIMD8<Float>, _ padded: Bool, _ mu: Float,
                      _ hit: inout Int,
                      _ code: inout UInt16) -> SIMD8<Float> {
        var out: SIMD8<Float>
        code = 0xFFFF
        if mu > 0 && padded {
            out = expand(E8P.nearest(compand(x, mu)), mu)
        } else if padded {
            code = E8P.pack(x)
            out = E8P.unpack(code)
        } else {
            out = bounded(x, &hit)
        }
        return out
    }

    static func assign(_ block: UnsafePointer<Float>, _ n: Int, _ d: Float,
                       _ out: UnsafeMutablePointer<Float>,
                       _ hit: inout Int, _ padded: Bool,
                       _ mu: Float,
                       _ codes: UnsafeMutablePointer<UInt16>?) {
        let inv = 1 / d
        var code: UInt16 = 0
        for v in 0..<(n / dim) {
            let c = point(load(block + v * dim) * inv, padded, mu, &hit, &code)
            for i in 0..<dim { out[v * dim + i] = c[i] }
            if let codes { codes[v] = code }
        }
    }

    // A multiplicative refit COLLAPSES here: when the scale is large enough
    // that every vector decodes to the origin the denominator is zero, the
    // next scale is the floor, and the one after that decodes outside any
    // shell. Search a bounded grid instead, so no iterate can leave it.

    static func fit(_ block: UnsafePointer<Float>, _ n: Int, _ opts: Q2Fit,
                    _ weight: UnsafePointer<Float>?,
                    _ padded: Bool, _ mu: Float) -> Float {
        var square: Float = 0
        for i in 0..<n { square += block[i] * block[i] }
        let seed = max((square / Float(n)).squareRoot() / 0.9, 1e-12)
        var best = seed
        var mark = Float.infinity
        var code = [Float](repeating: 0, count: n)
        var hit = 0
        code.withUnsafeMutableBufferPointer { cb in
            let c = cb.baseAddress!
            for j in -6...6 {
                let d = seed * pow(1.18, Float(j))
                assign(block, n, d, c, &hit, padded, mu, nil)
                var acc: Float = 0
                for i in 0..<n {
                    let gap = block[i] - c[i] * d
                    acc += (weight?[i] ?? 1) * gap * gap
                }
                if acc < mark {
                    mark = acc
                    best = d
                }
            }
        }
        return best
    }

    // The whole 128-block is quantized against the codebook, then ONE
    // propagation carries its error to later columns. Feeding each group of
    // 8 through U[g+t, .] as well double-applies what the 8x8 diagonal block
    // should absorb.

    static func scales(_ blk: inout [Float], _ rows: Int, _ i1: Int,
                       _ k: Int, _ opts: Q2Fit, _ scale: inout [Float],
                       _ padded: Bool, _ mu: Float) {
        let qk = Q2E8.qk
        let weights = opts.importance ?? []
        blk.withUnsafeMutableBufferPointer { bb in
        scale.withUnsafeMutableBufferPointer { sb in
        weights.withUnsafeBufferPointer { wb in
                    nonisolated(unsafe) let bp = bb.baseAddress!
                    nonisolated(unsafe) let sp = sb.baseAddress!
                    nonisolated(unsafe) let imp =
                        weights.count == k ? wb.baseAddress! + i1 : nil
                    DispatchQueue.concurrentPerform(iterations: rows) { r in
                        sp[r] = fit(bp + r * qk, qk, opts, imp,
                                    padded, mu)
                    }
        }}}
    }

    static func group(_ blk: inout [Float], _ err: inout [Float],
                      _ back: inout [Float], _ rows: Int, _ k: Int,
                      _ i1: Int, _ g: Int, _ hinv: [Float],
                      _ scale: [Float], _ hits: inout [Int],
                      _ peaks: inout [Float], _ padded: Bool,
                      _ mu: Float, _ codes: inout [UInt16],
                      _ scale16: inout [Float]) {
        let qk = Q2E8.qk
        blk.withUnsafeMutableBufferPointer { bb in
        err.withUnsafeMutableBufferPointer { eb in
        back.withUnsafeMutableBufferPointer { kb in
        hits.withUnsafeMutableBufferPointer { hb in
        peaks.withUnsafeMutableBufferPointer { pb in
        codes.withUnsafeMutableBufferPointer { cb in
        hinv.withUnsafeBufferPointer { ib in
        scale.withUnsafeBufferPointer { sb in
                    nonisolated(unsafe) let bp = bb.baseAddress!
                    nonisolated(unsafe) let ep = eb.baseAddress!
                    nonisolated(unsafe) let kp = kb.baseAddress!
                    nonisolated(unsafe) let hp = hb.baseAddress!
                    nonisolated(unsafe) let pp = pb.baseAddress!
                    nonisolated(unsafe) let cp = cb.baseAddress!
                    nonisolated(unsafe) let ip = ib.baseAddress!
                    nonisolated(unsafe) let sp = sb.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: rows) { r in
                        let d = sp[r]
                        let at = bp + r * qk + g
                        var hit = 0
                        var code: UInt16 = 0
                        let c = point(load(at) * (1 / d), padded, mu, &hit,
                                      &code)
                        cp[r * (k / dim) + (i1 + g) / dim] = code
                        hp[r] += hit
                        pp[r] = max(pp[r], norm2(c))
                        for t in 0..<dim {
                            let q = c[t] * d
                            kp[r * k + i1 + g + t] = q
                            let seat = (i1 + g + t) * k + i1 + g + t
                            ep[r * qk + g + t] = (at[t] - q) / ip[seat]
                        }
                    }
        }}}}}}}}
    }

    public static func gptq(_ values: [Float], rows: Int, k: Int,
                            opts: Q2Fit, hinv: [Float],
                            padded: Bool,
                            mu: Float) -> ([Float], E8Stats) {
        let qk = Q2E8.qk
        let nblk = k / qk
        var work = values
        var back = [Float](repeating: 0, count: rows * k)
        var blk = [Float](repeating: 0, count: rows * qk)
        var err = [Float](repeating: 0, count: rows * qk)
        var scale = [Float](repeating: 0, count: rows)
        var hits = [Int](repeating: 0, count: rows)
        var peaks = [Float](repeating: 0, count: rows)
        var codes = [UInt16](repeating: 0, count: rows * k / dim)
        var scale16 = [Float](repeating: 0, count: rows * nblk)
        for b in 0..<nblk {
            let i1 = b * qk
            for r in 0..<rows {
                for c in 0..<qk { blk[r * qk + c] = work[r * k + i1 + c] }
            }
            scales(&blk, rows, i1, k, opts, &scale, padded, mu)
            for g in stride(from: 0, to: qk, by: dim) {
                group(&blk, &err, &back, rows, k, i1, g, hinv, scale,
                      &hits, &peaks, padded, mu, &codes, &scale16)
                let span = qk - g - dim
                if span > 0 {
                    blk.withUnsafeMutableBufferPointer { bb in
                    err.withUnsafeBufferPointer { ep in
                    hinv.withUnsafeBufferPointer { ib in
                        cblas_sgemm(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            Int32(rows), Int32(span), Int32(dim), -1,
                            ep.baseAddress! + g, Int32(qk),
                            ib.baseAddress! + (i1 + g) * k + i1 + g + dim,
                            Int32(k), 1,
                            bb.baseAddress! + g + dim, Int32(qk))
                    }}}
                }
            }
            for r in 0..<rows { scale16[r * nblk + b] = scale[r] }
            Q2GPTQ.spread(&work, err, rows, k, i1, hinv)
        }
        var stats = E8Stats()
        stats.vectors = rows * k / dim
        stats.clamped = hits.reduce(0, +)
        stats.peak = peaks.max() ?? 0
        stats.codes = codes
        stats.scales = scale16
        return (back, stats)
    }

    public static func quantize(_ values: [Float], rows: Int, k: Int,
                                opts: Q2Fit, padded: Bool,
                                mu: Float) -> ([Float], E8Stats) {
        let qk = Q2E8.qk
        let nblk = k / qk
        var back = [Float](repeating: 0, count: rows * k)
        var hits = [Int](repeating: 0, count: rows)
        var peaks = [Float](repeating: 0, count: rows)
        let weights = opts.importance ?? []
        values.withUnsafeBufferPointer { vb in
        back.withUnsafeMutableBufferPointer { bb in
        hits.withUnsafeMutableBufferPointer { hb in
        peaks.withUnsafeMutableBufferPointer { kb in
        weights.withUnsafeBufferPointer { wb in
                    nonisolated(unsafe) let src = vb.baseAddress!
                    nonisolated(unsafe) let ref = bb.baseAddress!
                    nonisolated(unsafe) let hip = hb.baseAddress!
                    nonisolated(unsafe) let pkp = kb.baseAddress!
                    nonisolated(unsafe) let imp =
                        weights.count == k ? wb.baseAddress : nil
                    DispatchQueue.concurrentPerform(iterations: rows) { r in
                        var code = [Float](repeating: 0, count: qk)
                        var hit = 0
                        var peak: Float = 0
                        code.withUnsafeMutableBufferPointer { cb in
                            let c = cb.baseAddress!
                            for b in 0..<nblk {
                                let at = r * k + b * qk
                                let block = src + at
                                let w = imp.map { p in p + b * qk }
                                let d = fit(block, qk, opts, w,
                                            padded, mu)
                                assign(block, qk, d, c, &hit, padded,
                                       mu, nil)
                                for v in 0..<(qk / dim) {
                                    peak = max(peak, norm2(load(c + v * dim)))
                                }
                                for i in 0..<qk { ref[at + i] = c[i] * d }
                            }
                        }
                        hip[r] = hit
                        pkp[r] = peak
                    }
        }}}}}
        var stats = E8Stats()
        stats.vectors = rows * k / dim
        stats.clamped = hits.reduce(0, +)
        stats.peak = peaks.max() ?? 0
        return (back, stats)
    }
}

// Q2_E8 on disk: SoA within a row, matching the layout Q2E8.quantize already
// uses for its soa arm -- nblk fp16 scales, then nblk groups of 16 uint16
// codes. 2*nblk + 32*nblk = 34*nblk bytes, byte for byte the Q2_X block, so
// every size calculation and the 16 KB tensor alignment carry over.

public enum Q2E8Pack {
    public static func e8p(_ codes: [UInt16], _ scales: [Float],
                           rows: Int, k: Int) -> [UInt8] {
        let qk = Q2E8.qk
        let nblk = k / qk
        let per = qk / E8.dim
        let rowBytes = nblk * Q2E8.blockBytes
        var out = [UInt8](repeating: 0, count: rows * rowBytes)
        out.withUnsafeMutableBufferPointer { ob in
            let p = ob.baseAddress!
            for r in 0..<rows {
                let base = r * rowBytes
                for b in 0..<nblk {
                    let half = Float16(scales[r * nblk + b]).bitPattern
                    p[base + b * 2] = UInt8(half & 0xFF)
                    p[base + b * 2 + 1] = UInt8(half >> 8)
                    let at = base + nblk * 2 + b * 32
                    for v in 0..<per {
                        let c = codes[r * (k / E8.dim) + b * per + v]
                        p[at + v * 2] = UInt8(c & 0xFF)
                        p[at + v * 2 + 1] = UInt8(c >> 8)
                    }
                }
            }
        }
        return out
    }
}

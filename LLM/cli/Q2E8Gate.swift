import Foundation
import LLM

func runGPTQGate(_ args: [String]) {
    let k = 96
    var s: UInt64 = 0x2545_F491_4F6C_DD1D
    func next() -> Float {
        s = s &* 6364136223846793005 &+ 1442695040888963407
        // s >> 33 caps at 2^31-1, so dividing THAT by UInt32.max yields
        // [0, 0.5) and the old `* 2 - 1` gave [-1, 0) -- never positive.
        return Float(UInt32(truncatingIfNeeded: s >> 32)) / Float(UInt32.max)
            * 2 - 1
    }
    var x = [Float](repeating: 0, count: k * k)
    for i in 0..<(k * k) { x[i] = next() }
    var h = [Float](repeating: 0, count: k * k)
    for i in 0..<k {
        for j in 0..<k {
            var acc: Float = 0
            for t in 0..<k { acc += x[t * k + i] * x[t * k + j] }
            h[i * k + j] = acc
        }
        h[i * k + i] += Float(k)
    }
    let u = Q2GPTQ.inverseCholesky(h, k, damp: 0)
    var utu = [Float](repeating: 0, count: k * k)
    for i in 0..<k {
        for j in 0..<k {
            var acc: Float = 0
            for t in 0...min(i, j) { acc += u[t * k + i] * u[t * k + j] }
            utu[i * k + j] = acc
        }
    }
    var worst: Float = 0
    var peak: Float = 0
    for i in 0..<k {
        for j in 0..<k {
            var acc: Float = 0
            for t in 0..<k { acc += h[i * k + t] * utu[t * k + j] }
            let want: Float = i == j ? 1 : 0
            worst = max(worst, abs(acc - want))
            peak = max(peak, abs(utu[i * k + j]))
        }
    }
    let lk = 64, lrows = 40
    var lh = [Float](repeating: 0, count: lk * lk)
    for i in 0..<lk { lh[i * lk + i] = 1 }
    var ld = [Float](repeating: 0, count: lrows * lk)
    for (t, sigma) in [Float(3), 2, 1].enumerated() {
        ld[(t + 5) * lk + (t * 7 + 2)] = sigma
    }
    for hk in [64, 96, 128] {
        var hh = [Float](repeating: 0, count: hk * hk)
        var hx = [Float](repeating: 0, count: hk * hk)
        for i in 0..<(hk * hk) { hx[i] = next() }
        for i in 0..<hk {
            for j in 0..<hk {
                var acc: Float = 0
                for t in 0..<hk { acc += hx[t * hk + i] * hx[t * hk + j] }
                hh[i * hk + j] = acc
            }
            hh[i * hk + i] += Float(hk)
        }
        var hd = [Float](repeating: 0, count: 24 * hk)
        for i in 0..<hd.count { hd[i] = next() }
        let before = Q2GPTQ.quadratic(hd, 24, hk, hh)
        let sign = Hadamard.signs(hk)
        var rd = hd
        var rhh = hh
        Hadamard.rows(&rd, 24, hk, sign)
        Hadamard.congruence(&rhh, hk, sign)
        let after = Q2GPTQ.quadratic(rd, 24, hk, rhh)
        let drift = abs(after - before) / max(before, 1e-30)
        FileHandle.standardError.write(Data(
            String(format: "[hadamard] K=%d block=%d  tr(DHD^T) %.6e -> "
                   + "%.6e  rel %.2e  %@\n", hk, Hadamard.blockSize(hk),
                   before, after, drift,
                   drift < 1e-4 ? "PASS" : "FAIL").utf8))
    }
    var rh = [Float](repeating: 0, count: lk * lk)
    var rx = [Float](repeating: 0, count: lk * lk)
    for i in 0..<(lk * lk) { rx[i] = next() }
    for i in 0..<lk {
        for j in 0..<lk {
            var acc: Float = 0
            for t in 0..<lk { acc += rx[t * lk + i] * rx[t * lk + j] }
            rh[i * lk + j] = acc
        }
        rh[i * lk + i] += Float(lk)
    }
    var rd = [Float](repeating: 0, count: lrows * lk)
    for i in 0..<rd.count { rd[i] = next() }
    let rbasis = LowRank.basis(rh, lk)
    _ = LowRank.shared(rd, lrows, lk, rbasis.0, rbasis.1)
    let viaEigen = LowRank.lastSharedTotal
    let direct = Q2GPTQ.quadratic(rd, lrows, lk, rh)
    let rel = abs(viaEigen - direct) / max(direct, 1e-30)
    FileHandle.standardError.write(Data(
        String(format: "[lowrank] eigen total %.6e  direct %.6e  rel %.2e  %@\n",
               viaEigen, direct, rel, rel < 1e-5 ? "PASS" : "FAIL").utf8))
    let basis = LowRank.basis(lh, lk)
    let left = LowRank.remaining(ld, lrows, lk, basis.0, basis.1)
    let want = [5.0 / 14, 1.0 / 14, 0.0]
    var gap = 0.0
    for (n, r) in [0, 1, 2].enumerated() { gap = max(gap, abs(left[r] - want[n])) }
    FileHandle.standardError.write(Data(
        String(format: "[lowrank] sv 3,2,1: r1 %.6f (want %.6f)  r2 %.6f "
               + "(want %.6f)  r4 %.2e  max gap %.2e  %@\n",
               left[0], want[0], left[1], want[1], left[2], gap,
               gap < 1e-6 ? "PASS" : "FAIL").utf8))
    var wv = [Float](repeating: 0, count: 128)
    var hv = [Float](repeating: 0, count: 128)
    for i in 0..<128 { wv[i] = next() }
    for i in 0..<128 { hv[i] = abs(next()) + 0.05 }
    wv.withUnsafeBufferPointer { wb in
        hv.withUnsafeBufferPointer { hb in
            let fit = Ternary.solve(wb.baseAddress!, 128, hb.baseAddress!)
            var live = 0
            for i in 0..<128 where abs(wv[i]) >= fit.cut { live += 1 }
            FileHandle.standardError.write(Data(
                String(format: "[tern] scale %.9f  cut %.9f  live %d/128\n",
                       fit.scale, fit.cut, live).utf8))
        }
    }
    // Two populations. Flat: every 8-vector at the same scale, which is what
    // a matched codebook is designed for. Spread: each vector drawn from a
    // log-uniform scale over 30x, which is what a real 128-weight block looks
    // like and is where E8P's hole at the origin should bite.
    for (shape, spread) in [("flat", Float(0)), ("spread", Float(30))] {
        var data = [Float](repeating: 0, count: 128 * 64)
        for v in 0..<(128 * 64 / 8) {
            var scale: Float = 1
            if spread > 0 {
                scale = pow(spread, next() * 0.5 - 0.5)
            }
            for j in 0..<8 {
                var u: Float = 0
                for _ in 0..<12 { u += next() }
                data[v * 8 + j] = u * 0.5 * scale
            }
        }
        let opts = Q2Fit(iterations: 8, renorm: false)
        var line = ""
        for (tag, book, mu) in [("shell", false, Float(0)),
                                ("E8P", true, Float(0)),
                                ("mu1", true, Float(1)),
                                ("mu4", true, Float(4)),
                                ("mu16", true, Float(16))] {
            let got = E8.quantize(data, rows: 64, k: 128, opts: opts,
                                  padded: book, mu: mu)
            var num = 0.0
            var den = 0.0
            for i in 0..<data.count {
                let d = Double(data[i] - got.0[i])
                num += d * d
                den += Double(data[i]) * Double(data[i])
            }
            line += String(format: "  %@ %.4f", tag, (num / den).squareRoot())
        }
        FileHandle.standardError.write(Data(
            ("[book] " + shape + ":" + line + "\n").utf8))
    }
    let lower = (0..<k).reduce(Float(0)) { m, i in
        (0..<i).reduce(m) { n, j in max(n, abs(u[i * k + j])) }
    }
    FileHandle.standardError.write(Data(
        String(format: "[gptq] k=%d  max|H(U^T U) - I| %.3e  "
               + "max|below diagonal| %.3e  peak %.3e  %@\n",
               k, worst, lower, peak,
               worst < 1e-3 && lower == 0 ? "PASS" : "FAIL").utf8))
    exit(worst < 1e-3 && lower == 0 ? 0 : 1)
}

func runQ2E8Gate(_ args: [String]) {
    let rows = 64
    let k = 512
    var values = [Float](repeating: 0, count: rows * k)
    var s: UInt64 = 0x9E37_79B9_7F4A_7C15
    for i in 0..<values.count {
        s = s &* 6364136223846793005 &+ 1442695040888963407
        let bits = UInt32(truncatingIfNeeded: s >> 33)
        values[i] = (Float(bits) / Float(UInt32.max) * 2 - 1) * 0.05
    }
    let dir = args.firstIndex(of: "--q2e8-gate").map { i in
        i + 1 < args.count ? args[i + 1] : "tmp/q2e8-gate"
    } ?? "tmp/q2e8-gate"
    try? FileManager.default.createDirectory(
        atPath: dir, withIntermediateDirectories: true)
    values.withUnsafeBytes { raw in
        try? Data(raw).write(to: URL(fileURLWithPath: dir + "/values.bin"))
    }
    for (tag, soa) in [("aos", false), ("soa", true)] {
        for (fitTag, opts) in [("plain", Q2Fit(iterations: 8, renorm: false)),
                               ("renorm", Q2Fit(iterations: 8, renorm: true))] {
            let (packed, back) = Q2E8.quantize(values, rows: rows, k: k,
                                              soa: soa, opts: opts)
            packed.withUnsafeBytes { raw in
                try? Data(raw).write(to: URL(
                    fileURLWithPath: "\(dir)/\(tag)-\(fitTag).pack"))
            }
            back.withUnsafeBytes { raw in
                try? Data(raw).write(to: URL(
                    fileURLWithPath: "\(dir)/\(tag)-\(fitTag).back"))
            }
        }
    }
    let lut = FP8.e4m3
    lut.withUnsafeBytes { raw in
        try? Data(raw).write(to: URL(fileURLWithPath: dir + "/fp8lut.bin"))
    }
    FileHandle.standardError.write(Data(
        "[q2e8-gate] rows \(rows) k \(k) -> \(dir)\n".utf8))
    exit(0)
}

// The packed 16-bit code and the float codepoint must agree, or the file and
// the perplexity that approved it describe different models.

func runE8PGate(_ args: [String]) {
    var s: UInt64 = 0x1234_5678_9ABC_DEF0
    func next() -> Float {
        s = s &* 6364136223846793005 &+ 1442695040888963407
        // s >> 33 caps at 2^31-1, so dividing THAT by UInt32.max yields
        // [0, 0.5) and the old `* 2 - 1` gave [-1, 0) -- never positive.
        return Float(UInt32(truncatingIfNeeded: s >> 32)) / Float(UInt32.max)
            * 2 - 1
    }
    var worst: Float = 0
    var used = Set<UInt16>()
    for _ in 0..<20000 {
        var v = SIMD8<Float>()
        for i in 0..<8 {
            var u: Float = 0
            for _ in 0..<12 { u += next() }
            v[i] = u * 0.5
        }
        let direct = E8P.nearest(v)
        let code = E8P.pack(v)
        used.insert(code)
        let back = E8P.unpack(code)
        for i in 0..<8 { worst = max(worst, abs(direct[i] - back[i])) }
    }
    let table = E8P.packedTable()
    var tworst: Float = 0
    for j in 0..<256 {
        for i in 0..<8 {
            let b = Float((table[j] >> (2 * i)) & 3) + 0.5
            tworst = max(tworst, abs(b - E8P.source[j][i]))
        }
    }
    FileHandle.standardError.write(Data(
        String(format: "[e8p] pack/unpack maxdiff %.3e over 20000 vectors, "
               + "%d distinct codes  %@\n[e8p] 512B table maxdiff %.3e  %@\n",
               worst, used.count, worst == 0 ? "PASS" : "FAIL",
               tworst, tworst == 0 ? "PASS" : "FAIL").utf8))
    exit(worst == 0 && tworst == 0 ? 0 : 1)
}

import Foundation

public struct Q2Fit: Sendable {
    public var iterations: Int
    public var renorm: Bool
    public var importance: [Float]?
    public var wide: Float?

    public init(iterations: Int = 8, renorm: Bool = true,
                importance: [Float]? = nil, wide: Float? = nil) {
        self.iterations = iterations
        self.renorm = renorm
        self.importance = importance
        self.wide = wide
    }
}

public enum Q2E8 {
    public static let qk = 128
    public static let blockBytes = 34
    public static let riser: Float = 3
    static let lanes = 16
    static let vectors = qk / lanes

    // A block scale is fitted positive, so the stored fp16 sign bit is free and
    // carries which of the two codebooks the block chose.

    static func book(_ d: Float, _ opts: Q2Fit) -> Float {
        d < 0 ? (opts.wide ?? riser) : riser
    }

    static func level(_ code: UInt8, _ outer: Float) -> Float {
        let mag = code == 0 || code == 3 ? outer : 1
        return code < 2 ? -mag : mag
    }

    static func code(_ v: Float, _ d: Float, _ outer: Float) -> UInt8 {
        let far = abs(v) >= (1 + outer) / 2 * d
        return v < 0 ? (far ? 0 : 1) : (far ? 3 : 2)
    }

    static func levels(_ block: UnsafePointer<Float>, _ d: Float,
                       _ outer: Float,
                       _ out: UnsafeMutablePointer<Float>) {
        let inv = 1.0 / d
        let cut = SIMD16<Float>(repeating: (1 + outer) / 2)
        let far = SIMD16<Float>(repeating: outer)
        let near = SIMD16<Float>(repeating: 1)
        for v in 0..<vectors {
            var x = SIMD16<Float>()
            for i in 0..<lanes { x[i] = block[v * lanes + i] }
            let t = x * inv
            let size = t.replacing(with: -t, where: t .< SIMD16<Float>())
            let mag = near.replacing(with: far, where: size .>= cut)
            let l = mag.replacing(with: -mag, where: t .< SIMD16<Float>())
            for i in 0..<lanes { out[v * lanes + i] = l[i] }
        }
    }

    static func absmax(_ block: UnsafePointer<Float>) -> Float {
        var acc = SIMD16<Float>()
        for v in 0..<vectors {
            var x = SIMD16<Float>()
            for i in 0..<lanes { x[i] = abs(block[v * lanes + i]) }
            acc = acc.replacing(with: x, where: x .> acc)
        }
        return acc.max()
    }

    static func norm(_ p: UnsafePointer<Float>, _ n: Int) -> Float {
        var acc: Float = 0
        for i in 0..<n { acc += p[i] * p[i] }
        return acc.squareRoot()
    }

    static func solve(_ block: UnsafePointer<Float>, _ opts: Q2Fit,
                      _ weight: UnsafePointer<Float>?,
                      _ outer: Float) -> Float {
        var d = max(absmax(block) / outer, 1e-8)
        var level = [Float](repeating: 0, count: qk)
        level.withUnsafeMutableBufferPointer { lb in
            let l = lb.baseAddress!
            for _ in 0..<opts.iterations {
                levels(block, d, outer, l)
                var num: Float = 0
                var den: Float = 0
                if let weight {
                    for i in 0..<qk {
                        num += weight[i] * block[i] * l[i]
                        den += weight[i] * l[i] * l[i]
                    }
                } else {
                    for i in 0..<qk {
                        num += block[i] * l[i]
                        den += l[i] * l[i]
                    }
                }
                d = max(num / max(den, 1e-12), 1e-8)
            }
            if opts.renorm {
                levels(block, d, outer, l)
                var have: Float = 0
                for i in 0..<qk { have += l[i] * l[i] * d * d }
                let want = norm(block, qk)
                if have > 0 { d *= want / have.squareRoot() }
            }
        }
        return d
    }

    static func cost(_ block: UnsafePointer<Float>, _ d: Float,
                     _ outer: Float,
                     _ weight: UnsafePointer<Float>?) -> Float {
        var level = [Float](repeating: 0, count: qk)
        var acc: Float = 0
        level.withUnsafeMutableBufferPointer { lb in
            let l = lb.baseAddress!
            levels(block, d, outer, l)
            for i in 0..<qk {
                let gap = block[i] - l[i] * d
                acc += (weight?[i] ?? 1) * gap * gap
            }
        }
        return acc
    }

    static func fit(_ block: UnsafePointer<Float>, _ opts: Q2Fit,
                    _ weight: UnsafePointer<Float>?) -> Float {
        let base = solve(block, opts, weight, riser)
        var out = base
        if let wide = opts.wide {
            let alt = solve(block, opts, weight, wide)
            if cost(block, alt, wide, weight)
                < cost(block, base, riser, weight) {
                out = -alt
            }
        }
        return out
    }

    static func codes(_ block: UnsafePointer<Float>, _ d: Float,
                      _ outer: Float,
                      _ out: UnsafeMutablePointer<UInt8>) {
        for i in 0..<qk { out[i] = code(block[i], d, outer) }
    }

    static func packBytes(_ code: UnsafePointer<UInt8>,
                          _ out: UnsafeMutablePointer<UInt8>) {
        for b in 0..<(qk / 4) {
            let j = b * 4
            out[b] = code[j] | (code[j + 1] << 2)
                | (code[j + 2] << 4) | (code[j + 3] << 6)
        }
    }

    static func dequant(_ code: UnsafePointer<UInt8>, _ d: Float,
                        _ outer: Float,
                        _ out: UnsafeMutablePointer<Float>) {
        for i in 0..<qk { out[i] = level(code[i], outer) * d }
    }

    public static func quantize(_ values: [Float], rows: Int, k: Int,
                         soa: Bool, opts: Q2Fit) -> ([UInt8], [Float]) {
        let nblk = k / qk
        let rowBytes = nblk * blockBytes
        var packed = [UInt8](repeating: 0, count: rows * rowBytes)
        var back = [Float](repeating: 0, count: rows * k)
        let weights = opts.importance ?? []
        values.withUnsafeBufferPointer { vb in
        packed.withUnsafeMutableBufferPointer { pb in
        back.withUnsafeMutableBufferPointer { bb in
        weights.withUnsafeBufferPointer { wb in
                    nonisolated(unsafe) let src = vb.baseAddress!
                    nonisolated(unsafe) let dst = pb.baseAddress!
                    nonisolated(unsafe) let ref = bb.baseAddress!
                    nonisolated(unsafe) let imp =
                        weights.count == k ? wb.baseAddress : nil
                    DispatchQueue.concurrentPerform(iterations: rows) { r in
                        var code = [UInt8](repeating: 0, count: qk)
                        code.withUnsafeMutableBufferPointer { cb in
                            let c = cb.baseAddress!
                            for b in 0..<nblk {
                                let at = r * k + b * qk
                                let block = src + at
                                let w = imp.map { p in p + b * qk }
                                let d = fit(block, opts, w)
                                let outer = book(d, opts)
                                let s = abs(d)
                                codes(block, s, outer, c)
                                dequant(c, s, outer, ref + at)
                                let half = Float16(d).bitPattern
                                let base = soa
                                    ? r * rowBytes + b * 2
                                    : r * rowBytes + b * blockBytes
                                dst[base] = UInt8(half & 0xFF)
                                dst[base + 1] = UInt8(half >> 8)
                                let codeAt = soa
                                    ? r * rowBytes + nblk * 2 + b * 32
                                    : r * rowBytes + b * blockBytes + 2
                                packBytes(c, dst + codeAt)
                            }
                        }
                    }
        }}}}
        return (packed, back)
    }
}

public enum Q80 {
    public static let qk = 32
    public static let blockBytes = 34

    public static func quantize(_ values: [Float]) -> ([UInt8], [Float]) {
        let blocks = values.count / qk
        var packed = [UInt8](repeating: 0, count: blocks * blockBytes)
        var back = [Float](repeating: 0, count: values.count)
        values.withUnsafeBufferPointer { vb in
        packed.withUnsafeMutableBufferPointer { pb in
        back.withUnsafeMutableBufferPointer { bb in
                    nonisolated(unsafe) let src = vb.baseAddress!
                    nonisolated(unsafe) let dst = pb.baseAddress!
                    nonisolated(unsafe) let ref = bb.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: blocks) { b in
                        block(src + b * qk, dst + b * blockBytes,
                              ref + b * qk)
                    }
        }}}
        return (packed, back)
    }

    static func block(_ src: UnsafePointer<Float>,
                      _ dst: UnsafeMutablePointer<UInt8>,
                      _ ref: UnsafeMutablePointer<Float>) {
        var peak: Float = 0
        for i in 0..<qk { peak = max(peak, abs(src[i])) }
        let d = peak > 0 ? peak / 127 : Float(1e-8)
        let half = Float16(d).bitPattern
        dst[0] = UInt8(half & 0xFF)
        dst[1] = UInt8(half >> 8)
        for i in 0..<qk {
            let q = (src[i] / d).rounded(.toNearestOrEven)
            let code = Int8(max(-127, min(127, q)))
            dst[2 + i] = UInt8(bitPattern: code)
            ref[i] = Float(code) * d
        }
    }
}

public enum Ternary {

    public static func solve(_ block: UnsafePointer<Float>, _ n: Int,
                             _ weight: UnsafePointer<Float>?)
        -> (scale: Float, cut: Float) {
        var order = [Int](0..<n)
        order.sort { a, b in abs(block[a]) > abs(block[b]) }
        var mass = 0.0
        var mount = 0.0
        var best = -1.0
        var keep = 1
        for k in 0..<n {
            let i = order[k]
            let h = Double(weight?[i] ?? 1)
            mass += h * Double(abs(block[i]))
            mount += h
            let score = mount > 0 ? mass * mass / mount : 0
            if score > best {
                best = score
                keep = k + 1
            }
        }
        let last = abs(block[order[keep - 1]])
        var num = 0.0
        var den = 0.0
        for k in 0..<keep {
            let i = order[k]
            let h = Double(weight?[i] ?? 1)
            num += h * Double(abs(block[i]))
            den += h
        }
        let scale = den > 0 ? Float(num / den) : 1e-8
        return (max(scale, 1e-8), last)
    }

    public static func codes(_ block: UnsafePointer<Float>, _ n: Int,
                             _ cut: Float,
                             _ out: UnsafeMutablePointer<UInt8>) {
        for i in 0..<n {
            let live = abs(block[i]) >= cut
            out[i] = live ? (block[i] < 0 ? 0 : 2) : 1
        }
    }

    public static func quantize(_ values: [Float], rows: Int, k: Int,
                                importance: [Float]?) -> ([UInt8], [Float]) {
        let nblk = k / Q2E8.qk
        let rowBytes = nblk * Q2E8.blockBytes
        var packed = [UInt8](repeating: 0, count: rows * rowBytes)
        var back = [Float](repeating: 0, count: rows * k)
        let weights = importance ?? []
        values.withUnsafeBufferPointer { vb in
        packed.withUnsafeMutableBufferPointer { pb in
        back.withUnsafeMutableBufferPointer { bb in
        weights.withUnsafeBufferPointer { wb in
                    nonisolated(unsafe) let src = vb.baseAddress!
                    nonisolated(unsafe) let dst = pb.baseAddress!
                    nonisolated(unsafe) let ref = bb.baseAddress!
                    nonisolated(unsafe) let imp =
                        weights.count == k ? wb.baseAddress : nil
                    DispatchQueue.concurrentPerform(iterations: rows) { r in
                        var code = [UInt8](repeating: 0, count: Q2E8.qk)
                        code.withUnsafeMutableBufferPointer { cb in
                            let c = cb.baseAddress!
                            for b in 0..<nblk {
                                let at = r * k + b * Q2E8.qk
                                let block = src + at
                                let w = imp.map { p in p + b * Q2E8.qk }
                                let fit = solve(block, Q2E8.qk, w)
                                codes(block, Q2E8.qk, fit.cut, c)
                                let half = Float16(fit.scale).bitPattern
                                let base = r * rowBytes + b * Q2E8.blockBytes
                                dst[base] = UInt8(half & 0xFF)
                                dst[base + 1] = UInt8(half >> 8)
                                Q2E8.packBytes(c, dst + base + 2)
                                for i in 0..<Q2E8.qk {
                                    ref[at + i] =
                                        (Float(c[i]) - 1) * fit.scale
                                }
                            }
                        }
                    }
        }}}}
        return (packed, back)
    }
}

public enum FP8 {
    public static let e4m3: [Float] = {
        var table = [Float](repeating: 0, count: 256)
        for bits in 0..<256 {
            let sign: Float = (bits & 0x80) != 0 ? -1 : 1
            let exp = (bits >> 3) & 0x0F
            let man = bits & 0x07
            let value: Float
            if exp == 0x0F && man == 0x07 {
                value = Float.nan
            } else if exp == 0 {
                value = sign * Float(man) * exp2(Float(-9))
            } else {
                value = sign * (1.0 + Float(man) / 8.0) * exp2(Float(exp - 7))
            }
            table[bits] = value
        }
        return table
    }()

    static func decode(_ raw: UnsafePointer<UInt8>, _ n: Int,
                       _ out: UnsafeMutablePointer<Float>) {
        e4m3.withUnsafeBufferPointer { lut in
            let t = lut.baseAddress!
            for i in 0..<n { out[i] = t[Int(raw[i])] }
        }
    }
}

public enum BF16 {
    static func decode(_ raw: UnsafeRawPointer, _ n: Int,
                       _ out: UnsafeMutablePointer<Float>) {
        for i in 0..<n {
            let h = (raw + i * 2).loadUnaligned(as: UInt16.self)
            out[i] = Float(bitPattern: UInt32(h) << 16)
        }
    }

    static func encode(_ values: [Float]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: values.count * 2)
        out.withUnsafeMutableBufferPointer { ob in
            for i in 0..<values.count {
                let bits = values[i].bitPattern
                let round = (bits >> 16) & 1 &+ 0x7FFF
                let half = UInt16(truncatingIfNeeded: (bits &+ round) >> 16)
                ob[i * 2] = UInt8(half & 0xFF)
                ob[i * 2 + 1] = UInt8(half >> 8)
            }
        }
        return out
    }
}

public enum F16 {
    static func encode(_ values: [Float]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: values.count * 2)
        out.withUnsafeMutableBufferPointer { ob in
            for i in 0..<values.count {
                let half = Float16(values[i]).bitPattern
                ob[i * 2] = UInt8(half & 0xFF)
                ob[i * 2 + 1] = UInt8(half >> 8)
            }
        }
        return out
    }
}

public enum Q40 {
    public static let qk = 32
    public static let blockBytes = 18

    public static func quantize(_ values: [Float]) -> ([UInt8], [Float]) {
        let blocks = values.count / qk
        var packed = [UInt8](repeating: 0, count: blocks * blockBytes)
        var back = [Float](repeating: 0, count: values.count)
        values.withUnsafeBufferPointer { vb in
        packed.withUnsafeMutableBufferPointer { pb in
        back.withUnsafeMutableBufferPointer { bb in
                    nonisolated(unsafe) let src = vb.baseAddress!
                    nonisolated(unsafe) let dst = pb.baseAddress!
                    nonisolated(unsafe) let ref = bb.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: blocks) { b in
                        block(src + b * qk, dst + b * blockBytes,
                              ref + b * qk)
                    }
        }}}
        return (packed, back)
    }

    static func block(_ src: UnsafePointer<Float>,
                      _ dst: UnsafeMutablePointer<UInt8>,
                      _ ref: UnsafeMutablePointer<Float>) {
        var peak: Float = 0
        var signed: Float = 0
        for i in 0..<qk where abs(src[i]) > peak {
            peak = abs(src[i])
            signed = src[i]
        }
        let d = peak > 0 ? signed / -8.0 : Float(1e-8)
        let half = Float16(d).bitPattern
        dst[0] = UInt8(half & 0xFF)
        dst[1] = UInt8(half >> 8)
        var code = [UInt8](repeating: 0, count: qk)
        for i in 0..<qk {
            let q = (src[i] / d).rounded(.toNearestOrEven) + 8
            code[i] = UInt8(max(0, min(15, q)))
            ref[i] = (Float(code[i]) - 8.0) * d
        }
        for i in 0..<(qk / 2) {
            dst[2 + i] = code[i] | (code[i + qk / 2] << 4)
        }
    }
}

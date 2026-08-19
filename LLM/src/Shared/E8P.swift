import Foundation

// QuIP#'s padded E8 codebook. 2^16 points over 8 weights, so EXACTLY 2 bits
// per weight, stored as an 8-bit index into a 256-entry table of absolute
// values, 7 sign bits, and one bit choosing a +-1/4 offset. The 8th sign is
// forced by D8hat's parity rule, which is what makes 7 bits enough.

public enum E8P {
    public static let dim = 8
    public static let entries = 256
    static let shift: Float = 0.25

    // Raw pointers, not arrays and not a class: this is read 512 times per
    // quantized vector, where an array subscript pays a bounds check and a
    // class reference pays an ARC retain per call. Both showed up as a 3x
    // slowdown of the whole sweep.

    nonisolated(unsafe) static let entry:
        UnsafeMutablePointer<SIMD8<Float>> = {
        let out = UnsafeMutablePointer<SIMD8<Float>>.allocate(capacity: entries)
        for (j, v) in build().0.enumerated() { out[j] = v }
        return out
    }()

    nonisolated(unsafe) static let square: UnsafeMutablePointer<Float> = {
        let out = UnsafeMutablePointer<Float>.allocate(capacity: entries)
        for (j, v) in build().0.enumerated() { out[j] = (v * v).sum() }
        return out
    }()

    nonisolated(unsafe) static let evenFlip: UnsafeMutablePointer<Bool> = {
        let out = UnsafeMutablePointer<Bool>.allocate(capacity: entries)
        for (j, v) in build().1.enumerated() { out[j] = v }
        return out
    }()

    public static var source: [SIMD8<Float>] { build().0 }
    static var evenFlips: [Bool] { build().1 }

    // Cheap witness that the three tables agree and are fully populated: a
    // partly-built table still quantizes, it just quantizes to the wrong
    // codebook, and nothing downstream would say so.

    public static func checksum() -> (Int, Int, Double) {
        var acc = 0.0
        var parity = 0
        for j in 0..<entries {
            acc += Double(square[j]) * Double(j + 1)
            parity += evenFlip[j] ? 1 : 0
            acc += Double(entry[j].sum())
        }
        return (build().0.count, parity, acc)
    }

    // Absolute values of D8hat = Z^8 + 1/2: every coordinate is an odd
    // numerator over two. 227 of them have norm^2 <= 10; the table is padded
    // to 256 with the first 29 at norm^2 == 12.

    static func build() -> ([SIMD8<Float>], [Bool]) {
        var near: [[Int]] = []
        var pad: [[Int]] = []
        var digit = [Int](repeating: 0, count: dim)
        let value = [1, 3, 5, 7]
        for code in 0..<(1 << (2 * dim)) {
            var rest = code
            var norm = 0
            for i in 0..<dim {
                digit[i] = value[rest & 3]
                rest >>= 2
                norm += digit[i] * digit[i]
            }
            if norm <= 40 {
                near.append(digit)
            } else if norm == 48 && pad.count < entries - 227 {
                pad.append(digit)
            }
        }
        let all = near + pad
        var table: [SIMD8<Float>] = []
        var parity: [Bool] = []
        for n in all {
            var v = SIMD8<Float>()
            var sum = 0
            for i in 0..<dim {
                v[i] = Float(n[i]) / 2
                sum += n[i]
            }
            table.append(v)
            parity.append(sum % 4 == 0)
        }
        return (table, parity)
    }

    // Exact nearest point. For a fixed table entry the best sign pattern is
    // sign(y); when its flip count has the wrong parity the cheapest repair
    // is to flip the coordinate minimising s_i * |y_i|, which costs exactly
    // 4 * s_i * |y_i|. So the search over 2^16 points collapses to 2 * 256.

    public static func nearest(_ x: SIMD8<Float>) -> SIMD8<Float> {
        let (pick, side) = choose(x)
        return place(x, pick, side)
    }

    // The 16-bit code: 8 bits of table entry, 7 sign bits for coordinates
    // 0..6, 1 bit picking the +-1/4 offset. Coordinate 7's sign is NOT stored
    // -- it is forced so the total flip count matches the entry's parity,
    // which is what makes 7 bits enough for 8 signs.

    public static func pack(_ x: SIMD8<Float>) -> UInt16 {
        let (pick, side) = choose(x)
        // Read the signs off the FINAL codepoint, not off sign(y): `place`
        // repairs parity by flipping the cheapest coordinate, and that flip
        // can land anywhere -- including inside the seven stored bits.
        let c = place(x, pick, side)
        var bits = UInt16(pick)
        for i in 0..<(dim - 1) where c[i] - side * shift < 0 {
            bits |= UInt16(1) << (8 + i)
        }
        if side < 0 { bits |= 0x8000 }
        return bits
    }

    // The kernel's decode, in Swift, so the packed file and the reconstruction
    // the converter scored can be checked against each other.

    public static func unpack(_ code: UInt16) -> SIMD8<Float> {
        let e = Int(code & 0xFF)
        let sgn = Int((code >> 8) & 0x7F)
        let side: Float = (code & 0x8000) != 0 ? -shift : shift
        var flips = 0
        for i in 0..<(dim - 1) where (sgn >> i) & 1 == 1 { flips += 1 }
        let want = evenFlip[e] ? 0 : 1
        var out = SIMD8<Float>()
        let s = entry[e]
        for i in 0..<(dim - 1) {
            out[i] = (sgn >> i) & 1 == 1 ? -s[i] : s[i]
        }
        out[dim - 1] = (flips % 2) != want ? -s[dim - 1] : s[dim - 1]
        return out + SIMD8<Float>(repeating: side)
    }

    static func choose(_ x: SIMD8<Float>) -> (Int, Float) {
        let tab = entry
        let sqr = square
        let par = evenFlip
        var best = Float.infinity
        var pick = 0
        var side: Float = 1
        for step in 0..<2 {
            let sign: Float = step == 0 ? 1 : -1
            let y = x - SIMD8<Float>(repeating: sign * shift)
            let a = y.replacing(with: -y, where: y .< SIMD8<Float>())
            var flips = 0
            for i in 0..<dim where y[i] < 0 { flips += 1 }
            let even = flips % 2 == 0
            let energy = (a * a).sum()
            for j in 0..<entries {
                let s = tab[j]
                var cost = energy - 2 * (a * s).sum() + sqr[j]
                if even != par[j] { cost += 4 * (s * a).min() }
                if cost < best {
                    best = cost
                    pick = j
                    side = sign
                }
            }
        }
        return (pick, side)
    }

    // The 2-bit-per-coordinate form the kernel uses: every entry is eight
    // half-integers with numerator in {1,3,5,7}, so value = ((n-1)/2) + 0.5
    // and the whole 256-entry table is 512 bytes rather than 4 KB.

    public static func packedTable() -> [UInt16] {
        var out = [UInt16](repeating: 0, count: entries)
        for j in 0..<entries {
            var bits = UInt16(0)
            for i in 0..<dim {
                let b = UInt16((entry[j][i] - 0.5).rounded())
                bits |= b << (2 * i)
            }
            out[j] = bits
        }
        return out
    }

    static func place(_ x: SIMD8<Float>, _ pick: Int,
                      _ side: Float) -> SIMD8<Float> {
        let y = x - SIMD8<Float>(repeating: side * shift)
        let a = y.replacing(with: -y, where: y .< SIMD8<Float>())
        let s = entry[pick]
        var flips = 0
        for i in 0..<dim where y[i] < 0 { flips += 1 }
        var out = s.replacing(with: -s, where: y .< SIMD8<Float>())
        if (flips % 2 == 0) != evenFlip[pick] {
            let weighted = s * a
            var worst = 0
            for i in 0..<dim where weighted[i] < weighted[worst] { worst = i }
            out[worst] = -out[worst]
        }
        return out + SIMD8<Float>(repeating: side * shift)
    }
}

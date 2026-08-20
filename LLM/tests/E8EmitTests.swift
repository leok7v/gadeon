import XCTest
@testable import LLM

final class E8EmitTests: XCTestCase {

    private func fill(_ n: Int, seed: UInt64) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        var s = seed &+ 0x9E37_79B9_7F4A_7C15
        for i in 0..<n {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let bits = UInt32(truncatingIfNeeded: s >> 33)
            out[i] = Float(bits) / Float(UInt32.max) * 2 - 1
        }
        return out
    }

    private let rows = 6
    private let k = 256

    func testQuantizeEmitsCodesAndScales() {
        let got = E8.quantize(fill(rows * k, seed: 5), rows: rows, k: k,
                              opts: Q2Fit(iterations: 8, renorm: false),
                              padded: true, mu: 0)
        XCTAssertEqual(got.1.codes.count, rows * k / E8.dim)
        XCTAssertEqual(got.1.scales.count, rows * k / Q2E8.qk)
        XCTAssertTrue(got.1.scales.allSatisfy { d in d > 0 },
                      "a zero scale packs a dead block")
    }

    func testThePackerProducesAFullTensor() {
        let got = E8.quantize(fill(rows * k, seed: 6), rows: rows, k: k,
                              opts: Q2Fit(iterations: 8, renorm: false),
                              padded: true, mu: 0)
        let bytes = Q2E8Pack.e8p(got.1.codes, got.1.scales,
                                 rows: rows, k: k)
        XCTAssertEqual(bytes.count,
                       rows * (k / Q2E8.qk) * Q2E8.blockBytes)
        XCTAssertEqual(bytes.count, GGUF.rowByteCount(.q2_e8, rows * k))
    }

    func testPackedCodesDequantToTheScoredReconstruction() {
        let got = E8.quantize(fill(rows * k, seed: 7), rows: rows, k: k,
                              opts: Q2Fit(iterations: 8, renorm: false),
                              padded: true, mu: 0)
        let bytes = Q2E8Pack.e8p(got.1.codes, got.1.scales,
                                 rows: rows, k: k)
        let rowBytes = (k / Q2E8.qk) * Q2E8.blockBytes
        var back = [Float](repeating: 0, count: k)
        var worst: Float = 0
        bytes.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for r in 0..<rows {
                back.withUnsafeMutableBufferPointer { ob in
                    Q2E8Row.dequant(base + r * rowBytes, count: k,
                                    into: ob.baseAddress!)
                }
                for i in 0..<k {
                    worst = max(worst, abs(back[i] - got.0[r * k + i]))
                }
            }
        }
        XCTAssertLessThan(worst, 1e-3,
                          "emit and read disagree by \(worst)")
    }
}

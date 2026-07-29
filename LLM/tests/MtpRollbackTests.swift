import XCTest
@testable import LLM

// MtpRollback.stateAfter reconstructs a GDN layer's recurrent state after an
// accepted speculative prefix m. Hand-computed against the closed form
//   S_m = exp(gcum[m-1])*S0 + sum_{i<m} exp(gcum[m-1]-gcum[i]) * outer(k_i, delta_i)
// on a 1-head 2x2 case (the CoreML-vs-formula equality is proven separately in
// qwen35_mtp_rollback_check.py, cos 1.0).

final class MtpRollbackTests: XCTestCase {
    // one head, DK=DV=C=2
    private let s0: [Float] = [1, 0, 0, 1]              // identity [DK,DV]
    private let k: [Float] = [1, 2, 5, 6]               // k0=[1,2], k1=[5,6]
    private let delta: [Float] = [3, 4, 7, 8]           // d0=[3,4], d1=[7,8]

    private func run(m: Int, gcum: [Float]) -> [Float] {
        MtpRollback.stateAfter(m: m, H: 1, DK: 2, DV: 2, C: 2,
                               s0: s0, delta: delta, k: k, gcum: gcum)
    }

    func testFullPrefixNoDecay() {
        // m=2, g=[0,0]: S0 + outer(k0,d0) + outer(k1,d1)
        let out = run(m: 2, gcum: [0, 0])
        assertClose(out, [39, 44, 48, 57])
    }

    func testPartialPrefixNoDecay() {
        // m=1: only token 0 contributes
        let out = run(m: 1, gcum: [0, 0])
        assertClose(out, [4, 4, 6, 9])
    }

    func testDecayWeighting() {
        // m=2, gcum=[-1,-1.5]: eM=exp(-1.5), w0=exp(-0.5), w1=1
        let out = run(m: 2, gcum: [-1, -1.5])
        let eM = expf(-1.5), w0 = expf(-0.5)
        let want: [Float] = [eM * 1 + w0 * 3 + 35, w0 * 4 + 40,
                             w0 * 6 + 42, eM * 1 + w0 * 8 + 48]
        assertClose(out, want)
    }

    // m>=4 is the config the correctness eval flagged (n>=3 diverges at m=4) and is
    // NOT covered by the m=1/2 cases above -- these isolate the Swift MtpRollback
    // port at m=4/m=5. C=5, one head, DK=DV=2.
    private let k5: [Float] = [1, 2, 5, 6, 1, 1, 2, 0, 1, 1]     // 5 x [DK=2]
    private let d5: [Float] = [3, 4, 7, 8, 1, 1, 0, 3, 2, 2]     // 5 x [DV=2]
    // outer(k_t,d_t) row-major [i*DV+j]=k_t[i]*d_t[j]:
    //   o0=[3,4,6,8] o1=[35,40,42,48] o2=[1,1,1,1] o3=[0,6,0,0] o4=[2,2,2,2]

    private func run5(m: Int, gcum: [Float]) -> [Float] {
        MtpRollback.stateAfter(m: m, H: 1, DK: 2, DV: 2, C: 5,
                               s0: s0, delta: d5, k: k5, gcum: gcum)
    }

    func testM4NoDecay() {
        // S0 + o0+o1+o2+o3
        assertClose(run5(m: 4, gcum: [0, 0, 0, 0, 0]), [40, 51, 49, 58])
    }

    func testM5NoDecay() {
        // + o4
        assertClose(run5(m: 5, gcum: [0, 0, 0, 0, 0]), [42, 53, 51, 60])
    }

    func testM4Decay() {
        // gLast=gcum[3]; w_i=exp(gcum[3]-gcum[i]); tests the m=4 gcum index + exp.
        let g: [Float] = [-1, -1.5, -1.8, -2.0, -2.1]
        let eM = expf(g[3])
        let w = (0 ..< 4).map { expf(g[3] - g[$0]) }
        let o: [[Float]] = [[3, 4, 6, 8], [35, 40, 42, 48], [1, 1, 1, 1], [0, 6, 0, 0]]
        var want = [Float](repeating: 0, count: 4)
        for i in 0 ..< 4 {
            want[i] = eM * s0[i]
            for t in 0 ..< 4 { want[i] += w[t] * o[t][i] }
        }
        assertClose(run5(m: 4, gcum: g), want)
    }

    private func assertClose(_ got: [Float], _ want: [Float],
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, want.count, file: file, line: line)
        for i in 0 ..< got.count {
            XCTAssertEqual(got[i], want[i], accuracy: 1e-3, file: file, line: line)
        }
    }
}

import Foundation

// Host reconstruction of one GatedDeltaNet layer's recurrent state after an
// accepted speculative prefix of length m, from the verify trunk's per-token
// rollback ingredients (delta, k, gcum). This unrolls GDN.step's per-token
// recurrence: after processing tokens 0..m-1 of the verify chunk,
//   S_m[h] = e_m * S0[h] + sum_{i<m} exp(gcum[h,m-1]-gcum[h,i]) * (k[h,i] (x) delta[h,i])
// with e_m = exp(gcum[h,m-1]). delta_i is causal, so a single verify pass (every
// draft row real) serves every accepted m. Proven fp16-exact against the trunk's
// own rec_out for both full and partial m (qwen35_mtp_rollback_check.py, cos 1.0),
// so the base GDN state rolls back with no second trunk pass -- the whole reason
// MTP self-spec is a decode win on the weight-read-bound ANE.

enum MtpRollback {

    // S_m for one value head. gcum [C] this head's inclusive gate cumsum; k [C,DK]
    // and delta [C,DV] this head's per-token ingredients; s0/out [DK,DV] this
    // head's prior/rolled-back state (row-major, i*DV+j). m in 1...C.

    private static func headState(m: Int, DK: Int, DV: Int,
                                  gcum: ArraySlice<Float>, k: ArraySlice<Float>,
                                  delta: ArraySlice<Float>,
                                  s0: ArraySlice<Float>, out: inout [Float],
                                  outBase: Int) {
        let g = gcum.startIndex, kB = k.startIndex, dB = delta.startIndex
        let sB = s0.startIndex
        let gLast = gcum[g + m - 1]
        let eM = expf(gLast)
        for i in 0 ..< DK {
            for j in 0 ..< DV {
                out[outBase + i * DV + j] = eM * s0[sB + i * DV + j]
            }
        }
        for t in 0 ..< m {
            let w = expf(gLast - gcum[g + t])
            for i in 0 ..< DK {
                let wk = w * k[kB + t * DK + i]
                let row = outBase + i * DV
                for j in 0 ..< DV { out[row + j] += wk * delta[dB + t * DV + j] }
            }
        }
    }

    // Reconstruct all H heads' state after the accepted prefix m. Row-major fp32:
    // s0/return [H,DK,DV]; delta [H,C,DV]; k [H,C,DK]; gcum [H,C].

    static func stateAfter(m: Int, H: Int, DK: Int, DV: Int, C: Int,
                           s0: [Float], delta: [Float], k: [Float],
                           gcum: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: H * DK * DV)
        for h in 0 ..< H {
            headState(m: m, DK: DK, DV: DV,
                      gcum: gcum[h * C ..< (h + 1) * C],
                      k: k[h * C * DK ..< (h + 1) * C * DK],
                      delta: delta[h * C * DV ..< (h + 1) * C * DV],
                      s0: s0[h * DK * DV ..< (h + 1) * DK * DV],
                      out: &out, outBase: h * DK * DV)
        }
        return out
    }
}

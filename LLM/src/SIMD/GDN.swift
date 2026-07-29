// GatedDeltaNet (linear-attention) layer, one token at a time. Mirrors
// qwen35.cpp build_layer_attn_linear + delta-net-base build_delta_net_autoregressive.
// The per-token recurrence is exact, so looping it over the prompt reproduces the
// chunked/fused GPU path to fp tolerance.
import Foundation

// Mutable per-GDN-layer state: causal-conv ring (last dConv-1 pre-conv mixes) and
// the recurrent scan matrices S[key,value] per value head.
final class GDNState {
    var conv: [Float]   // [convDim * (dConv-1)], time-minor: conv[t*convDim + c]
    var rec: [Float]    // [nVHead * dState * dState], head-major; S[h][i*dState+j]
    init(_ cfg: BonsaiConfig) {
        conv = [Float](repeating: 0, count: cfg.convDim * (cfg.dConv - 1))
        rec = [Float](repeating: 0, count: cfg.nVHead * cfg.dState * cfg.dState)
    }
}

enum GDN {
    // returns the layer's contribution (before the residual add) [nEmbd]
    static func step(_ n: [Float], _ L: BonsaiLayer, _ st: GDNState, _ cfg: BonsaiConfig) -> [Float] {
        let dS = cfg.dState, nK = cfg.nKHead, nV = cfg.nVHead
        let keyDim = cfg.keyDim, valDim = cfg.valueDim, convDim = cfg.convDim
        let kc = cfg.dConv                      // 4

        var qkvMix = [Float](repeating: 0, count: convDim)
        Q2_0.matvec(L.wqkv!, x: n, out: &qkvMix)          // [convDim]
        var z = [Float](repeating: 0, count: valDim)
        Q2_0.matvec(L.wqkvGate!, x: n, out: &z)           // [valueDim]

        var betaPre = [Float](repeating: 0, count: nV)
        Q2_0.matvec(L.ssmBeta!, x: n, out: &betaPre)
        var alphaPre = [Float](repeating: 0, count: nV)
        Q2_0.matvec(L.ssmAlpha!, x: n, out: &alphaPre)

        let dt = F32T.ptr(L.ssmDt!)               // [nV]
        let aNeg = F32T.ptr(L.ssmA!)              // [nV]  (= -exp(A_log))
        var beta = [Float](repeating: 0, count: nV)
        var g = [Float](repeating: 0, count: nV)
        for h in 0..<nV {
            beta[h] = sigmoidf(betaPre[h])
            g[h] = softplusf(alphaPre[h] + dt[h]) * aNeg[h]   // <= 0
        }

        // causal depthwise conv over [conv_state(3) , qkvMix] + silu.
        // window per channel c: [s0,s1,s2,cur]; out = sum_j window[j]*w[j*? ...].
        // ggml conv1d weight [dConv, convDim] (dConv fastest): w(j,c) at c*dConv+j.
        let cw = F32T.ptr(L.ssmConv1d!)
        var convOut = [Float](repeating: 0, count: convDim)
        let cs = st.conv                          // [ (dConv-1) * convDim ], time-minor
        for c in 0..<convDim {
            var acc: Float = 0
            for j in 0..<(kc - 1) { acc += cs[j * convDim + c] * cw[c * kc + j] }
            acc += qkvMix[c] * cw[c * kc + (kc - 1)]
            convOut[c] = silu(acc)
        }
        // shift the conv ring: drop oldest, append this token's pre-conv mix
        if kc >= 2 {
            for j in 0..<(kc - 2) {
                for c in 0..<convDim { st.conv[j * convDim + c] = st.conv[(j + 1) * convDim + c] }
            }
            for c in 0..<convDim { st.conv[(kc - 2) * convDim + c] = qkvMix[c] }
        }

        // split q|k|v and per-head l2-norm on q,k
        var q = Array(convOut[0..<keyDim])                       // [dS*nK]
        var k = Array(convOut[keyDim..<(2 * keyDim)])            // [dS*nK]
        let v = Array(convOut[(2 * keyDim)..<(2 * keyDim + valDim)]) // [dS*nV]
        Kern.l2normRows(&q, d: dS, rows: nK, eps: cfg.eps)
        Kern.l2normRows(&k, d: dS, rows: nK, eps: cfg.eps)

        let qScale = 1 / Float(dS).squareRoot()
        var o = [Float](repeating: 0, count: valDim)             // [dS*nV]

        q.withUnsafeBufferPointer { qBuf in
        k.withUnsafeBufferPointer { kBuf in
        st.rec.withUnsafeMutableBufferPointer { recBuf in
            let rec = recBuf.baseAddress!
            for hv in 0..<nV {
                let hk = hv % nK                                 // ggml_repeat tile mapping
                let qh = qBuf.baseAddress! + hk * dS
                let kh = kBuf.baseAddress! + hk * dS
                let voff = hv * dS
                let S = rec + hv * dS * dS                       // S[i*dS + j]
                let gamma = expf(g[hv])
                let b = beta[hv]
                // decay + sk[j] = sum_i S[i,j]*k[i]
                var sk = [Float](repeating: 0, count: dS)
                for i in 0..<dS {
                    let ki = kh[i]
                    let row = S + i * dS
                    for j in 0..<dS {
                        let s = row[j] * gamma
                        row[j] = s
                        sk[j] += s * ki
                    }
                }
                // d[j] = beta*(v[j]-sk[j]); S[i,j] += k[i]*d[j]; o[j] = sum_i S[i,j]*q[i]*scale
                var d = [Float](repeating: 0, count: dS)
                for j in 0..<dS { d[j] = b * (v[voff + j] - sk[j]) }
                for i in 0..<dS {
                    let ki = kh[i]
                    let qi = qh[i] * qScale
                    let row = S + i * dS
                    for j in 0..<dS {
                        row[j] += ki * d[j]
                        o[voff + j] += row[j] * qi
                    }
                }
            }
        }}}

        // gated per-head RMSNorm(o, ssm_norm) * silu(z)
        let sn = F32T.ptr(L.ssmNorm!)             // [dS]
        Kern.rmsnormRows(&o, d: dS, rows: nV, w: sn, eps: cfg.eps)
        for i in 0..<valDim { o[i] *= silu(z[i]) }

        var out = [Float](repeating: 0, count: cfg.nEmbd)
        Q2_0.matvec(L.ssmOut!, x: o, out: &out)
        return out
    }
}

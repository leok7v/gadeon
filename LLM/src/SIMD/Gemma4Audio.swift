import Accelerate
import Foundation

// The gemma-4 audio tower on the CPU, mirroring Gemma4AudioModel.forward.
// A Conformer / USM encoder: a stride-2 convolutional front end that turns
// log-mel frames into tokens, then macaron blocks -- half-step feed-forward,
// attention, a light depthwise conv, half-step feed-forward.
//
// The attention is Transformer-XL RELATIVE attention over BLOCKED local
// windows, which sounds exotic and is not: queries are cut into
// non-overlapping blocks of `chunk`, keys and values into overlapping windows
// of `chunk + left - 1 + right`, and the content-position term is a rel_shift
// of q against a projected positional encoding. That encoding is SINUSOIDAL
// and CONSTANT, so `relative_k_proj(pos_emb)` is computed ONCE at load rather
// than per forward -- it is the single optimization that matters here.
//
// The reference clamps to gradient_clipping (1e10) at several points. Those
// are no-ops in f32, whose finite max is 3.4e38, so they are not reproduced.

public struct Gemma4AudioConfig {
    public let embd: Int
    public let layers: Int
    public let chunk: Int
    public let contextLeft: Int
    public let contextRight: Int
    public let logitCap: Float
    public let residualWeight: Float
    let heads: Int
    let headDim: Int
    let ff: Int
    let convKernel: Int
    let outDim: Int
    let eps: Float
    let activation: GemmaActivation

    // Queries see `chunk` positions; keys see the window around them.
    var pastHorizon: Int { contextLeft - 1 }
    var context: Int { chunk + pastHorizon + contextRight }

    init(_ g: GGUF) throws {
        func i(_ k: String) -> Int { g.int("gemma4.audio." + k)! }
        embd = i("embedding_length")
        layers = i("block_count")
        chunk = i("chunk_size")
        contextLeft = i("context_left")
        contextRight = i("context_right")
        logitCap = Float(g.double("gemma4.audio.logit_cap")!)
        residualWeight = Float(g.double("gemma4.audio.residual_weight")!)
        eps = Float(g.double("gemma4.attention.layer_norm_rms_epsilon")!)
        // per_dim_scale is [head_dim], which fixes the head count.
        headDim = g.tensor("a.blk.0.attn_per_dim_scale").dims[0]
        heads = embd / headDim
        ff = g.tensor("a.blk.0.ffw1_up.weight").dims[1]
        convKernel = g.tensor("a.blk.0.lconv_dw.weight").dims[0]
        outDim = g.tensor("a.output_proj.weight").dims[1]
        activation = try GemmaActivation.read(g, "gemma4.audio.activation")
    }
}

// How a clip is marked up in a turn and how long one may be. The processor
// writes `boa + audioToken * N + eoa`, and N follows from the clip's length
// at `msPerToken`, capped at `maxSoftTokens`.

public struct Gemma4AudioWire: Sendable {
    public let boa: Int32
    public let eoa: Int32
    public let token: Int32
    public let msPerToken: Int
    public let maxSoftTokens: Int

    public var maxSeconds: Double {
        Double(maxSoftTokens * msPerToken) / 1000
    }

    init(_ g: GGUF) throws {
        boa = Int32(try requireInt(g, "gemma4.boa_token_id",
                                   "an audio turn has no opening marker"))
        eoa = Int32(try requireInt(g, "gemma4.eoa_token_id",
                                   "an audio turn has no closing marker"))
        token = Int32(try requireInt(g, "gemma4.audio_token_id",
                                     "audio has no placeholder to expand"))
        msPerToken = try requireInt(
            g, "gemma4.audio.ms_per_token",
            "a clip's length cannot be turned into a token count")
        maxSoftTokens = try requireInt(g, "gemma4.audio.max_soft_tokens",
                                       "the clip ceiling is unknown")
    }
}

public final class Gemma4Audio {
    public let cfg: Gemma4AudioConfig
    let gguf: GGUF
    // relative_k_proj(pos_emb), [context/2 + 1][heads][headDim]. The encoding
    // is a constant, so this whole term is too.
    let relKey: [Float]
    private let srq: [String: SRQ]

    public init(_ model: Gemma4Model) throws {
        gguf = model.gguf
        cfg = try Gemma4AudioConfig(model.gguf)
        relKey = Gemma4Audio.relativeKeys(model.gguf, cfg)
        var scales: [String: SRQ] = [:]
        for il in 0..<cfg.layers {
            for tag in ["attn_q", "attn_k", "attn_v", "attn_out",
                        "ffw1_up", "ffw1_down", "ffw2_up", "ffw2_down",
                        "lconv_start", "lconv_end"] {
                let name = "a.blk.\(il).\(tag).weight"
                scales[name] = SRQ(model.gguf, name)
            }
        }
        srq = scales
    }

    // Sinusoidal encoding for positions [context/2 ... 0] descending, then the
    // per-layer projection. Layer 0's projection is used for every layer only
    // if they share it -- they do not, so this holds all layers back to back.
    private static func relativeKeys(_ g: GGUF,
                                     _ cfg: Gemma4AudioConfig) -> [Float] {
        let e = cfg.embd
        let rows = cfg.context / 2 + 1
        let half = e / 2
        var pos = [Float](repeating: 0, count: rows * e)
        for r in 0..<rows {
            let p = Float(cfg.context / 2 - r)
            for i in 0..<half {
                let inv = expf(-Float(i) * logf(10000) / Float(half - 1))
                pos[r * e + i] = sinf(p * inv)
                pos[r * e + half + i] = cosf(p * inv)
            }
        }
        var out = [Float](repeating: 0, count: cfg.layers * rows * e)
        for il in 0..<cfg.layers {
            let w = Dense.transposedFloats(
                g.tensor("a.blk.\(il).attn_rel_k.weight"), e, e)
            var slice = [Float](repeating: 0, count: rows * e)
            vDSP_mmul(pos, 1, w, 1, &slice, 1, vDSP_Length(rows),
                      vDSP_Length(e), vDSP_Length(e))
            for i in 0..<(rows * e) { out[il * rows * e + i] = slice[i] }
        }
        return out
    }

    // [frames][melBins] log-mel -> the tower's soft tokens and their
    // projection into the LM space.
    public func forward(mel: [Float], frames: Int, bins: Int,
                        tap: ((String, [Float]) -> Void)? = nil)
        -> (tower: [Float], proj: [Float], count: Int) {
        var (x, n) = subsample(mel, frames, bins)
        tap?("sub", x)
        for il in 0..<cfg.layers {
            layer(il, &x, n, il == 0 ? tap : nil)
            tap?("l\(il)", x)
        }
        let tower = outputProject(x, n)
        return (tower, project(tower, n), n)
    }

    // Run a span of layers over a supplied hidden state. Feeding it the
    // reference's own activation isolates a layer's arithmetic from the error
    // its input already carries.
    public func runLayers(_ x: [Float], _ n: Int, from: Int,
                          through: Int) -> [Float] {
        var h = x
        for il in from...through { layer(il, &h, n) }
        return h
    }

    // Two stride-2 convolutions over the [frames, bins] spectrogram, each
    // followed by a channel LayerNorm without bias and a ReLU, then the
    // channel and width axes fold together into the model width.
    func subsample(_ mel: [Float], _ frames: Int, _ bins: Int)
        -> ([Float], Int) {
        var h = mel
        var height = frames, width = bins, channels = 1
        for stage in 0..<2 {
            let w = Dense.floats(
                gguf.tensor("a.subsample.\(stage).conv.weight"))
            let g = Dense.floats(
                gguf.tensor("a.subsample.\(stage).norm.weight"))
            let outC = w.count / (channels * 9)
            let oh = (height + 1) / 2, ow = (width + 1) / 2
            var next = [Float](repeating: 0, count: oh * ow * outC)
            for y in 0..<oh {
                for x in 0..<ow {
                    for oc in 0..<outC {
                        var acc: Float = 0
                        for ic in 0..<channels {
                            for ky in 0..<3 {
                                let iy = 2 * y + ky - 1
                                for kx in 0..<3 {
                                    let ix = 2 * x + kx - 1
                                    let inside = iy >= 0 && iy < height &&
                                        ix >= 0 && ix < width
                                    if inside {
                                        let wi = ((oc * channels + ic) * 3
                                            + ky) * 3 + kx
                                        acc += w[wi] * h[(ic * height + iy)
                                            * width + ix]
                                    }
                                }
                            }
                        }
                        next[(y * ow + x) * outC + oc] = acc
                    }
                }
            }
            layerNormRows(&next, d: outC, rows: oh * ow, w: g)
            for i in 0..<next.count { next[i] = max(0, next[i]) }
            // Back to channel-major for the next convolution.
            var chw = [Float](repeating: 0, count: next.count)
            for p in 0..<(oh * ow) {
                for c in 0..<outC { chw[c * oh * ow + p] = next[p * outC + c] }
            }
            h = chw
            height = oh
            width = ow
            channels = outC
        }
        // [C][H][W] -> [H][W*C], the reference's permute-then-reshape.
        var rows = [Float](repeating: 0, count: height * width * channels)
        for y in 0..<height {
            for x in 0..<width {
                for c in 0..<channels {
                    rows[y * width * channels + x * channels + c] =
                        h[(c * height + y) * width + x]
                }
            }
        }
        let e = cfg.embd
        var out = [Float](repeating: 0, count: height * e)
        let w = Dense.transposedFloats(gguf.tensor("a.subsample.proj.weight"),
                                       width * channels, e)
        vDSP_mmul(rows, 1, w, 1, &out, 1, vDSP_Length(height),
                  vDSP_Length(e), vDSP_Length(width * channels))
        return (out, height)
    }

    private func layer(_ il: Int, _ x: inout [Float], _ n: Int,
                       _ tap: ((String, [Float]) -> Void)? = nil) {
        let e = cfg.embd
        feedForward(il, "ffw1", &x, n)
        tap?("s_ff1", x)
        let residual = x
        var h = x
        rms(&h, e, n, "a.blk.\(il).norm_pre_attn.weight")
        var a = attention(il, h, n, tap)
        tap?("s_attn", a)
        rms(&a, e, n, "a.blk.\(il).norm_post_attn.weight")
        vDSP_vadd(residual, 1, a, 1, &x, 1, vDSP_Length(n * e))
        lightConv(il, &x, n)
        tap?("s_lconv", x)
        feedForward(il, "ffw2", &x, n)
        tap?("s_ff2", x)
        rms(&x, e, n, "a.blk.\(il).norm_out.weight")
    }

    // The macaron halves: pre-norm, expand, silu, contract, post-norm, then a
    // HALF-WEIGHT residual add rather than a full one.
    private func feedForward(_ il: Int, _ tag: String, _ x: inout [Float],
                             _ n: Int) {
        let e = cfg.embd, f = cfg.ff
        let residual = x
        var h = x
        rms(&h, e, n, "a.blk.\(il).\(tag)_norm.weight")
        var up = linear("a.blk.\(il).\(tag)_up.weight", h, n, e, f)
        let act = cfg.activation
        for i in 0..<(n * f) { up[i] = act.apply(up[i]) }
        var down = linear("a.blk.\(il).\(tag)_down.weight", up, n, f, e)
        rms(&down, e, n, "a.blk.\(il).\(tag)_post_norm.weight")
        let s = cfg.residualWeight
        for i in 0..<(n * e) { x[i] = down[i] * s + residual[i] }
    }

    // GLU, a causal depthwise convolution, a norm, silu, and back.
    private func lightConv(_ il: Int, _ x: inout [Float], _ n: Int) {
        let e = cfg.embd
        let residual = x
        var h = x
        rms(&h, e, n, "a.blk.\(il).lconv_norm.weight")
        let wide = linear("a.blk.\(il).lconv_start.weight", h, n, e, 2 * e)
        var g = [Float](repeating: 0, count: n * e)
        for t in 0..<n {
            for c in 0..<e {
                let a = wide[t * 2 * e + c]
                let b = wide[t * 2 * e + e + c]
                g[t * e + c] = a / (1 + expf(-b))
            }
        }
        let k = cfg.convKernel
        let dw = Dense.floats(gguf.tensor("a.blk.\(il).lconv_dw.weight"))
        var conv = [Float](repeating: 0, count: n * e)
        for t in 0..<n {
            for c in 0..<e {
                var acc: Float = 0
                for j in 0..<k {
                    let src = t - (k - 1) + j
                    if src >= 0 { acc += g[src * e + c] * dw[c * k + j] }
                }
                conv[t * e + c] = acc
            }
        }
        rms(&conv, e, n, "a.blk.\(il).lconv_conv_norm.weight")
        let act = cfg.activation
        for i in 0..<(n * e) { conv[i] = act.apply(conv[i]) }
        let out = linear("a.blk.\(il).lconv_end.weight", conv, n, e, e)
        for i in 0..<(n * e) { x[i] = out[i] + residual[i] }
    }

    private func attention(_ il: Int, _ h: [Float], _ n: Int,
                           _ tap: ((String, [Float]) -> Void)? = nil)
        -> [Float] {
        let e = cfg.embd, d = cfg.headDim, nH = cfg.heads
        var q = linear("a.blk.\(il).attn_q.weight", h, n, e, e)
        var k = linear("a.blk.\(il).attn_k.weight", h, n, e, e)
        let v = linear("a.blk.\(il).attn_v.weight", h, n, e, e)
        // q_scale and k_scale fold a change of logarithm base into the
        // projections; per_dim_scale is learned and softplus'd.
        tap?("s_q", q)
        tap?("s_k", k)
        tap?("s_v", v)
        tap?("s_relk", relativeSlice(il))
        let scales = queryScales(il)
        let kScale = keyScale
        for t in 0..<n {
            for hd in 0..<nH {
                for c in 0..<d {
                    q[t * e + hd * d + c] *= scales[c]
                    k[t * e + hd * d + c] *= kScale
                }
            }
        }
        let ctxOut = windowed(il, q, k, v, n)
        return linear("a.blk.\(il).attn_out.weight", ctxOut, n, e, e)
    }

    // The blocked-window relative attention itself, with q/k/v already scaled.
    // Split out because it is the one piece of this tower with a contract a
    // GPU kernel can get subtly wrong, so it needs to be gateable alone.
    func windowed(_ il: Int, _ q: [Float], _ k: [Float], _ v: [Float],
                  _ n: Int) -> [Float] {
        let e = cfg.embd, d = cfg.headDim, nH = cfg.heads
        let chunk = cfg.chunk, ctx = cfg.context, past = cfg.pastHorizon
        let blocks = (n + chunk - 1) / chunk
        let padded = blocks * chunk
        let relRows = ctx / 2 + 1
        let relBase = il * relRows * e
        var out = [Float](repeating: 0, count: padded * e)
        var scores = [Float](repeating: 0, count: ctx)
        for b in 0..<blocks {
            for qi in 0..<chunk where b * chunk + qi < n {
                let qAbs = b * chunk + qi
                for hd in 0..<nH {
                    let qo = qAbs * e + hd * d
                    for o in 0..<ctx {
                        let kAbs = b * chunk + o - past
                        // rel_shift sends (query, key-offset) to encoding row
                        // o - qi, whose position value is the lag. The window
                        // holds `past` positions INCLUDING self, so the lag
                        // runs 0..past-1 and row 0 (the oldest) is never
                        // reached -- an off-by-one here is invisible in the
                        // first block, where the extra key falls before
                        // position zero and is masked anyway.
                        let rel = o - qi
                        let ok = kAbs >= 0 && kAbs < n && rel >= 1
                            && rel <= past
                        if ok {
                            var ac: Float = 0
                            let ko = kAbs * e + hd * d
                            for c in 0..<d { ac += q[qo + c] * k[ko + c] }
                            // The content-position term reads the shifted row
                            // of the constant projected encoding.
                            var bd: Float = 0
                            let ro = relBase + rel * e + hd * d
                            for c in 0..<d { bd += q[qo + c] * relKey[ro + c] }
                            let cap = cfg.logitCap
                            scores[o] = tanhf((ac + bd) / cap) * cap
                        } else {
                            scores[o] = -Float.greatestFiniteMagnitude
                        }
                    }
                    GK.softmaxInPlace(&scores, ctx)
                    let oo = qAbs * e + hd * d
                    for o in 0..<ctx where scores[o] != 0 {
                        let kAbs = b * chunk + o - past
                        if kAbs >= 0 && kAbs < n {
                            let vo = kAbs * e + hd * d
                            let s = scores[o]
                            for c in 0..<d { out[oo + c] += s * v[vo + c] }
                        }
                    }
                }
            }
        }
        return Array(out[0..<(n * e)])
    }

    func outputProject(_ x: [Float], _ n: Int) -> [Float] {
        let e = cfg.embd, o = cfg.outDim
        var out = [Float](repeating: 0, count: n * o)
        vDSP_mmul(x, 1, mat("a.output_proj.weight", e, o), 1, &out, 1,
                  vDSP_Length(n), vDSP_Length(o), vDSP_Length(e))
        let bias = Dense.floats(gguf.tensor("a.output_proj.bias"))
        for t in 0..<n {
            for c in 0..<o { out[t * o + c] += bias[c] }
        }
        return out
    }

    // embed_audio: a scale-free norm then a plain projection, matching the
    // vision side.
    func project(_ tower: [Float], _ n: Int) -> [Float] {
        let d = cfg.outDim
        let w = gguf.tensor("mm.audio.weight")
        let outDim = w.dims[1]
        var h = tower
        GK.rmsnormRowsNoWeight(&h, d: d, rows: n, eps: cfg.eps)
        var out = [Float](repeating: 0, count: n * outDim)
        vDSP_mmul(h, 1, mat("mm.audio.weight", d, outDim), 1, &out, 1,
                  vDSP_Length(n), vDSP_Length(outDim), vDSP_Length(d))
        return out
    }

    private func mat(_ name: String, _ inDim: Int,
                     _ outDim: Int) -> [Float] {
        Dense.transposedFloats(gguf.tensor(name), inDim, outDim)
    }

    // One QUANTIZED linear, clamp included. Every quantized matmul in this
    // tower goes through here rather than calling vDSP_mmul directly, so a
    // site cannot silently skip the clamp -- which is invisible in the output
    // shape and fatal to the result.
    private func linear(_ name: String, _ x: [Float], _ n: Int,
                        _ inDim: Int, _ outDim: Int) -> [Float] {
        let s = srq[name] ?? SRQ.none
        var h = x
        SRQ.apply(&h, s.input)
        var out = [Float](repeating: 0, count: n * outDim)
        vDSP_mmul(h, 1, mat(name, inDim, outDim), 1, &out, 1,
                  vDSP_Length(n), vDSP_Length(outDim), vDSP_Length(inDim))
        SRQ.apply(&out, s.output)
        return out
    }

    private func rms(_ x: inout [Float], _ d: Int, _ rows: Int,
                     _ name: String) {
        GK.rmsnormRows(&x, d: d, rows: rows,
                       w: Dense.floats(gguf.tensor(name)), eps: cfg.eps)
    }

    // LayerNorm over the last axis with a scale and no bias, which is what the
    // subsampling stages use (the rest of the model is RMSNorm).
    private func layerNormRows(_ x: inout [Float], d: Int, rows: Int,
                               w: [Float]) {
        for r in 0..<rows {
            var mean: Float = 0
            var sq: Float = 0
            x.withUnsafeBufferPointer { p in
                vDSP_meanv(p.baseAddress! + r * d, 1, &mean, vDSP_Length(d))
                vDSP_measqv(p.baseAddress! + r * d, 1, &sq, vDSP_Length(d))
            }
            let inv = 1 / (sq - mean * mean + cfg.eps).squareRoot()
            for i in 0..<d {
                x[r * d + i] = (x[r * d + i] - mean) * inv * w[i]
            }
        }
    }
}

extension Gemma4Audio {
    // The query's per-channel scale: the constant fold of the logarithm base
    // change times the learned, softplus'd per_dim_scale.
    func queryScales(_ il: Int) -> [Float] {
        let d = cfg.headDim
        let perDim = Dense.floats(
            gguf.tensor("a.blk.\(il).attn_per_dim_scale"))
        let q = powf(Float(d), -0.5) / logf(2)
        return (0..<d).map { c in q * logf(1 + expf(perDim[c])) }
    }

    var keyScale: Float { logf(1 + expf(1)) / logf(2) }

    // This layer's slice of the constant projected positional encoding.
    func relativeSlice(_ il: Int) -> [Float] {
        let rows = cfg.context / 2 + 1
        let span = rows * cfg.embd
        return Array(relKey[(il * span)..<((il + 1) * span)])
    }

    var ggufHandle: GGUF { gguf }
}

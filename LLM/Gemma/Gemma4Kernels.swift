// Elementwise / norm / rope primitives for gemma-4, kept apart from Kern so
// the ternary path's numerics cannot drift when these change.
//
// Two differences from Kern are load-bearing, not stylistic:
//   * the weight arrives as [Float], not UnsafePointer<Float>. Every gemma
//     norm is BF16 in the repacked file, so F32T.ptr would trap on it -- see
//     Findings F25.
//   * rope takes a ROTATED PAIR COUNT rather than an nRot. A full-attention
//     head is 512 wide and rotates only its first 64 pairs, the rest being an
//     identity rotation (zero inverse frequency); the pairing is still
//     (j, j + head_dim/2). Kern.ropeNeox's (i, i + nRot/2) is a different
//     permutation and would be silently wrong here (F5).
import Foundation

// The feed-forward activation a tower's config names. It is a model-shaped
// VALUE, like the rope base or the softcap, so a checkpoint that swaps two
// curves this build already implements needs no code change. A name it does
// not implement stops the load: the curves differ by a few percent per
// element, which reads as fluent output rather than as a fault.
//
// This is the FEED-FORWARD activation only. The light-conv's GLU sigmoid and
// per_dim_scale's softplus are structural to those blocks, not a choice the
// config offers, and recording them would be noise.

enum GemmaActivation: String {
    case geluTanh = "gelu_pytorch_tanh"
    case silu = "silu"

    @inline(__always)
    func apply(_ v: Float) -> Float {
        let out: Float
        switch self {
        case .geluTanh: out = GK.geluTanh(v)
        case .silu: out = v / (1 + expf(-v))
        }
        return out
    }

    // Absence THROWS rather than defaulting: picking a curve for a file that
    // named none is the silent wrong answer this key exists to prevent, and
    // a model file arrives from a download rather than from the build, so
    // the caller -- not a crash -- decides what to tell the user.
    static func read(_ g: GGUF, _ key: String) throws -> GemmaActivation {
        let name = g.string(key) ?? ""
        let found = GemmaActivation(rawValue: name)
        if found == nil {
            throw GGUFErr.parse(
                "\(key) is \(name.isEmpty ? "absent" : name); this build "
                + "implements gelu_pytorch_tanh and silu. Re-emit the file "
                + "with a repack that records it.")
        }
        return found!
    }
}

enum GK {

    // Gemma4RMSNorm: normed * weight, with NO `1 + w`. Gemma 1/2/3 used
    // (1 + w) and llama.cpp's converter bakes the +1 in; gemma 4 does not, so
    // neither the converter nor this kernel may add it (F3).
    static func rmsnorm(_ x: [Float], _ w: [Float], _ eps: Float) -> [Float] {
        let d = x.count
        var ss: Float = 0
        for v in x { ss += v * v }
        let scale = 1 / (ss / Float(d) + eps).squareRoot()
        var o = [Float](repeating: 0, count: d)
        for i in 0..<d { o[i] = x[i] * scale * w[i] }
        return o
    }

    // Per-head RMSNorm of a [d, rows] buffer (d fastest), weight [d].
    static func rmsnormRows(_ x: inout [Float], d: Int, rows: Int,
                            w: [Float], eps: Float) {
        x.withUnsafeMutableBufferPointer { b in
            let p = b.baseAddress!
            for r in 0..<rows {
                let off = r * d
                var ss: Float = 0
                for i in 0..<d { ss += p[off + i] * p[off + i] }
                let s = 1 / (ss / Float(d) + eps).squareRoot()
                for i in 0..<d { p[off + i] = p[off + i] * s * w[i] }
            }
        }
    }

    // The scale-free variant (with_scale=False): normalize, no weight. V goes
    // through this on every non-shared layer.
    static func rmsnormRowsNoWeight(_ x: inout [Float], d: Int, rows: Int,
                                    eps: Float) {
        x.withUnsafeMutableBufferPointer { b in
            let p = b.baseAddress!
            for r in 0..<rows {
                let off = r * d
                var ss: Float = 0
                for i in 0..<d { ss += p[off + i] * p[off + i] }
                let s = 1 / (ss / Float(d) + eps).squareRoot()
                for i in 0..<d { p[off + i] *= s }
            }
        }
    }

    static func softmaxInPlace(_ x: inout [Float], _ n: Int) {
        var mx = -Float.greatestFiniteMagnitude
        for i in 0..<n { mx = max(mx, x[i]) }
        var s: Float = 0
        for i in 0..<n { let e = expf(x[i] - mx); x[i] = e; s += e }
        let inv = 1 / s
        for i in 0..<n { x[i] *= inv }
    }

    // gelu_pytorch_tanh, the activation both the MLP and the per-layer gate
    // use.
    @inline(__always)
    static func geluTanh(_ v: Float) -> Float {
        let t = 0.7978845608028654 * (v + 0.044715 * v * v * v)
        return 0.5 * v * (1 + tanhf(t))
    }

    // Half-rotation rope over heads laid out [headDim, nHead]. Pair (j, j +
    // headDim/2) rotates by pos * base^(-2j/headDim) for j < `rotated`, and
    // is left untouched beyond it -- which is exactly what a zero inverse
    // frequency does. `rotated` == headDim/2 gives the plain default rope the
    // sliding layers use; 64 of 256 gives the proportional rope the full
    // layers use.
    static func rope(_ x: inout [Float], headDim: Int, nHead: Int,
                     rotated: Int, base: Float, pos: Int) {
        let half = headDim / 2
        x.withUnsafeMutableBufferPointer { b in
            let p = b.baseAddress!
            for j in 0..<rotated {
                let freq = powf(base, -2 * Float(j) / Float(headDim))
                let ang = Float(pos) * freq
                let c = cosf(ang), s = sinf(ang)
                for h in 0..<nHead {
                    let off = h * headDim + j
                    let a = p[off]
                    let bb = p[off + half]
                    p[off] = a * c - bb * s
                    p[off + half] = a * s + bb * c
                }
            }
        }
    }
}

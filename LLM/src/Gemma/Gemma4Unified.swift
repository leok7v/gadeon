import Foundation

// The multimodal path of a UNIFIED gemma-4, which has no tower behind either
// modality. An image becomes 48-pixel blocks and audio becomes 640-sample
// frames, and each goes through one small stack into the language model's
// embedding space:
//
//   image  LN -> Dense -> LN -> + factorized position -> LN -> RMS -> Linear
//   audio                                                     RMS -> Linear
//
// So this is a few matrix-vector products per token, not an encoder. It stays
// on the CPU deliberately: the whole image path is ~7 GFLOP against the text
// tower's per-token 12, and putting it on the GPU would buy milliseconds for
// a LayerNorm kernel the GPU does not otherwise need.
//
// TWO EPSILONS, and they are different numbers. The three LayerNorms are
// plain torch LayerNorms carrying its DEFAULT 1e-5 -- a value that appears in
// no config file the checkpoint ships -- while the scale-free RMSNorm in
// front of each projection uses the model's own rms_norm_eps. Using one for
// both is a silent wrong answer in every image.

public struct Gemma4UnifiedMedia {
    let embd: Int
    let normEps: Float
    let visionRmsEps: Float
    let audioRmsEps: Float
    public let samplesPerToken: Int
    public let audioMaxTokens: Int

    private let ln1w, ln1b: [Float]
    private let ln2w, ln2b: [Float]
    private let ln3w, ln3b: [Float]
    private let dense: GGUFTensor
    private let denseBias: [Float]
    private let posEmbd: GGUFTensor
    private let visionProj: GGUFTensor
    private let audioProj: GGUFTensor

    public init(_ model: Gemma4Model) throws {
        let g = model.gguf
        func v(_ name: String) -> [Float] { Dense.floats(g.tensor(name)) }
        embd = model.cfg.nEmbd
        normEps = Float(g.double("gemma4.vision.norm_epsilon") ?? 1e-5)
        visionRmsEps = Float(g.double("gemma4.vision.rms_epsilon") ?? 1e-6)
        audioRmsEps = Float(g.double("gemma4.audio.rms_epsilon") ?? 1e-6)
        samplesPerToken = try requireInt(
            g, "gemma4.audio.samples_per_token",
            "a soft token IS a frame of raw samples, so its width is the "
            + "whole audio frontend")
        audioMaxTokens = g.int("gemma4.audio.max_soft_tokens") ?? 750
        ln1w = v("v.patch_norm.1.weight")
        ln1b = v("v.patch_norm.1.bias")
        ln2w = v("v.patch_norm.2.weight")
        ln2b = v("v.patch_norm.2.bias")
        ln3w = v("v.patch_norm.3.weight")
        ln3b = v("v.patch_norm.3.bias")
        dense = g.tensor("v.patch_embd.weight")
        denseBias = v("v.patch_embd.bias")
        posEmbd = g.tensor("v.position_embd.weight")
        visionProj = g.tensor("mm.vision.weight")
        audioProj = g.tensor("mm.audio.weight")
    }

    // The pixel width one soft token is cut from, off the projection itself
    // rather than from the patch geometry, so a mismatch fails loudly here
    // instead of reading past the end of a patch.
    public var patchDim: Int { dense.dims[0] }

    // `pixels` is patch-major, `patchDim` wide per patch; `pos` is the (x, y)
    // of each in the merged grid.
    public func image(pixels: [Float],
                      pos: [(Int, Int)]) -> [[Float]] {
        let dim = patchDim
        var out = [[Float]](repeating: [], count: pos.count)
        for i in 0..<pos.count {
            let patch = Array(pixels[(i * dim)..<((i + 1) * dim)])
            out[i] = embedPatch(patch, pos[i])
        }
        return out
    }

    private func embedPatch(_ patch: [Float],
                            _ at: (x: Int, y: Int)) -> [Float] {
        var h = layerNorm(patch, ln1w, ln1b, normEps)
        var y = [Float](repeating: 0, count: embd)
        GQ.matvec(dense, x: h, out: &y)
        for i in 0..<embd { y[i] += denseBias[i] }
        y = layerNorm(y, ln2w, ln2b, normEps)
        // Factorized: one table row per coordinate VALUE per axis, the two
        // summed. The table is [posemb_size][2][embd], so the row index is
        // the coordinate times two plus the axis.
        addPosition(&y, row: at.x * 2)
        addPosition(&y, row: at.y * 2 + 1)
        y = layerNorm(y, ln3w, ln3b, normEps)
        h = y
        GK.rmsnormRowsNoWeight(&h, d: embd, rows: 1, eps: visionRmsEps)
        var out = [Float](repeating: 0, count: embd)
        GQ.matvec(visionProj, x: h, out: &out)
        return out
    }

    private func addPosition(_ y: inout [Float], row: Int) {
        var slice = [Float](repeating: 0, count: embd)
        slice.withUnsafeMutableBufferPointer { buf in
            GQ.gather(posEmbd, row: row, from: 0, count: embd,
                      into: buf.baseAddress!)
        }
        for i in 0..<embd { y[i] += slice[i] }
    }

    // Raw 16 kHz samples in [-1, 1] into soft tokens. There is no mel, no
    // window and no overlap: a token is the next `samplesPerToken` samples,
    // and a final short frame is zero-filled the way the processor pads.
    public func audio(_ pcm: [Float]) -> [[Float]] {
        let per = samplesPerToken
        let frames = min((pcm.count + per - 1) / per, audioMaxTokens)
        var out = [[Float]](repeating: [], count: frames)
        for f in 0..<frames {
            var frame = [Float](repeating: 0, count: per)
            let lo = f * per
            let hi = min(lo + per, pcm.count)
            if lo < hi {
                for i in lo..<hi { frame[i - lo] = pcm[i] }
            }
            GK.rmsnormRowsNoWeight(&frame, d: per, rows: 1, eps: audioRmsEps)
            var token = [Float](repeating: 0, count: embd)
            GQ.matvec(audioProj, x: frame, out: &token)
            out[f] = token
        }
        return out
    }

    // Mean and variance over the vector, then scale and shift. Torch's
    // LayerNorm divides the variance by N rather than N-1, which is the one
    // detail that would move every value if taken the other way.
    private func layerNorm(_ x: [Float], _ w: [Float], _ b: [Float],
                           _ eps: Float) -> [Float] {
        let n = Float(x.count)
        var sum: Float = 0
        for v in x { sum += v }
        let mean = sum / n
        var sq: Float = 0
        for v in x { sq += (v - mean) * (v - mean) }
        let inv = 1 / (sq / n + eps).squareRoot()
        var out = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count { out[i] = (x[i] - mean) * inv * w[i] + b[i] }
        return out
    }
}

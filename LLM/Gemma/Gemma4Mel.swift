import Accelerate
import Foundation

// The log-mel frontend the gemma-4 audio tower expects, mirroring
// Gemma4AudioFeatureExtractor. Pure DSP over a waveform: no weights, no model,
// so it gates on its own against the reference dump's input_features.
//
// Three details are not the obvious defaults and each shifts every frame:
// the time padding is SEMICAUSAL (frame_length/2 zeros PREPENDED, nothing
// appended) so the first frame is centred at t=0; the window is the PERIODIC
// Hann (divisor frame_length, not frame_length - 1); and the log is taken of
// mel + floor rather than of max(mel, floor), so the floor biases every bin
// rather than only the silent ones.

public struct Gemma4MelConfig: Sendable {
    public let sampleRate: Int
    public let bins: Int
    public let fftLength: Int
    public let frameLength: Int
    public let hopLength: Int
    public let melFloor: Float
    public let minHz: Float
    public let maxHz: Float

    // The processor's own values, for a file emitted before the repack began
    // recording them. The runtime path reads the file instead.
    public static let processorDefault = Gemma4MelConfig(
        sampleRate: 16000, bins: 128, fftLength: 512, frameLength: 320,
        hopLength: 160, melFloor: 0.001, minHz: 0, maxHz: 8000)

    // Every field from the file when it carries them; nil says the emit
    // predates the keys and the caller must decide.
    init?(_ g: GGUF) {
        func i(_ k: String) -> Int? { g.int("gemma4.audio.mel." + k) }
        guard let sr = i("sample_rate"), let b = i("bins"),
              let fft = i("fft_length"), let fr = i("frame_length"),
              let hop = i("hop_length"),
              let floor = g.double("gemma4.audio.mel.floor"),
              let lo = g.double("gemma4.audio.mel.min_hz"),
              let hi = g.double("gemma4.audio.mel.max_hz") else {
            return nil
        }
        sampleRate = sr
        bins = b
        fftLength = fft
        frameLength = fr
        hopLength = hop
        melFloor = Float(floor)
        minHz = Float(lo)
        maxHz = Float(hi)
    }

    public init(sampleRate: Int, bins: Int, fftLength: Int, frameLength: Int,
                hopLength: Int, melFloor: Float, minHz: Float, maxHz: Float) {
        self.sampleRate = sampleRate
        self.bins = bins
        self.fftLength = fftLength
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.melFloor = melFloor
        self.minHz = minHz
        self.maxHz = maxHz
    }
}

public final class Gemma4Mel {
    public let cfg: Gemma4MelConfig
    private let window: [Float]
    private let filters: [Float]        // [freqBins][melBins]
    private let setup: FFTSetup
    private let log2n: vDSP_Length

    public init(_ cfg: Gemma4MelConfig) {
        self.cfg = cfg
        let n = cfg.frameLength
        window = (0..<n).map { i in
            0.5 - 0.5 * cosf(2 * .pi * Float(i) / Float(n))
        }
        filters = Gemma4Mel.melFilterBank(cfg)
        log2n = vDSP_Length(log2(Double(cfg.fftLength)).rounded())
        setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
    }

    deinit { vDSP_destroy_fftsetup(setup) }

    // HTK mel triangles, unnormalized: 130 edges evenly spaced in mel between
    // minHz and maxHz, each filter rising from edge i to i+1 and falling to
    // i+2. `norm=None` upstream means the triangles peak at 1 rather than
    // being scaled by their bandwidth.
    static func melFilterBank(_ cfg: Gemma4MelConfig) -> [Float] {
        let freqBins = cfg.fftLength / 2 + 1
        let m = cfg.bins
        func toMel(_ hz: Float) -> Float {
            2595 * log10f(1 + hz / 700)
        }
        func toHz(_ mel: Float) -> Float {
            700 * (powf(10, mel / 2595) - 1)
        }
        let loMel = toMel(cfg.minHz), hiMel = toMel(cfg.maxHz)
        let edges = (0..<(m + 2)).map { i in
            toHz(loMel + (hiMel - loMel) * Float(i) / Float(m + 1))
        }
        let nyquist = Float(cfg.sampleRate) / 2
        let fftFreqs = (0..<freqBins).map { i in
            nyquist * Float(i) / Float(freqBins - 1)
        }
        var out = [Float](repeating: 0, count: freqBins * m)
        for j in 0..<m {
            let lo = edges[j], mid = edges[j + 1], hi = edges[j + 2]
            for i in 0..<freqBins {
                let f = fftFreqs[i]
                let down = (f - lo) / (mid - lo)
                let up = (hi - f) / (hi - mid)
                out[i * m + j] = max(0, min(down, up))
            }
        }
        return out
    }

    // [frames][bins] log-mel. `samples` is the raw waveform at the config's
    // sample rate.
    public func features(_ samples: [Float])
        -> (values: [Float], frames: Int) {
        let cfg = self.cfg
        let pad = cfg.frameLength / 2
        var wave = [Float](repeating: 0, count: pad + samples.count)
        for i in 0..<samples.count { wave[pad + i] = samples[i] }
        // The extractor unfolds frame_length + 1 samples and then drops the
        // last one when preemphasis is off, so the usable span is one sample
        // longer than a frame.
        let span = cfg.frameLength + 1
        let frames = wave.count >= span
            ? (wave.count - span) / cfg.hopLength + 1 : 0
        let freqBins = cfg.fftLength / 2 + 1
        var mag = [Float](repeating: 0, count: freqBins)
        var out = [Float](repeating: 0, count: frames * cfg.bins)
        for t in 0..<frames {
            spectrum(wave, t * cfg.hopLength, &mag)
            for j in 0..<cfg.bins {
                var acc: Float = 0
                for i in 0..<freqBins {
                    acc += mag[i] * filters[i * cfg.bins + j]
                }
                out[t * cfg.bins + j] = logf(acc + cfg.melFloor)
            }
        }
        return (out, frames)
    }

    // One frame's magnitude spectrum. vDSP's packed real FFT puts DC in
    // real[0] and Nyquist in imag[0], and scales everything by two.
    private func spectrum(_ wave: [Float], _ start: Int,
                          _ mag: inout [Float]) {
        let n = cfg.fftLength
        var padded = [Float](repeating: 0, count: n)
        for i in 0..<cfg.frameLength {
            padded[i] = wave[start + i] * window[i]
        }
        var real = [Float](repeating: 0, count: n / 2)
        var imag = [Float](repeating: 0, count: n / 2)
        real.withUnsafeMutableBufferPointer { rp in
            imag.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!,
                                            imagp: ip.baseAddress!)
                padded.withUnsafeBufferPointer { p in
                    p.baseAddress!.withMemoryRebound(
                        to: DSPComplex.self, capacity: n / 2) { c in
                        vDSP_ctoz(c, 2, &split, 1, vDSP_Length(n / 2))
                    }
                }
                vDSP_fft_zrip(setup, &split, 1, log2n,
                              FFTDirection(FFT_FORWARD))
            }
        }
        mag[0] = abs(real[0]) / 2
        mag[n / 2] = abs(imag[0]) / 2
        for k in 1..<(n / 2) {
            mag[k] = (real[k] * real[k] + imag[k] * imag[k]).squareRoot() / 2
        }
    }
}

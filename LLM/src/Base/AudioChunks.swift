// Cut a recording into pieces a model can take in one bite, at the pauses
// between sentences rather than at arbitrary offsets.
//
// A tower has a hard ceiling -- gemma-4 hears 30 s at a time
// (audio.max_soft_tokens at ms_per_token) -- so a longer clip has to be split
// whatever we do. Splitting on a timer cuts words in half; splitting on the
// silences the speaker already left costs nothing and keeps each piece a
// whole thought.
//
// The floor is ADAPTIVE, a low percentile of the clip's own frame energies,
// because a fixed dB threshold is wrong twice: it calls a quiet room's speech
// silence, and a noisy room's silence speech. The recording tells us what its
// own background is.
import Accelerate
import Foundation

public enum AudioChunks {
    // One piece of the recording. `hardCut` marks a piece the packer had to
    // end mid-speech because no pause fell within the ceiling -- the caller
    // should say so rather than let a sentence vanish at a seam.
    public struct Chunk: Sendable {
        public let range: Range<Int>
        public let hardCut: Bool
    }

    // 20 ms hops: long enough that one glottal pulse does not read as silence,
    // short enough to place a cut inside the gap between two sentences.
    private static let hopSeconds = 0.020
    // A pause worth cutting at. Below this, ordinary gaps between words start
    // qualifying and the pieces fragment.
    private static let pauseSeconds = 0.300
    // Speech sits this far above the measured background. A ratio, not a dB
    // offset, so it holds at any input gain.
    private static let speechOverFloor: Float = 3
    // Where the background level is read from the sorted frame energies. Low
    // enough to sit under the speech, high enough that a few clicks of true
    // digital silence do not drag it to zero.
    private static let floorPercentile = 0.20

    // The pieces, in order, covering the whole clip.
    public static func split(_ pcm: [Float], rate: Double,
                             maxSeconds: Double) -> [Chunk] {
        let hop = max(Int(rate * hopSeconds), 1)
        let cap = max(Int(rate * maxSeconds), hop)
        var out: [Chunk] = []
        if pcm.count <= cap {
            out = [Chunk(range: 0 ..< pcm.count, hardCut: false)]
        } else {
            let cuts = pauseCuts(pcm, hop: hop)
            var start = 0
            while start < pcm.count {
                let ceiling = start + cap
                let end = ceiling >= pcm.count
                    ? pcm.count
                    : (cuts.last { at in at > start && at <= ceiling }
                        ?? ceiling)
                out.append(Chunk(range: start ..< end,
                                 hardCut: end < pcm.count && !cuts.contains(end)))
                start = end
            }
        }
        return out
    }

    // Sample indexes at the MIDDLE of every pause: cutting at a pause's edge
    // clips the breath before the next sentence or leaves it dangling on the
    // previous piece, and the tower's own front end pads semicausally.
    private static func pauseCuts(_ pcm: [Float], hop: Int) -> [Int] {
        let energy = frameEnergies(pcm, hop: hop)
        let quiet = energy.sorted()
        let floor = quiet.isEmpty
            ? 0 : quiet[min(Int(Double(quiet.count) * floorPercentile),
                            quiet.count - 1)]
        let threshold = floor * speechOverFloor
        let minRun = max(Int(pauseSeconds / hopSeconds), 1)
        var out: [Int] = []
        var run = 0
        for i in 0 ... energy.count {
            let quietHere = i < energy.count && energy[i] <= threshold
            if quietHere {
                run += 1
            } else {
                if run >= minRun { out.append((i - run / 2) * hop) }
                run = 0
            }
        }
        return out
    }

    // RMS per hop. vDSP over a plain loop because a 40 s clip at 16 kHz is
    // 32000 frames and this runs on the main path of every dropped file.
    private static func frameEnergies(_ pcm: [Float], hop: Int) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(pcm.count / hop + 1)
        pcm.withUnsafeBufferPointer { p in
            var at = 0
            while at < pcm.count {
                let n = min(hop, pcm.count - at)
                var rms: Float = 0
                vDSP_rmsqv(p.baseAddress! + at, 1, &rms, vDSP_Length(n))
                out.append(rms)
                at += n
            }
        }
        return out
    }
}

// Turn a live microphone stream into the utterances worth sending, and drop
// everything else.
//
// The model charges 25 soft tokens for every second of audio whatever is in
// it, so the question a mic asks is not when to stop recording but which
// audio ever becomes tokens. This is the same energy-and-adaptive-floor
// detector AudioChunks uses on a finished file, turned around: there it marks
// pauses to CUT AT and keeps the whole recording, here it marks speech to
// KEEP and the silence between never reaches the tower.
//
// Two measurements shape it and both cost accuracy if ignored:
//
// The margin is asymmetric. Cutting 100 ms either side of a speech run eats
// word-FINAL consonants -- "planks" comes back "plank", "background" comes
// back "back", and two sentences merge -- because a trailing consonant is low
// energy and the run ends before the word does. 300 ms of post-roll restores
// all of it; 500 ms is no better and costs tokens.
//
// A span with no speech in it makes the model INVENT speech: 15.5 s of a real
// room's tone returned "This is the ground." So a doubtful utterance is
// DROPPED rather than sent -- a false negative costs a repeat, a false
// positive puts words in the user's mouth.
import Accelerate
import Foundation

// @unchecked Sendable over a lock rather than an argument: push() is called
// from the capture thread and finish() from whatever thread stopped it, and
// one uncontended lock per hardware buffer (~11 a second) is not worth
// reasoning about happens-before to avoid.
public final class SpeechGate: @unchecked Sendable {
    private let lock = NSLock()

    // One stretch of speech with its margins already applied, ready to become
    // a span.
    public struct Utterance: Sendable {
        public let samples: [Float]
        // Where it began in the stream, for a caller that wants to show it.
        public let startSeconds: Double
        // Ended because it hit the ceiling rather than because the speaker
        // stopped, so the next utterance continues the same sentence.
        public let hardCut: Bool
        public let rate: Double

        public var seconds: Double { Double(samples.count) / rate }
    }

    // Held BEFORE the trigger: a detector knows speech began only after it
    // began, so without this the first phoneme is gone and it reads as a
    // model failure.
    private static let preRoll = 0.150
    // Kept AFTER it: the measured one. See the file comment.
    private static let postRoll = 0.300
    // Silence that ends an utterance. Longer than the 300 ms pause
    // AudioChunks splits a file at, because a stop consonant inside a
    // sentence goes quiet for longer than the gap between two words and
    // would otherwise cut the sentence in half.
    private static let hangover = 0.700
    // Shorter than a syllable: a click, a key, a chair. Dropped rather than
    // sent, since a span with no speech in it is answered with invented
    // speech.
    private static let minSpeech = 0.250
    private static let hop = 0.020
    private static let speechOverFloor: Float = 3
    // The background is read from a rolling window of recent QUIET frames.
    // Only quiet ones: a long utterance would otherwise drag the floor up
    // behind itself and the gate would swallow its own tail.
    private static let floorWindow = 5.0
    private static let floorPercentile = 0.20
    // A ratio is scale-free, which is the point of it and also its one blind
    // spot: three times nothing is nothing, so a background measured as zero
    // makes every later frame speech, the run never ends, and the session is
    // one ceiling-length hard cut with the whole clock inside it.
    //
    // This bounds what one bad measurement can do -- but it must stay far
    // BELOW any real room, because it is an absolute in an otherwise relative
    // measure and it stops scaling the moment a device runs quieter than it.
    // Measured on an iPad, whose input is far softer than a phone held close:
    // a whole spoken sentence peaked at 2.9x a bar this clamp was holding up,
    // so only the loudest vowel cleared it and 0.8s of the sentence survived.
    // `notAudio` below is the actual guard against a dead input; this only
    // has to be too loud to be silence, not as loud as a room.
    private static let floorAtLeast: Float = 5e-5
    // No room is this quiet: a frame under it is an input that is not running
    // yet -- a microphone hands over exact zeros while the hardware spins up
    // -- and taking it for the background is how the floor reaches zero.
    private static let notAudio: Float = 1e-5

    private let rate: Double
    private let ceiling: Double
    private let hopSize: Int
    private var pending: [Float] = []        // samples not yet a whole hop
    private var preRollRing: [Float] = []
    private var current: [Float] = []
    private var quiet: [Float] = []          // recent non-speech energies
    private var speaking = false
    private var recent: Float = 0
    private var openPreRoll = 0
    private var loudestFrame: Float = 0
    private var quietRun = 0
    private var consumed = 0                 // samples the stream has produced
    private var startedAt = 0

    // `maxSeconds` is the tower's own ceiling, so an utterance that runs past
    // it is cut and continued rather than growing until it cannot be encoded.
    public init(rate: Double, maxSeconds: Double) {
        self.rate = rate
        self.ceiling = maxSeconds
        self.hopSize = max(Int(rate * SpeechGate.hop), 1)
    }

    // Feed captured samples; take back whatever utterances completed.
    public func push(_ block: [Float]) -> [Utterance] {
        lock.lock()
        defer { lock.unlock() }
        pending.append(contentsOf: block)
        var out: [Utterance] = []
        while pending.count >= hopSize {
            let frame = Array(pending[0 ..< hopSize])
            pending.removeFirst(hopSize)
            if let done = step(frame) { out.append(done) }
            consumed += hopSize
        }
        return out
    }

    // End of capture: the speech still in hand is an utterance too.
    public func finish() -> [Utterance] {
        lock.lock()
        defer { lock.unlock() }
        var out: [Utterance] = []
        if !pending.isEmpty {
            current.append(contentsOf: speaking ? pending : [])
            pending = []
        }
        if let done = close(hard: false) { out.append(done) }
        return out
    }

    // One hop. Returns an utterance when this frame ended one.
    private func step(_ frame: [Float]) -> Utterance? {
        let e = energy(frame)
        let bar = threshold()
        // Kept for the UI: how far this frame stands over the bar it has to
        // clear, which is the only honest measure of "being heard" -- a raw
        // level would move with the room rather than with the voice.
        recent = bar > 0 ? e / bar : 0
        if e > loudestFrame { loudestFrame = e }
        let loud = e > bar
        var out: Utterance? = nil
        if loud {
            if !speaking {
                speaking = true
                startedAt = consumed - preRollRing.count
                // Kept so the length test can discount it: the pre-roll is
                // margin, not speech, and counting it would let a click that
                // arrived after a pause pass as a syllable.
                openPreRoll = preRollRing.count
                current = preRollRing
                preRollRing = []
            }
            current.append(contentsOf: frame)
            quietRun = 0
            if Double(current.count) / rate >= ceiling {
                out = close(hard: true)
            }
        } else if speaking {
            current.append(contentsOf: frame)
            quietRun += 1
            if Double(quietRun) * SpeechGate.hop >= SpeechGate.hangover {
                out = close(hard: false)
            }
        } else {
            note(energy(frame))
            preRollRing.append(contentsOf: frame)
            let cap = Int(rate * SpeechGate.preRoll)
            if preRollRing.count > cap {
                preRollRing.removeFirst(preRollRing.count - cap)
            }
        }
        return out
    }

    // Finish the utterance in hand: trim the hangover back to the post-roll
    // margin, and refuse anything too short to be a syllable.
    private func close(hard: Bool) -> Utterance? {
        let keep = Int(rate * SpeechGate.postRoll)
        let tail = min(Int(Double(quietRun) * SpeechGate.hop * rate),
                       current.count)
        let cut = current.count - max(tail - keep, 0)
        // What was actually SPOKEN: the run minus its pre-roll margin and
        // minus the hangover that ended it. Measuring it off `cut` instead
        // subtracts the hangover a second time and never subtracts the
        // pre-roll, which understates a real utterance by about half and
        // drops short ones ("yes", "hello") that clear the bar honestly.
        let voiced = Double(max(0, current.count - openPreRoll - tail)) / rate
        var out: Utterance? = nil
        if voiced >= SpeechGate.minSpeech {
            out = Utterance(samples: Array(current[0 ..< cut]),
                            startSeconds: Double(startedAt) / rate,
                            hardCut: hard, rate: rate)
        }
        // A hard cut continues the SAME speech, so the gate stays open and
        // the next utterance picks up where this one stopped -- no pre-roll
        // to prepend, no silence between them.
        current = []
        preRollRing = []
        speaking = hard
        openPreRoll = 0
        startedAt = hard ? consumed : startedAt
        quietRun = 0
        return out
    }

    // Frame energy AFTER pre-emphasis, which is what makes this a speech
    // detector rather than a loudness one: a room's rumble, a fan and a
    // desk thump live under ~200 Hz, and the first difference kills them
    // while leaving the formants that carry speech.
    private func energy(_ frame: [Float]) -> Float {
        var lifted = frame
        for i in stride(from: frame.count - 1, to: 0, by: -1) {
            lifted[i] = frame[i] - 0.97 * frame[i - 1]
        }
        var rms: Float = 0
        vDSP_rmsqv(lifted, 1, &rms, vDSP_Length(lifted.count))
        return rms
    }

    private func note(_ e: Float) {
        if e > SpeechGate.notAudio {
            quiet.append(e)
            let cap = Int(SpeechGate.floorWindow / SpeechGate.hop)
            if quiet.count > cap { quiet.removeFirst(quiet.count - cap) }
        }
    }

    // Speech sits a fixed RATIO above the measured background, not a fixed
    // dB, so the gate holds at any input gain. Until enough quiet frames have
    // been seen the bar is deliberately impossible, which costs the first
    // fraction of a second of a session and never invents an utterance.
    private func threshold() -> Float {
        let ranked = quiet.sorted()
        var out = Float.greatestFiniteMagnitude
        if ranked.count >= 8 {
            let at = min(Int(Double(ranked.count) * SpeechGate.floorPercentile),
                         ranked.count - 1)
            out = max(ranked[at], SpeechGate.floorAtLeast)
                * SpeechGate.speechOverFloor
        }
        return out
    }

    // The bar a frame has to clear to count as speech right now. A session
    // that heard the wrong thing is explained by this number and by nothing
    // else the caller can see -- an utterance count says a gate latched, not
    // what it was listening for.
    public var speechThreshold: Float {
        lock.lock()
        defer { lock.unlock() }
        return threshold()
    }

    // A speech run is open right now: this is what the gate would keep, not
    // merely what the microphone is picking up, so a fan or a passing car
    // does not read as being heard.
    public var hearing: Bool {
        lock.lock()
        defer { lock.unlock() }
        return speaking
    }

    // The loudest FRAME energy this session, in the same units as the bar.
    // A cumulative sample rms cannot answer whether speech clears the
    // threshold -- it is dominated by the silence around it -- and this is
    // the one number that compares directly.
    public var loudestFrameEnergy: Float {
        lock.lock()
        defer { lock.unlock() }
        return loudestFrame
    }

    // 0 at the bar, 1 well over it. Compressed rather than linear because
    // speech runs an order of magnitude above the floor and a linear scale
    // spends its whole range on the loudest syllable.
    public var level: Float {
        lock.lock()
        defer { lock.unlock() }
        let over = max(0, recent - 1)
        return min(1, over / 4)
    }
}

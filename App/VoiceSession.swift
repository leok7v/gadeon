import Foundation
import LLM
import Observation

// The speaking half of a voice conversation: it takes the answer as it
// streams, decides what of it is worth saying, and says it.
//
// Text runs AHEAD of speech by design. Generation is several times faster
// than the voice, so the transcript settles long before the sound does; the
// queue absorbs that and the UI shows it. Throttling generation to match was
// considered and rejected -- it buys only buffered text, which is bytes.

@MainActor @Observable final class VoiceSession {

    // How much of a reply is read aloud. ONE control rather than two
    // switches: "read the reasoning" means nothing while nothing is read at
    // all, and a pair of toggles offers a fourth state that does not exist.
    enum Mode: String, CaseIterable, Identifiable, Sendable {
        case off, replies, everything
        var id: String { rawValue }

        var label: String {
            switch self {
            case .off: return "Off"
            case .replies: return "Replies"
            case .everything: return "Everything"
            }
        }

        var detail: String {
            switch self {
            case .off:
                return "Replies are shown, never spoken."
            case .replies:
                return "The answer is read aloud. Thinking stays silent, so "
                    + "a long think is a pause."
            case .everything:
                return "The reasoning is read as it arrives, then the "
                    + "answer, so a long think is something to listen to "
                    + "instead of silence over a finished-looking screen."
            }
        }
    }

    var mode: Mode = VoiceSession.startMode() {
        didSet {
            UserDefaults.standard.set(mode.rawValue, forKey: "voiceMode")
            if mode != .off {
                UserDefaults.standard.set(mode.rawValue, forKey: "voiceLast")
            }
            if mode == .off { stopSpeaking() }
        }
    }

    // Carried over from the two switches this replaced, so an install that
    // already chose them keeps its choice.
    private static func startMode() -> Mode {
        let d = UserDefaults.standard
        var result = Mode(rawValue: d.string(forKey: "voiceMode") ?? "")
        if result == nil {
            let spoken = d.bool(forKey: "speakReplies")
            let reasoning = d.object(forKey: "speakReasoning") as? Bool ?? true
            result = spoken ? (reasoning ? .everything : .replies) : .off
        }
        return result!
    }

    // The composer's speaker button is a two-state view of the same setting:
    // turning it back on restores the depth the user last chose rather than
    // silently demoting them to replies-only.
    var enabled: Bool {
        get { mode != .off }
        set {
            if newValue {
                let last = UserDefaults.standard.string(forKey: "voiceLast")
                mode = Mode(rawValue: last ?? "") ?? .everything
            } else {
                mode = .off
            }
        }
    }

    var speakReasoning: Bool { mode == .everything }

    var voiceName: String = UserDefaults.standard
        .string(forKey: "voiceName") ?? Speech.defaultVoice.name {
        didSet { UserDefaults.standard.set(voiceName, forKey: "voiceName") }
    }

    var speed: Double = {
        let v = UserDefaults.standard.double(forKey: "voiceSpeed")
        return v > 0 ? v : 1.0
    }() {
        didSet { UserDefaults.standard.set(speed, forKey: "voiceSpeed") }
    }

    // Sound is still playing (or still being made). Distinct from the chat's
    // `busy`: the answer can be complete while the voice is halfway through.
    private(set) var speaking = false
    private(set) var paused = false

    // A spoken turn is under way and the app is not yet ready for the next
    // one. NOT the same as `speaking`, and the difference is what the
    // transport controls key off: sound stops between the wait cue and the
    // first answer sentence, and again between sentences, so a control row
    // driven by `speaking` flickers away and back several times a turn.
    // This spans the whole turn, from its start until the queue has drained.
    private(set) var engaged = false
    @ObservationIgnored private var turnRunning = false
    // Silenced for the REST OF THIS TURN. Emptying the queue is not enough on
    // its own: the turn goes on producing sentences, each one starts the
    // voice again, and the silence the listener asked for lasts until the
    // next sentence arrives.
    @ObservationIgnored private var muted = false

    @ObservationIgnored private let player = VoicePlayer()
    @ObservationIgnored private var answer = SpeakableText()
    @ObservationIgnored private var reasoning = SpeakableText()
    // A wait cue is spoken at most once per wait, and only when the wait is
    // long enough to read as silence rather than as a pause.
    @ObservationIgnored private var cueTask: Task<Void, Never>?
    @ObservationIgnored private var cued = false
    private static let cueDelay = 3.5

    init() {
        player?.onActivity = { [weak self] active in
            Task { @MainActor in
                self?.speaking = active
                if !active {
                    self?.player?.idle()
                    self?.settle()
                }
            }
        }
    }

    // The turn is over once generation has finished AND the queue has run
    // dry; either alone still owes the listener sound.
    private func settle() {
        if !turnRunning && !(player?.isActive ?? false) { engaged = false }
    }

    var available: Bool { player != nil }

    // ---- turn lifecycle -------------------------------------------------

    // A new turn starts: the previous turn's sound is stale the moment this
    // one begins, so it goes rather than queueing behind.
    func beginTurn() {
        stopSpeaking()
        answer = SpeakableText()
        reasoning = SpeakableText()
        muted = false
        if enabled {
            engaged = true
            turnRunning = true
        }
        armCue()
    }

    func answerArrived(_ piece: String) {
        disarmCue()
        if enabled {
            for segment in answer.push(piece) { say(segment) }
        }
    }

    // Any reasoning at all means the thinking wait is over, whether or not
    // it is being read aloud.
    func reasoningArrived(_ piece: String) {
        disarmCue()
        if enabled && speakReasoning {
            for segment in reasoning.push(piece) { say(segment) }
        }
    }

    func endTurn() {
        disarmCue()
        if enabled {
            for segment in reasoning.finish() { say(segment) }
            for segment in answer.finish() { say(segment) }
        }
        answer = SpeakableText()
        reasoning = SpeakableText()
        turnRunning = false
        settle()
    }

    // ---- the wait cue ---------------------------------------------------

    // With the reasoning unspoken, a long think is silence over a screen that
    // looks finished. One plain sentence covers it; a rotating ticker would
    // not, because audio cannot be ignored the way a glanceable quip can.
    private func armCue() {
        cueTask?.cancel()
        cued = false
        cueTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(VoiceSession.cueDelay))
            if !Task.isCancelled && enabled && !cued {
                cued = true
                say(SpokenCue.phrase(.thinking))
            }
        }
    }

    private func disarmCue() {
        cueTask?.cancel()
        cueTask = nil
    }

    // A tool round is the other invisible wait, and the longer of the two.
    //
    // ARMED, not spoken. A local tool (the calculator, the clock) answers in
    // microseconds, and announcing one costs more than it covers: the cue
    // queues AHEAD of the reply and delays the thing the listener is waiting
    // for. A delay decides that per round instead of a hardcoded list of
    // which tools are slow, and it stays right when a search is cached or an
    // expression is enormous.
    private static let toolCueDelay = 1.2

    func toolStarted(_ name: String) {
        cueTask?.cancel()
        cueTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(VoiceSession.toolCueDelay))
            if !Task.isCancelled && enabled {
                say(SpokenCue.forTool(name))
            }
        }
    }

    func toolFinished() {
        disarmCue()
    }

    // ---- controls -------------------------------------------------------

    // Barge-in: silence, and nothing else. The turn keeps generating, which
    // is what makes interrupting cheap -- the answer still completes and can
    // be read.
    func stopSpeaking() {
        disarmCue()
        player?.stop()
        paused = false
        speaking = false
        turnRunning = false
        engaged = false
        muted = true
    }

    private func say(_ text: String) {
        if !muted {
            player?.enqueue(text, voice: voiceName, speed: Float(speed))
        }
    }

    func pause() {
        player?.pause()
        paused = true
    }

    func resume() {
        player?.resume()
        paused = false
    }

    // Settings: hear a voice before choosing it. Straight to the player, so
    // a preview works while a turn is silenced.
    func preview(_ voice: SpeechVoice) {
        player?.stop()
        player?.enqueue(SpokenCue.preview, voice: voice.name,
                        speed: Float(speed))
    }
}

// The spoken counterparts of the on-screen quips. Deliberately a SEPARATE,
// short, plain set: the transcript's phrases are written to be glanced at
// (and carry emoji, which a voice cannot say), and repetition that is
// charming in text is grating in sound.

enum SpokenCue {

    enum Kind { case thinking, consulting }

    static let preview = "This is how I will read your replies."

    private static let thinking = [
        "Let me think about that.",
        "One moment.",
        "Thinking that through.",
    ]

    private static let consulting = [
        "Looking that up.",
        "Let me check a source.",
        "Just a moment while I look.",
    ]

    // Naming the tool is the whole value: "let me search the web" tells the
    // listener what the pause is FOR, which a generic line cannot. An
    // unrecognised name falls back rather than reading a raw identifier.
    private static let byTool: [String: String] = [
        "calculator": "Let me work that out.",
        "get_current_time": "Let me check the time.",
        "web_search": "Let me search the web.",
        "fetch_url": "Let me open that page.",
        "get_news": "Let me check the headlines.",
        "get_weather": "Let me check the forecast.",
        "wikipedia_query": "Let me see what Wikipedia says.",
    ]

    @MainActor static func forTool(_ name: String) -> String {
        byTool[name] ?? phrase(.consulting)
    }

    // Rotated rather than random, so a repeat is as far away as the set
    // allows and a short session never says the same line twice.
    nonisolated(unsafe) private static var turn = 0

    @MainActor static func phrase(_ kind: Kind) -> String {
        let list = kind == .thinking ? thinking : consulting
        let pick = list[turn % list.count]
        turn += 1
        return pick
    }
}

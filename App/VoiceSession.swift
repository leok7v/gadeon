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
            let reasoning = d.object(forKey: "speakReasoning") as? Bool ?? false
            result = spoken ? (reasoning ? .everything : .replies) : .off
        }
        return result!
    }

    // The composer's speaker button is a two-state view of the same setting:
    // turning it back on restores the depth the user last chose rather than
    // silently demoting them to replies-only.
    //
    // With no choice on record it lands on `replies`, not `everything`:
    // reading the reasoning aloud is useful for watching the model work and
    // wearing for using it, since the thinking is several times the length of
    // the answer and the answer is what the listener is waiting for.
    var enabled: Bool {
        get { mode != .off }
        set {
            if newValue {
                let last = UserDefaults.standard.string(forKey: "voiceLast")
                mode = Mode(rawValue: last ?? "") ?? .replies
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
    // The ANSWER sentence being read aloud right now, so the transcript can
    // show where the voice has got to. Only answer text: the reasoning and
    // the wait cues are spoken too, and neither is anywhere on screen to
    // point at.
    private(set) var spokenText: String?

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

    // The hop to the main actor is asynchronous, so the value a callback
    // CARRIES is only what was true when it was raised: a notify from a
    // synthesis already in flight can land after a stop and re-assert
    // speaking, which nothing afterwards clears. Reading the player instead
    // makes the callback idempotent -- it always settles on what is true when
    // it runs, whatever order the callbacks arrive in.

    // Every ANSWER segment handed to the player, by its tag. A cue or a piece
    // of reasoning gets no entry, so the transcript is never asked to point at
    // words it does not contain.
    @ObservationIgnored private var spoken: [Int: String] = [:]
    @ObservationIgnored private var nextTag = 0

    init() {
        player?.onActivity = { [weak self] _ in
            Task { @MainActor in
                let active = self?.player?.isActive ?? false
                self?.speaking = active
                // Sound playing IS a spoken exchange, however it started.
                // `beginTurn` can only raise this for a turn that began with
                // the voice already on, so a listener who switches it on
                // mid-reply would otherwise hear the answer with no transport
                // to pause or stop it. Only ever RAISED here: lowering stays
                // with `settle`, which waits for the turn to end and the
                // queue to drain, so the row cannot flicker between
                // sentences.
                if active { self?.engaged = true }
                if !active {
                    self?.player?.idle()
                    self?.settle()
                }
            }
        }
        player?.onSpeaking = { [weak self] tag in
            Task { @MainActor in
                self?.spokenText = tag.flatMap { t in self?.spoken[t] }
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
    func beginTurn(cue: SpokenCue.Kind = .thinking) {
        cueKind = cue
        stopSpeaking()
        answer = SpeakableText()
        reasoning = SpeakableText()
        // The ledger belongs to ONE turn: the previous turn's sentences are
        // not in the transcript this one will point at, and keeping them
        // would grow unbounded across a long conversation.
        spoken = [:]
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
            for segment in answer.push(piece) { say(segment, shown: true) }
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
    // Set by the turn that armed it, so the wait cue names what is actually
    // being worked on rather than always claiming to think.
    @ObservationIgnored private var cueKind: SpokenCue.Kind = .thinking

    private func armCue() {
        cueTask?.cancel()
        cued = false
        cueTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(VoiceSession.cueDelay))
            if !Task.isCancelled && enabled && !cued {
                cued = true
                say(SpokenCue.phrase(cueKind))
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
        spokenText = nil
        turnRunning = false
        engaged = false
        muted = true
    }

    // `shown` marks a segment that came from the ANSWER, so the transcript can
    // be asked to point at it. The recorded form is the segment BEFORE
    // `terminated` adds its full stop: the transcript holds the model's own
    // words, and a stop this layer invented is not among them.
    private func say(_ text: String, shown: Bool = false) {
        if !muted {
            let tag = nextTag
            nextTag += 1
            if shown {
                spoken[tag] = text
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            player?.enqueue(terminated(text), tag: tag, voice: voiceName,
                            speed: Float(speed))
        }
    }

    // The synthesizer renders the last phoneme's release off the TERMINAL
    // PUNCTUATION, so text without any comes back cut short -- and the loss is
    // a roughly constant tail, which makes it fatal exactly where it is least
    // expected. Measured through `gadeon-cli --tts`: "Hi" 0.10s against "Hi."
    // 0.75s, "Hello" 0.25s against 0.57s, while "one two three four five"
    // loses 11% and a long sentence nothing audible. A one-word reply is
    // therefore swallowed whole.
    //
    // Only . ! ? work: a comma, colon, semicolon or ellipsis all measure the
    // same 0.23s as the bare word. `finish()` flushes the end of a turn
    // "complete or not", so an unterminated fragment reaching here is ordinary
    // rather than exceptional.
    //
    // Fixed HERE and not in Speech.synthesize because that seam is gated
    // byte-for-byte against the reference implementation's WAV; changing what
    // it is handed would move the oracle rather than fix the caller.

    private func terminated(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let closed = trimmed.last.map { last in
            last == "." || last == "!" || last == "?"
        } ?? true
        return closed ? trimmed : trimmed + "."
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
        player?.enqueue(SpokenCue.preview, tag: -1, voice: voice.name,
                        speed: Float(speed))
    }
}

// The spoken counterparts of the on-screen quips. Deliberately a SEPARATE,
// short, plain set: the transcript's phrases are written to be glanced at
// (and carry emoji, which a voice cannot say), and repetition that is
// charming in text is grating in sound.

enum SpokenCue {

    // What the listener is waiting THROUGH. Naming it is the whole value: a
    // pause that says "I am looking at your video frame by frame" is a pause
    // the listener understands, where a generic line leaves them wondering
    // whether anything is happening at all.
    enum Kind { case thinking, consulting, looking, watching, reading }

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

    private static let looking = [
        "I am studying your picture with a magnifying glass.",
        "Let me take a good look at that.",
        "I need a moment to take this in.",
    ]

    private static let watching = [
        "I am looking at your video frame by frame.",
        "Watching this through.",
        "Give me a moment with the footage.",
    ]

    private static let reading = [
        "I need a moment to read what you attached.",
        "Let me look at what is inside that.",
        "Reading through it now.",
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
        let list: [String]
        switch kind {
        case .thinking: list = thinking
        case .consulting: list = consulting
        case .looking: list = looking
        case .watching: list = watching
        case .reading: list = reading
        }
        let pick = list[turn % list.count]
        turn += 1
        return pick
    }
}

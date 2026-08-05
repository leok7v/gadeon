import AVFoundation
import Foundation
import LLM

// Synthesis + playback for the speaking session: text segments in, sound out,
// in order, with a stop that takes effect immediately.
//
// Synthesis runs on its own serial queue -- Speech.synthesize is synchronous
// and holds an arena for the length of one call, so it must never be entered
// twice at once and must never block the main thread (a sentence is tens to
// hundreds of milliseconds of arithmetic).
//
// Playback is a player node fed scheduled buffers rather than a file player:
// consecutive sentences then abut with no gap and no temporary file, and stop
// is a node call rather than a wait.

final class VoicePlayer: @unchecked Sendable {

    // How many sentences may be synthesized ahead of the one being spoken.
    // Enough that playback never waits on the engine, few enough that a
    // barge-in throws away almost nothing.
    private static let readAhead = 2

    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private let synthQueue = DispatchQueue(label: "gadeon.tts.synth",
                                           qos: .userInitiated)
    private let lock = NSLock()
    // The transport calls a LISTENER makes -- stop, pause, resume -- run here
    // rather than on the caller. AVAudioPlayerNode.stop() blocks until the
    // render thread has drained, so calling it from the main actor makes a
    // user-interactive thread wait on an audio thread at default priority:
    // the inversion the Thread Performance Checker flags as a hang risk, on
    // the very path the Stop button takes.
    //
    // Scheduling does NOT come through here. It already runs on synthQueue,
    // so it never inverted anything, and it is ordered against a stop by the
    // epoch test both take under `lock`: a stop that wins the lock leaves the
    // schedule stale and it never reaches the node at all.
    //
    // Default QoS, deliberately, and NOT a higher class: `stop()` waits on an
    // audio helper thread that runs at default, so any queue above that only
    // moves the inversion one level down rather than removing it. Nothing is
    // drawing while this runs, and the decision it carries out was already
    // taken under the lock, so the work genuinely is not user-initiated.
    private let nodeQueue = DispatchQueue(label: "gadeon.tts.node")
    private var speech: Speech?
    // Each queued segment carries a TAG the player never interprets. The
    // caller uses it to say which piece of its own text this sound is, which
    // is the only way anything above can know where the voice has got to --
    // and keeping it opaque is what stops transcript geometry from reaching
    // an audio queue.
    private var pending: [(text: String, tag: Int)] = []
    // The tags of buffers handed to the node, oldest first: the head is what
    // is audible now.
    private var playing: [Int] = []
    private var inFlight = 0
    private var scheduled = 0
    // Bumped by every stop, so a synthesis that was already running when the
    // user interrupted cannot schedule its result into the new silence.
    private var epoch = 0
    private var running = false
    // A sentence finishing synthesis must not restart playback that the
    // listener paused -- schedule() calls play() on every buffer.
    private var isPaused = false

    // Told whenever the busy/idle state changes, so the UI can show that
    // sound is still trailing the finished transcript.
    var onActivity: (@Sendable (Bool) -> Void)?
    // Told the tag of whatever is audible now, or nil for silence.
    var onSpeaking: (@Sendable (Int?) -> Void)?

    init?() {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                sampleRate: Double(Speech.sampleRate),
                                channels: 1, interleaved: false)
        if let fmt {
            format = fmt
            engine.attach(node)
            engine.connect(node, to: engine.mainMixerNode, format: fmt)
        } else {
            return nil
        }
    }

    // The engine is built on first use, off the main thread: loading the
    // weights is tens of milliseconds and there is no reason to pay it at
    // launch for a user who never turns speech on.
    private func ready() -> Speech? {
        lock.lock()
        let have = speech
        lock.unlock()
        var result = have
        if result == nil {
            let built = Speech()
            lock.lock()
            if speech == nil { speech = built }
            result = speech
            lock.unlock()
        }
        return result
    }

    var isActive: Bool {
        lock.lock()
        defer { lock.unlock() }
        return inFlight > 0 || scheduled > 0 || !pending.isEmpty
    }

    // Every state change gets a line. Nothing in this path used to report at
    // all, so a voice that went quiet left no record of WHERE it stopped --
    // synthesis, scheduling, or the node -- and each looks the same from
    // outside. A handful of lines per sentence, against the decode path's
    // ~86 a second, so the cost that ruled logging out there does not apply.
    private func log(_ what: @autoclosure () -> String) {
        Diag.shared.report("[tts] " + what())
    }

    // The counters behind every decision here, so a line says what the player
    // believed at that moment rather than only what it did.
    private func counts() -> String {
        lock.lock()
        defer { lock.unlock() }
        return "inFlight \(inFlight) scheduled \(scheduled) "
            + "pending \(pending.count) epoch \(epoch) "
            + "paused \(isPaused) running \(running)"
    }

    func enqueue(_ text: String, tag: Int, voice: String, speed: Float) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            lock.lock()
            pending.append((trimmed, tag))
            lock.unlock()
            log("enqueue \(trimmed.count) chars | \(counts())")
            pump(voice: voice, speed: speed)
        }
    }

    // Silence now: drop what is queued, forget what is being synthesized,
    // and stop the node mid-buffer.
    func stop() {
        lock.lock()
        pending.removeAll()
        epoch += 1
        scheduled = 0
        isPaused = false
        playing.removeAll()
        lock.unlock()
        // The STATE above is what makes a stop immediate -- nothing new is
        // scheduled once the epoch moves -- so silencing the node a queue hop
        // later costs no correctness.
        nodeQueue.async { [weak self] in
            self?.node.stop()
            self?.node.reset()
        }
        log("stop | \(counts())")
        notify()
        notifySpeaking()
    }

    func pause() {
        lock.lock()
        isPaused = true
        lock.unlock()
        nodeQueue.async { [weak self] in self?.node.pause() }
        log("pause | \(counts())")
    }

    func resume() {
        lock.lock()
        isPaused = false
        lock.unlock()
        startEngine()
        nodeQueue.async { [weak self] in self?.node.play() }
        log("resume | \(counts())")
    }

    // Take one pending segment if there is room, synthesize it off-thread and
    // schedule the result. Re-entered on every completion, so the queue
    // drains without a timer.
    private func pump(voice: String, speed: Float) {
        lock.lock()
        let room = inFlight + scheduled < VoicePlayer.readAhead
        let next = (room && !pending.isEmpty) ? pending.removeFirst() : nil
        if next != nil { inFlight += 1 }
        let mark = epoch
        lock.unlock()
        if let next {
            notify()
            synthQueue.async { [weak self] in
                self?.render(next.text, tag: next.tag, voice: voice,
                             speed: speed, mark: mark)
            }
        }
    }

    private func render(_ text: String, tag: Int, voice: String,
                        speed: Float, mark: Int) {
        let picked = Speech.voice(named: voice)
        let t0 = Date()
        let pcm = ready()?.synthesize(text, voice: picked, speed: speed) ?? []
        lock.lock()
        inFlight -= 1
        let stale = mark != epoch
        lock.unlock()
        log(String(format: "render %d chars -> %d samples in %.2fs, %@ | %@",
                   text.count, pcm.count, Date().timeIntervalSince(t0),
                   stale ? "STALE" : "live", counts()))
        if !stale && !pcm.isEmpty {
            schedule(pcm, tag: tag, mark: mark, voice: voice, speed: speed)
        } else {
            notify()
            pump(voice: voice, speed: speed)
        }
    }

    // The claim on `scheduled` is taken in the SAME lock as the staleness
    // test, and that is the whole point of this shape: `render` tested `mark`
    // and then let go of the lock, so a stop can land in the gap. A count
    // raised after one can never be lowered again -- `finished` refuses a
    // stale mark -- and the node it was queued on has just been reset, so
    // while paused the buffer never plays back and never completes either.
    // `isActive` then stays true with nothing left to play, which the UI shows
    // as a voice that is speaking forever in silence.
    private func schedule(_ pcm: [Float], tag: Int, mark: Int, voice: String,
                          speed: Float) {
        let frames = AVAudioFrameCount(pcm.count)
        let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                      frameCapacity: frames)
        if let buffer, let channel = buffer.floatChannelData {
            buffer.frameLength = frames
            pcm.withUnsafeBufferPointer { src in
                channel[0].update(from: src.baseAddress!, count: pcm.count)
            }
            lock.lock()
            let live = mark == epoch
            if live { scheduled += 1; playing.append(tag) }
            let held = isPaused
            lock.unlock()
            if live {
                startEngine()
                node.scheduleBuffer(buffer, completionCallbackType:
                                        .dataPlayedBack) { [weak self] _ in
                    self?.finished(mark: mark, voice: voice, speed: speed)
                }
                if !held { node.play() }
            }
            notify()
            notifySpeaking()
        }
        pump(voice: voice, speed: speed)
    }

    // The head of `playing` is what is audible, so a completion retires it and
    // whatever was queued behind it becomes the voice's current place.
    private func finished(mark: Int, voice: String, speed: Float) {
        lock.lock()
        if mark == epoch && scheduled > 0 { scheduled -= 1 }
        if mark == epoch && !playing.isEmpty { playing.removeFirst() }
        lock.unlock()
        notify()
        notifySpeaking()
        pump(voice: voice, speed: speed)
    }

    private func startEngine() {
        lock.lock()
        let already = running
        running = true
        lock.unlock()
        if !already {
            AudioSession.beginPlayback()
            try? engine.start()
        }
    }

    // The engine keeps running between sentences on purpose: starting it per
    // sentence costs a hardware route change, which is audible as a click.
    func idle() {
        lock.lock()
        let quiet = inFlight == 0 && scheduled == 0 && pending.isEmpty
        let wasRunning = running
        if quiet { running = false }
        lock.unlock()
        if quiet && wasRunning {
            engine.pause()
            AudioSession.endPlayback()
        }
    }

    private func notify() {
        onActivity?(isActive)
    }

    private func notifySpeaking() {
        lock.lock()
        let head = playing.first
        lock.unlock()
        onSpeaking?(head)
    }
}

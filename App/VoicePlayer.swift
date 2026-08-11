import AVFoundation
import Foundation
import LLM

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

    // Transport calls (stop/pause/resume) run here, off the main actor:
    // AVAudioPlayerNode.stop() blocks until the render thread drains.
    // `qos: .default` cannot be dropped -- a queue built without one is
    // UNSPECIFIED and inherits the submitter's class.

    private let nodeQueue = DispatchQueue(label: "gadeon.tts.node",
                                          qos: .default)
    private var speech: Speech?
    // Each segment's tag is opaque here; the caller uses it to track which
    // piece of its own text is playing.
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

    // 4 GB avoids jetsam from KittenTTS plus the audio engine alongside a
    // resident model.
    static let speechFloorGB = 4

    init?() {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                sampleRate: Double(Speech.sampleRate),
                                channels: 1, interleaved: false)
        if let fmt, installedGB >= VoicePlayer.speechFloorGB {
            format = fmt
            engine.attach(node)
            engine.connect(node, to: engine.mainMixerNode, format: fmt)
        } else {
            return nil
        }
    }

    // Built lazily off the main thread: loading the weights is tens of
    // milliseconds, paid only if speech is used.

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

    private func log(_ what: @autoclosure () -> String) {
        Diag.shared.report("[tts] " + what())
    }

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

    func stop() {
        lock.lock()
        pending.removeAll()
        epoch += 1
        scheduled = 0
        isPaused = false
        playing.removeAll()
        lock.unlock()
        // The epoch bump above makes this immediate -- nothing new schedules
        // once it moves -- so silencing the node a queue hop later costs no
        // correctness.
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

    // startEngine rides the SAME hop as the play it must precede: it starts
    // the engine and the audio session, which are hardware calls, and this is
    // a transport call off the main actor like the two above.
    func resume() {
        lock.lock()
        isPaused = false
        lock.unlock()
        nodeQueue.async { [weak self] in
            self?.startEngine()
            self?.node.play()
        }
        log("resume | \(counts())")
    }

    // Re-entered on every completion, so the queue drains without a timer.
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
    // test: `render` tests `mark` and releases the lock, so a stop can
    // land in the gap this closes.

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

    // The engine keeps running between sentences on purpose -- restarting
    // it per sentence is audible as a click.

    func idle() {
        lock.lock()
        let quiet = inFlight == 0 && scheduled == 0 && pending.isEmpty
        let wasRunning = running
        if quiet { running = false }
        lock.unlock()
        if quiet && wasRunning {
            nodeQueue.async { [weak self] in
                self?.engine.pause()
                AudioSession.endPlayback()
            }
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

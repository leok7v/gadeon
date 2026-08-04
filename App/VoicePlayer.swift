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
    private var speech: Speech?
    private var pending: [String] = []
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

    func enqueue(_ text: String, voice: String, speed: Float) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            lock.lock()
            pending.append(trimmed)
            lock.unlock()
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
        lock.unlock()
        node.stop()
        node.reset()
        notify()
    }

    func pause() {
        lock.lock()
        isPaused = true
        lock.unlock()
        node.pause()
    }

    func resume() {
        lock.lock()
        isPaused = false
        lock.unlock()
        startEngine()
        node.play()
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
                self?.render(next, voice: voice, speed: speed, mark: mark)
            }
        }
    }

    private func render(_ text: String, voice: String, speed: Float,
                        mark: Int) {
        let picked = Speech.voice(named: voice)
        let pcm = ready()?.synthesize(text, voice: picked, speed: speed) ?? []
        lock.lock()
        inFlight -= 1
        let stale = mark != epoch
        lock.unlock()
        if !stale && !pcm.isEmpty {
            schedule(pcm, mark: mark, voice: voice, speed: speed)
        } else {
            notify()
            pump(voice: voice, speed: speed)
        }
    }

    private func schedule(_ pcm: [Float], mark: Int, voice: String,
                          speed: Float) {
        let frames = AVAudioFrameCount(pcm.count)
        let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                      frameCapacity: frames)
        if let buffer, let channel = buffer.floatChannelData {
            buffer.frameLength = frames
            pcm.withUnsafeBufferPointer { src in
                channel[0].update(from: src.baseAddress!, count: pcm.count)
            }
            startEngine()
            lock.lock()
            scheduled += 1
            lock.unlock()
            node.scheduleBuffer(buffer, completionCallbackType:
                                    .dataPlayedBack) { [weak self] _ in
                self?.finished(mark: mark, voice: voice, speed: speed)
            }
            lock.lock()
            let held = isPaused
            lock.unlock()
            if !held { node.play() }
            notify()
        }
        pump(voice: voice, speed: speed)
    }

    private func finished(mark: Int, voice: String, speed: Float) {
        lock.lock()
        if mark == epoch && scheduled > 0 { scheduled -= 1 }
        lock.unlock()
        notify()
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
}

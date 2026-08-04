// The microphone as a stream of mono f32 samples at a requested rate.
//
// Model-agnostic for the same reason AudioFile is: the rate is a PARAMETER,
// because it is a property of a model's frontend rather than of audio. The
// hardware picks its own rate and AVAudioConverter brings it to ours, since
// asking the engine to run at 16 kHz fails on hardware that will not.
//
// The tap fires on a real-time audio thread and this does allocate on it --
// the converted block is a fresh array. That is a known compromise, not an
// oversight: the buffer is 4096 frames (~85 ms of slack) and the work is one
// convert plus one copy, so a stall would have to be extraordinary to glitch.
// If it ever does, the fix is a lock-free ring here and the gating moved to a
// worker, NOT a smaller buffer. What must never appear in this callback is a
// model, the GPU, or an await.
//
// NO AVAudioSession here. macOS has none, iOS wants a category and an
// activation bound to the app's lifecycle (interruptions, route changes,
// backgrounding), and this module builds under SwiftPM where the repo's
// platform-file split does not reach -- `swift build` compiles all of src/.
// So the CALLER prepares whatever session its platform has, and on macOS
// that is nothing at all.
@preconcurrency import AVFoundation
import Foundation

public final class Microphone: @unchecked Sendable {
    public enum Failure: Error, CustomStringConvertible {
        case denied
        case unavailable(String)

        public var description: String {
            let out: String
            switch self {
            case .denied:
                out = "Microphone access was refused."
            case .unavailable(let why):
                out = "The microphone is unavailable: \(why)"
            }
            return out
        }
    }

    private let engine = AVAudioEngine()
    private let rate: Double
    private var converter: AVAudioConverter?
    private let lock = NSLock()
    private var running = false

    public init(rate: Double) { self.rate = rate }

    // Ask once, up front: a tap installed before the answer arrives records
    // silence on iOS and throws on macOS.
    public static func permission() async -> Bool {
        await AVCaptureDevice.requestAccess(for: .audio)
    }

    // `onSamples` fires on the audio thread with mono f32 at the requested
    // rate. It must not block.
    public func start(_ onSamples: @escaping @Sendable ([Float]) -> Void)
        throws {
        lock.lock()
        defer { lock.unlock() }
        if !running {
            let input = engine.inputNode
            let native = input.outputFormat(forBus: 0)
            guard native.sampleRate > 0,
                  let want = AVAudioFormat(
                    commonFormat: .pcmFormatFloat32, sampleRate: rate,
                    channels: 1, interleaved: false) else {
                throw Failure.unavailable("no input format")
            }
            converter = AVAudioConverter(from: native, to: want)
            // A hardware-sized buffer: the gate holds its own hop remainder,
            // so the block size the tap happens to use changes nothing.
            input.installTap(onBus: 0, bufferSize: 4096, format: native) {
                [weak self] buffer, _ in
                if let s = self?.mono(buffer, to: want), !s.isEmpty {
                    onSamples(s)
                }
            }
            engine.prepare()
            do {
                try engine.start()
            } catch {
                input.removeTap(onBus: 0)
                throw Failure.unavailable(error.localizedDescription)
            }
            running = true
        }
    }

    public func stop() {
        lock.lock()
        defer { lock.unlock() }
        if running {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
            running = false
        }
    }

    // One captured buffer as mono f32 at our rate. Returns empty rather than
    // throwing: a dropped block on the audio thread is a glitch to survive,
    // not an error to propagate into a real-time callback.
    // One buffer, handed over once. A final class rather than a captured
    // var so the conversion block owns no mutable state of ours.
    //
    // @unchecked Sendable is true here rather than asserted: the block runs
    // SYNCHRONOUSLY on the thread that called convert(), which is the audio
    // thread this instance was created on and dies on. It is never shared.
    private final class OneShot: @unchecked Sendable {
        private var buffer: AVAudioPCMBuffer?
        init(_ b: AVAudioPCMBuffer) { buffer = b }
        func next(_ status: UnsafeMutablePointer<AVAudioConverterInputStatus>)
            -> AVAudioPCMBuffer? {
            let out = buffer
            status.pointee = out == nil ? .noDataNow : .haveData
            buffer = nil
            return out
        }
    }

    private func mono(_ buffer: AVAudioPCMBuffer,
                      to want: AVAudioFormat) -> [Float] {
        var out: [Float] = []
        let ratio = want.sampleRate / buffer.format.sampleRate
        let cap = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1024
        if let converter,
           let dst = AVAudioPCMBuffer(pcmFormat: want, frameCapacity: cap) {
            // The block is called synchronously inside convert(), and it has
            // exactly one buffer to give: hand it over on the first call and
            // report starvation on every later one, or the converter loops
            // asking for input that will never come.
            let source = OneShot(buffer)
            var err: NSError?
            converter.convert(to: dst, error: &err) { _, status in
                source.next(status)
            }
            if err == nil, let ch = dst.floatChannelData {
                out = Array(UnsafeBufferPointer(start: ch[0],
                                                count: Int(dst.frameLength)))
            }
        }
        return out
    }
}

import AVFoundation
import Foundation

// Playback no-ops while `recording` is true: its idle callback can land
// after the mic has already claimed the session, and deactivating then
// would kill a live tap.

enum AudioSession {

    nonisolated(unsafe) private static var recording = false
    private static let lock = NSLock()

    static func beginRecording() {
        lock.lock()
        recording = true
        lock.unlock()
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.record, mode: .measurement)
        try? session.setActive(true)
    }

    static func endRecording() {
        lock.lock()
        recording = false
        lock.unlock()
        try? AVAudioSession.sharedInstance().setActive(
            false, options: [.notifyOthersOnDeactivation])
    }

    static func beginPlayback() {
        lock.lock()
        let busy = recording
        lock.unlock()
        if !busy {
            let session = AVAudioSession.sharedInstance()
            try? session.setCategory(.playback, mode: .spokenAudio)
            try? session.setActive(true)
        }
    }

    static func endPlayback() {
        lock.lock()
        let busy = recording
        lock.unlock()
        if !busy {
            try? AVAudioSession.sharedInstance().setActive(
                false, options: [.notifyOthersOnDeactivation])
        }
    }

    static func describe() -> String {
        let s = AVAudioSession.sharedInstance()
        let route = s.currentRoute.inputs
            .map { i in "\(i.portType.rawValue)/\(i.portName)" }
            .joined(separator: ",")
        return String(format:
            "cat=%@ mode=%@ inputAvailable=%@ rate=%.0f inputs=[%@] "
            + "channels=%d",
            s.category.rawValue, s.mode.rawValue,
            s.isInputAvailable ? "yes" : "NO",
            s.sampleRate, route.isEmpty ? "none" : route,
            s.inputNumberOfChannels)
    }
}

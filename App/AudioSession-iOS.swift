import AVFoundation
import Foundation

// The one audio session, and the category each use needs of it. They are
// mutually exclusive on iOS -- .record makes output silent -- so the
// microphone and the voice take turns, and barge-in stops the speech BEFORE
// claiming the input.
//
// WHO IS USING IT is tracked because the two ends do not arrive in order. The
// player reports itself idle on a hop to the main actor, so its endPlayback
// can land AFTER the microphone has already claimed and activated the
// session; deactivating there leaves the tap running against a dead session,
// which delivers 44.1 kHz of digital silence rather than an error. The
// symptom is a full recording with no speech in it and the gate's bar sitting
// exactly on its floor.
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

    // .playback rather than .ambient: a spoken reply is the thing the user is
    // listening to, so it plays over other audio and through the silent
    // switch, which is where a muted phone would otherwise swallow it whole.
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

    // What the session actually IS, for the log. A microphone that yields
    // silence looks identical from the app's side to one that yields a quiet
    // room; the category, the route and the input count are what separate
    // them, and none of it is inferable after the fact.
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

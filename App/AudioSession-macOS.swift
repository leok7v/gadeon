import Foundation

// macOS has no AVAudioSession: input routing, output routing and category are
// the system's business. The iOS twin declares the same symbols and does the
// real work.
enum AudioSession {
    static func beginRecording() { }
    static func endRecording() { }
    static func beginPlayback() { }
    static func endPlayback() { }
    static func describe() -> String { "macOS: no session" }
}

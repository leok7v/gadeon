import Foundation
import LLM

public enum DiagGate: String, CaseIterable, Identifiable {

    case memory
    case beat
    case hang
    case pressure
    case transcript

    public var id: String { rawValue }

    public var label: String {
        let out: String
        switch self {
        case .memory: out = "Memory detail"
        case .beat: out = "Screen rebuilds"
        case .hang: out = "Main thread stalls"
        case .pressure: out = "Memory pressure"
        case .transcript: out = "Session transcript"
        }
        return out
    }

    public var detail: String {
        let out: String
        switch self {
        case .memory:
            out = "Record memory use step by step during downloads and "
                + "while a prompt is processed, so a climb shows as a curve."
        case .beat:
            out = "Monitor how often the chat screen is rebuilt and redrawn."
        case .hang:
            out = "Log the moments the app freezes, and whether it was busy "
                + "or waiting."
        case .pressure:
            out = "Log the system's low memory warnings."
        case .transcript:
            out = "Save everything sent to and received from the model, "
                + "word for word, in a local log file. Starts at the next "
                + "launch."
        }
        return out
    }

    public var flag: String { "log-\(rawValue)" }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var answers: [String: Bool] = [:]

    public var on: Bool {
        DiagGate.lock.lock()
        defer { DiagGate.lock.unlock() }
        if DiagGate.answers[rawValue] == nil {
            let text = Flags.value(flag) ?? ""
            DiagGate.answers[rawValue] = text.isEmpty
                ? self == .transcript
                : Flags.truthy(text)
        }
        return DiagGate.answers[rawValue] ?? false
    }

    public func set(_ value: Bool) {
        DiagGate.lock.lock()
        DiagGate.answers[rawValue] = value
        DiagGate.lock.unlock()
        Flags.store(flag, value)
    }

}

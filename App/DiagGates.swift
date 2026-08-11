import Foundation

enum DiagGate {
    static let memWatch = set("GADEON_LOG_MEM")
    static let beat = set("GADEON_LOG_BEAT")
    static let hang = !set("GADEON_LOG_NO_HANG")

    // Present and not an explicit denial: existence is the signal.

    private static func set(_ name: String) -> Bool {
        let v = ProcessInfo.processInfo.environment[name] ?? ""
        return !v.isEmpty && v != "0" && v.lowercased() != "false"
    }
}

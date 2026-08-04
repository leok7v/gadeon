import Foundation

// macOS has no os_proc_available_memory: there is no per-process jetsam limit
// to be near, so the honest answer is that the question does not apply rather
// than a number invented from free RAM.
extension Footprint {
    static func availableBytes() -> UInt64? { nil }
}

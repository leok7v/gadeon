import Foundation

// Which diagnostic channels write to the log, as environment switches declared
// in project.yml with `isEnabled: false` -- ready checkboxes in the scheme
// editor, the same shape OS_ACTIVITY_MODE and ANE_CACHE_MODE already use. So
// turning one on is a click rather than an edit here.
//
// The two loudest default OFF. CENSUS of one captured session, 836 lines:
// [mem] watch 244, [beat] 138, [app] 18, [hang] 8, [stall] 2 -- the first two
// are 91% of it. Both are already rate-limited (a beat coalesces to one line
// per second per label, and watch prints only on a real move), so this is a
// gate rather than a throttle.
//
// What survives with mem watch off is the TAGGED footprint reports -- before a
// model load, after it, at prefill start, after an encode. Those are the
// deltas across a moment, which is the question worth asking; the watch is the
// fine-grained curve between them, and it earns its noise only while actually
// hunting a jetsam.
//
// [hang] defaults ON. Eight lines in that session and each named a real stall,
// which is the whole reason it exists -- but a clean log for a screenshot is a
// fair ask, so it has an off switch too. Named for what checking it DOES, so
// every box in the scheme editor points the same way.

enum DiagGate {
    static let memWatch = set("GADEON_LOG_MEM")
    static let beat = set("GADEON_LOG_BEAT")
    static let hang = !set("GADEON_LOG_NO_HANG")

    // Present and not an explicit denial. A checkbox in Xcode carries whatever
    // value the yml gave it, so "the variable exists" is the real signal and
    // "0" is spelled out for anyone setting it from a shell instead.
    private static func set(_ name: String) -> Bool {
        let v = ProcessInfo.processInfo.environment[name] ?? ""
        return !v.isEmpty && v != "0" && v.lowercased() != "false"
    }
}

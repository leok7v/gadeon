import Foundation
import LLM

enum Footprint {

    struct Sample {
        let footprint: UInt64
        let resident: UInt64
        let dirty: UInt64
        let fileBacked: UInt64
        let peak: UInt64
        let headroomToJetsam: UInt64?
        let systemWired: UInt64
        let systemFree: UInt64
    }

    private static func systemVM() -> (wired: UInt64, free: UInt64) {
        var info = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size
                / MemoryLayout<integer_t>.size)
        let ok = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { q in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, q, &count)
            }
        }
        var out: (wired: UInt64, free: UInt64) = (0, 0)
        if ok == KERN_SUCCESS {
            // getpagesize(), not vm_page_size: the latter is a mutable global,
            // which strict concurrency refuses.
            let page = UInt64(getpagesize())
            out = (UInt64(info.wire_count) * page,
                   UInt64(info.free_count) * page)
        }
        return out
    }

    // Spells out TASK_VM_INFO_COUNT, which Swift cannot import: a wrong
    // denominator makes task_info fail and the memory log go silently dead.

    static func sample() -> Sample? {
        var info = task_vm_info_data_t()
        let info_data = MemoryLayout<task_vm_info_data_t>.size
        let natural = MemoryLayout<natural_t>.size
        var count = mach_msg_type_number_t(info_data / natural)
        let n = Int(count)
        let ok = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: n) { q in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), q,
                          &count)
            }
        }
        var out: Sample? = nil
        if ok == KERN_SUCCESS {
            let vm = systemVM()
            out = Sample(
                footprint: UInt64(info.phys_footprint),
                resident: UInt64(info.resident_size),
                dirty: UInt64(info.internal) + UInt64(info.compressed),
                fileBacked: UInt64(info.external),
                peak: UInt64(info.ledger_phys_footprint_peak),
                headroomToJetsam: Footprint.availableBytes(),
                systemWired: vm.wired, systemFree: vm.free)
        }
        return out
    }

    private static let stepMB: UInt64 = 8 * 1_048_576
    private static let tickMS = 250
    // These three are touched ONLY from `queue`, which is what makes the
    // unchecked annotations true rather than a silencer.
    nonisolated(unsafe) private static var last: UInt64 = 0
    nonisolated(unsafe) private static var lastWired: UInt64 = 0

    private static func moved(_ now: UInt64, _ then: UInt64) -> Bool {
        (now > then ? now - then : then - now) >= stepMB
    }

    private static let queue =
        DispatchQueue(label: "io.github.leok7v.gadeon.footprint")

    nonisolated(unsafe) private static var timer: DispatchSourceTimer?

    private static func describeDevice() {
        let m = ProcessInfo.processInfo
        let bytes = m.physicalMemory
        let tier = (bytes + (1 << 29)) >> 30
        let gib = (Double(bytes) / 1_073_741_824 * 100).rounded(.down) / 100
        Diag.shared.report(String(
            format: "[mem] device: %.2f GiB physical (tier %d GB), "
                + "%d cores, %@ %@",
            gib, tier, m.processorCount,
            m.operatingSystemVersionString, hardwareID()))
        Diag.shared.report("[mem] offers \(Models.all.joined(separator: ", "))"
            + (debugBuild ? " (debug)" : ""))
    }

    static func watch() {
        queue.async {
            if timer == nil {
                describeDevice()
                if DiagGate.memWatch {
                    let t = DispatchSource.makeTimerSource(queue: queue)
                    t.schedule(deadline: .now() + .milliseconds(tickMS),
                               repeating: .milliseconds(tickMS))
                    t.setEventHandler {
                        if let s = sample(),
                           moved(s.footprint, last)
                               || moved(s.systemWired, lastWired) {
                            last = s.footprint
                            lastWired = s.systemWired
                            report("watch")
                        }
                    }
                    timer = t
                    t.resume()
                }
            }
        }
    }

    static func report(_ tag: String,
                       file: String = #fileID, line: Int = #line) {
        if let s = sample() {
            func mb(_ v: UInt64) -> String {
                String(format: "%.0f MB", Double(v) / 1_048_576)
            }
            let head = s.headroomToJetsam.map { v in mb(v) } ?? "n/a"
            Diag.shared.report(
                "[mem] \(tag): footprint \(mb(s.footprint)) "
                + "(peak \(mb(s.peak)), headroom \(head)) | "
                + "resident \(mb(s.resident)) = dirty \(mb(s.dirty)) "
                + "+ file-backed \(mb(s.fileBacked)) | SYSTEM wired "
                + "\(mb(s.systemWired)) free \(mb(s.systemFree))",
                file: file, line: line)
        }
    }
}

extension Footprint {

    static func hardwareID() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var buf = [UInt8](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.machine", &buf, &size, nil, 0)
        let bytes = buf.prefix(while: { b in b != 0 })
        return String(decoding: bytes, as: UTF8.self)
    }

}

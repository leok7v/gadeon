import Foundation
import LLM

// What the app actually costs in memory, in the terms iOS kills on.
//
// The number that matters is phys_footprint, not resident size: jetsam decides
// on footprint, and the two differ by exactly the thing this app leans on --
// a model's weights are mmap'd READ-ONLY from the downloaded file, so they are
// clean file-backed pages the kernel can evict and reclaim under pressure.
// RSS counts them while they are resident; footprint does not hold them
// against you. That distinction is the whole argument for offering a 2.7 GB
// model on a 6 GB device, and until now it was an argument nobody had checked.
//
// So each report carries all of it: footprint (what kills you), resident (what
// is in RAM now), the dirty split (internal + compressed, which is the part
// eviction CANNOT reclaim), and the file-backed part (the mmap). `headroom` is
// how much footprint is left before this process is killed -- iOS answers that
// directly, macOS has no equivalent and says so.
enum Footprint {

    struct Sample {
        let footprint: UInt64     // phys_footprint -- the jetsam number
        let resident: UInt64      // resident_size -- what is in RAM now
        let dirty: UInt64         // internal + compressed -- unreclaimable
        let external: UInt64      // file-backed: the mmap'd weights
        let peak: UInt64          // ledger_phys_footprint_peak
        let headroom: UInt64?     // bytes before jetsam; nil where unanswerable
        let wired: UInt64         // SYSTEM-wide, the GPU's weights land here
        let freeRAM: UInt64       // SYSTEM-wide
    }

    // System-wide wired and free bytes, and the only numbers that describe
    // GPU memory at all.
    //
    // A bytesNoCopy MTLBuffer's pages are WIRED into kernel_task when a
    // command buffer references them. phys_footprint does not count that, and
    // os_proc_available_memory does not know about it -- it answers how far
    // this process is from its OWN limit. MEASURED on a 4 GB iPhone: the app
    // read 121 MB with 2226 MB of "headroom" while the machine had 30 MB free
    // and 3050 MB wired, and was killed for vm-pageshortage. Every per-process
    // number was healthy and every one of them was answering a question
    // nobody asked.
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
            // The counts are in host pages. getpagesize() rather than
            // vm_page_size: the latter is a mutable global, which strict
            // concurrency refuses, and the two agree on every platform this
            // ships to (16384 on arm64, 4096 on x86_64).
            let page = UInt64(getpagesize())
            out = (UInt64(info.wire_count) * page,
                   UInt64(info.free_count) * page)
        }
        return out
    }

    // task_vm_info is the one call that carries phys_footprint; the older
    // mach_task_basic_info has resident and virtual only, which is what makes
    // most "memory used" numbers on this platform the wrong ones.
    static func sample() -> Sample? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let ok = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { q in
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
                external: UInt64(info.external),
                peak: UInt64(info.ledger_phys_footprint_peak),
                headroom: Footprint.availableBytes(),
                wired: vm.wired, freeRAM: vm.free)
        }
        return out
    }

    // A steady sampler, because the interesting moment is not one we can name
    // in advance. The load reports were the obvious call sites and they miss
    // the answer entirely: an mmap is LAZY, so right after a load the weights
    // are not resident and the footprint is tiny. What settles it is what
    // happens once inference has touched every layer -- and that peak arrives
    // mid-decode, where no natural hook exists.
    //
    // Reports only on a MOVE of at least `stepMB`, so a quiet session stays
    // silent and a growing one produces a curve rather than a log flood.
    //
    // The rates are set by the FASTEST thing worth catching, not by what is
    // comfortable to read. MEASURED on a 4 GB iPhone: the first prefill went
    // from 125 MB to jetsam in under 1.4 s, so a 2 s tick sampled only the
    // calm before it and every number in that log was from before the event.
    // A quarter second over a 32 MB step cannot miss a climb that steep.
    //
    // A move in EITHER footprint or system wired counts. Watching footprint
    // alone is what let the 4 GB kill pass unobserved: it sat at 121 MB from
    // the first prefill chunk to the grave while wired climbed to 3050 MB.
    private static let stepMB: UInt64 = 8 * 1_048_576
    private static let tickMS = 250
    nonisolated(unsafe) private static var last: UInt64 = 0
    nonisolated(unsafe) private static var lastWired: UInt64 = 0

    private static func moved(_ now: UInt64, _ then: UInt64) -> Bool {
        (now > then ? now - then : then - now) >= stepMB
    }
    private static let queue =
        DispatchQueue(label: "io.github.leok7v.gadeon.footprint")
    // Both are touched ONLY from `queue`, which is what makes the unchecked
    // annotation true rather than a silencer for the compiler.
    nonisolated(unsafe) private static var timer: DispatchSourceTimer?

    // Say which tier this device actually landed in, once, at launch.
    // `devicectl` reports neither RAM nor chip -- marketingName and
    // productType are all it has -- so without this the only way to know
    // whether a phone is the 6 GB case or the 8 GB one is to recall a spec
    // sheet, which is exactly the kind of "fact" that turns out to have been
    // carried over from a different device.
    private static func describeDevice() {
        let m = ProcessInfo.processInfo
        let bytes = m.physicalMemory
        // The same rounding Models.all gates on, so the log names the tier
        // rather than a number the reader has to re-round.
        let tier = (bytes + (1 << 29)) >> 30
        // TRUNCATE rather than round the GiB figure. Rounding printed a
        // 7.4999 GiB device as "7.50 GB" beside "tier 7", which reads as a
        // bug in the tiering and invites someone to "fix" the arithmetic
        // Models.all actually gates on.
        let gib = (Double(bytes) / 1_073_741_824 * 100).rounded(.down) / 100
        Diag.shared.report(String(
            format: "[mem] device: %.2f GiB physical (tier %d GB), "
                + "%d cores, %@ %@",
            gib, tier, m.processorCount,
            m.operatingSystemVersionString, hardwareID()))
    }

    // The device line is unconditional -- one fact, once, and the tier it
    // names is what every memory decision in the app is gated on. Only the
    // 250 ms curve is behind the switch; see DiagGate for what that costs.
    // Called once, from Instrument.install.
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
                               || moved(s.wired, lastWired) {
                            last = s.footprint
                            lastWired = s.wired
                            report("watch")
                        }
                    }
                    timer = t
                    t.resume()
                }
            }
        }
    }

    // One line into the shared Diag, tagged so a devicectl pull can grep it.
    // `tag` names the moment -- a model load, an attachment encode -- because
    // the interesting question is always the DELTA across one of those.
    static func report(_ tag: String,
                       file: String = #fileID, line: Int = #line) {
        if let s = sample() {
            func mb(_ v: UInt64) -> String {
                String(format: "%.0f MB", Double(v) / 1_048_576)
            }
            let head = s.headroom.map { v in mb(v) } ?? "n/a"
            Diag.shared.report(
                "[mem] \(tag): footprint \(mb(s.footprint)) "
                + "(peak \(mb(s.peak)), headroom \(head)) | "
                + "resident \(mb(s.resident)) = dirty \(mb(s.dirty)) "
                + "+ file-backed \(mb(s.external)) | SYSTEM wired "
                + "\(mb(s.wired)) free \(mb(s.freeRAM))",
                file: file, line: line)
        }
    }
}

// The machine identifier (iPhone16,1 / iPad16,1 / Mac...) -- sysctl's answer,
// not a marketing name, because that is what a spec lookup keys on.
extension Footprint {
    static func hardwareID() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var buf = [UInt8](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.machine", &buf, &size, nil, 0)
        // sysctl counts the NUL; decoding it would append a stray scalar.
        let bytes = buf.prefix(while: { b in b != 0 })
        return String(decoding: bytes, as: UTF8.self)
    }
}

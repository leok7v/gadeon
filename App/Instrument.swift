import Foundation
import LLM
import MD

enum Instrument {

    // Wall-clock zero for the debug view's timestamps.
    static let launched = Date()

    static func install() -> Bool {
        // Forces the lazy static now; a first read from the debug view,
        // opened minutes later, would date the launch late.
        _ = launched
        MarkdownDiag.report = { s in Diag.shared.report(s) }
        MainThreadWatch.shared.start()
        Footprint.watch()
        // LLM reports through this closure since it cannot import App
        // directly; the detail twin stays nil unless asked for.
        Diag.memory = { tag in Footprint.report(tag) }
        if DiagGate.memWatch {
            Diag.memoryDetail = { tag in Footprint.report(tag) }
        }
        return true
    }

    // Reports only when a span overruns the 12ms frame budget.

    @discardableResult
    static func timed<T>(_ label: @autoclosure () -> String,
                            over: Double = 12, _ body: () -> T) -> T {
        let t0 = DispatchTime.now()
        let result = body()
        let ms = Double(DispatchTime.now().uptimeNanoseconds
            - t0.uptimeNanoseconds) / 1e6
        if ms >= over {
            Diag.shared.report(String(format: "[app] %.1fms %@", ms, label()))
        }
        return result
    }

    @MainActor private static var beats: [String: (n: Int, at: Date)] = [:]

    // A line in the shared diagnostics from a file that does not import LLM.

    static func note(_ text: String) { Diag.shared.report(text) }

    @MainActor static func beat(_ label: String) {
        if DiagGate.beat {
            var b = beats[label] ?? (0, Date())
            b.n += 1
            let dt = Date().timeIntervalSince(b.at)
            if dt >= 1.0 {
                Diag.shared.report("[beat] \(label): \(b.n) evals in "
                    + String(format: "%.1fs", dt))
                b = (0, Date())
            }
            beats[label] = b
        }
    }
}

// Pings the main QUEUE, not the thread: a run loop stuck in one long
// CoreAnimation commit starves dispatch while AppKit keeps delivering
// events, so "queue starved" and "thread blocked" are different claims.
// The CPU figure tells them apart -- near zero means main waited on
// something outside this process, near the stall itself means our own
// code is hot.

final class MainThreadWatch: @unchecked Sendable {

    static let shared = MainThreadWatch()

    private static let thresholdMs = 200.0
    private let queue = DispatchQueue(label: "io.github.leok7v.gadeon.watch")
    private var timer: DispatchSourceTimer?
    private var started = false

    func start() {
        queue.async { [weak self] in
            if let self, !self.started {
                self.started = true
                let t = DispatchSource.makeTimerSource(queue: self.queue)
                t.schedule(deadline: .now() + .milliseconds(100),
                           repeating: .milliseconds(100))
                t.setEventHandler { [weak self] in self?.ping() }
                self.timer = t
                t.resume()
            }
        }
    }

    // Every ping queued during a block runs when it ends, so `sent` before
    // the last report is the test for "same block" -- only the first of a
    // burst is a measurement.

    private var drained = DispatchTime.now().uptimeNanoseconds

    // Touched only from the block below, which runs on main -- also what
    // makes mach_thread_self() name the right thread. The send right is
    // held for the process, never released.

    private var mainThread: mach_port_t = 0
    private var mainCPUms = 0.0

    private func ping() {
        let sent = DispatchTime.now()
        DispatchQueue.main.async { [weak self] in
            let now = DispatchTime.now()
            let ms = Double(now.uptimeNanoseconds
                - sent.uptimeNanoseconds) / 1e6
            if let self {
                if self.mainThread == 0 {
                    self.mainThread = mach_thread_self()
                }
                let cpu = MainThreadWatch.cpuMs(self.mainThread)
                // Baseline predates the stall, so `burned` covers exactly
                // the starved span.
                let burned = cpu - self.mainCPUms
                self.mainCPUms = cpu
                if ms >= MainThreadWatch.thresholdMs,
                   sent.uptimeNanoseconds >= self.drained {
                    // Must advance even with the gate off, or the burst
                    // collapse misfires the moment it is switched on.
                    self.drained = now.uptimeNanoseconds
                    if DiagGate.hang {
                        Diag.shared.report(String(format:
                            "[hang] main queue starved %.0fms, main thread "
                            + "used %.0fms CPU", ms, burned))
                    }
                }
            }
        }
    }

    private static func cpuMs(_ thread: mach_port_t) -> Double {
        var info = thread_basic_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<thread_basic_info_data_t>.size
                / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self,
                                capacity: Int(count)) { raw in
                thread_info(thread, thread_flavor_t(THREAD_BASIC_INFO),
                            raw, &count)
            }
        }
        var ms = 0.0
        if kr == KERN_SUCCESS {
            let user_s    = info.user_time.seconds
            let user_us   = info.user_time.microseconds
            let system_s  = info.system_time.seconds
            let system_us = info.system_time.microseconds
            ms = Double(user_s + system_s) * 1000 +
                 Double(user_us + system_us) / 1000
        }
        return ms
    }

}

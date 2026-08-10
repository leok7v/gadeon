import Foundation
import LLM
import MD

// Debug instrumentation, wired once at launch: MD render timing flows into the
// shared Diag log, and a background watchdog measures how long the MAIN QUEUE
// goes undrained -- the direct signal for the "view flashes black" stall,
// timed independently of which sub-op blocked it. Both are frame-budget /
// threshold gated, so a smooth session is silent.

enum Instrument {
    // Wall-clock zero for the debug view's timestamps.
    static let launched = Date()

    static func install() -> Bool {
        // Fix that zero HERE. It is a lazy static, and the debug view may not
        // be opened for minutes -- reading it first from there would date the
        // launch to whenever someone tapped the ladybug.
        _ = launched
        MarkdownDiag.report = { s in Diag.shared.report(s) }
        MainThreadWatch.shared.start()
        Footprint.watch()
        // LLM reports through this because it cannot reach App directly. The
        // detail twin -- one line per prefill chunk, per downloaded file --
        // is left NIL unless asked for, so the cost of a silenced channel is
        // a nil check at the call site rather than a formatted string.
        Diag.memory = { tag in Footprint.report(tag) }
        if DiagGate.memWatch {
            Diag.memoryDetail = { tag in Footprint.report(tag) }
        }
        return true
    }

    // Time a main-actor span, reporting to Diag only when it overruns a frame
    // (12ms) -- the same frame-budget gate as MarkdownDiag, for App-side hot
    // spots (the incremental parse, the doc snapshot).
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

    // Count SwiftUI body re-evaluations per label and log the rate once a
    // second, so a per-token re-render storm surfaces as "N evals in ~1.0s".
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

// Pings the main QUEUE on a steady background timer and reports the round-trip
// latency: while the queue is not being drained the ping runs late by exactly
// that long, so a stall surfaces as one "[hang] ... Nms" line. Only stalls past
// the threshold are logged.
//
// The queue and the THREAD are different subjects, and the line names the one
// it measures. A run loop parked inside one long CoreAnimation commit starves
// dispatch blocks while AppKit keeps delivering events, so "the queue was
// starved for a minute" can be true while "the thread was blocked" is not --
// a 60s stall has been logged with the app writing its own button presses
// through the middle of it. The CPU figure separates the two cases worth
// telling apart: near zero means main was waiting on something outside this
// process, and a figure near the stall itself means our own code is hot.
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

    // Every ping queued DURING a block runs when it ends, so one stall
    // arrives as a descending run of them -- 888, 789, 689, ... one line per
    // 100 ms it lasted. Reported raw, a single 888 ms stall reads as seven
    // stalls, and a reader counting lines overstates it sevenfold. Only the
    // first of a burst is a measurement; every later one was already in
    // flight when it was reported, so `sent` before the last report is
    // exactly the test for "same block".
    private var drained = DispatchTime.now().uptimeNanoseconds

    // A send right to the main thread, and the CPU it had burned when the
    // last ping landed. Both are only ever touched from the block below,
    // which runs on main -- which is also what makes mach_thread_self() name
    // the right thread. The right is held for the process, never released.
    private var mainThread: mach_port_t = 0
    private var mainCPUms = 0.0

    private func ping() {
        let sent = DispatchTime.now()
        DispatchQueue.main.async { [weak self] in
            let now = DispatchTime.now()
            let ms = Double(now.uptimeNanoseconds
                - sent.uptimeNanoseconds) / 1e6
            if let self {
                if self.mainThread == 0 { self.mainThread = mach_thread_self() }
                let cpu = MainThreadWatch.cpuMs(self.mainThread)
                // Pings queued during a stall all run when it ends, oldest
                // first, so the one that reports it is also the one whose
                // baseline predates the stall -- the delta covers exactly the
                // starved span.
                let burned = cpu - self.mainCPUms
                self.mainCPUms = cpu
                if ms >= MainThreadWatch.thresholdMs,
                   sent.uptimeNanoseconds >= self.drained {
                    // `drained` still advances with the switch off, or the
                    // burst collapsing that makes one stall read as one line
                    // would come back wrong the moment it is switched on.
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
            ms = Double(info.user_time.seconds
                        + info.system_time.seconds) * 1000
                + Double(info.user_time.microseconds
                         + info.system_time.microseconds) / 1000
        }
        return ms
    }
}

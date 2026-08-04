import Foundation
import LLM
import MD

// Debug instrumentation, wired once at launch: MD render timing flows into the
// shared Diag log, and a background watchdog measures how long the MAIN thread
// is unresponsive -- the direct signal for the "view flashes black" stall,
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
        // LLM reports through this because it cannot reach App directly.
        Diag.memory = { tag in Footprint.report(tag) }
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
    @MainActor static func beat(_ label: String) {
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

// Pings the main queue on a steady background timer and reports the round-trip
// latency: while the main thread is blocked the ping runs late by exactly the
// block duration, so a stall surfaces as one "[hang] ... Nms" line naming how
// long the UI was frozen. Only stalls past the threshold are logged.
final class MainThreadWatch: @unchecked Sendable {

    static let shared = MainThreadWatch()

    private static let thresholdMs = 200.0
    private let queue = DispatchQueue(label: "io.github.leok7v.gadeon.watch")
    private var timer: DispatchSourceTimer?
    private var started = false

    func start() {
        queue.async { [weak self] in
            guard let self, !self.started else { return }
            self.started = true
            let t = DispatchSource.makeTimerSource(queue: self.queue)
            t.schedule(deadline: .now() + .milliseconds(100),
                       repeating: .milliseconds(100))
            t.setEventHandler { [weak self] in self?.ping() }
            self.timer = t
            t.resume()
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

    private func ping() {
        let sent = DispatchTime.now()
        DispatchQueue.main.async { [weak self] in
            let now = DispatchTime.now()
            let ms = Double(now.uptimeNanoseconds
                - sent.uptimeNanoseconds) / 1e6
            if let self, ms >= MainThreadWatch.thresholdMs,
               sent.uptimeNanoseconds >= self.drained {
                self.drained = now.uptimeNanoseconds
                Diag.shared.report(String(
                    format: "[hang] main thread stalled %.0fms", ms))
            }
        }
    }
}

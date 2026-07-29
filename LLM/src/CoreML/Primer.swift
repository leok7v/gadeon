import CoreML
import Foundation

// ANE cache priming: while a set downloads, each completed .mlmodelc is
// compiled IN-PROCESS on a background task -- loading an MLModel is what
// triggers the daemon compile; the model is discarded, the cache entry
// stays -- so the one-time "Optimizing" compile overlaps the download and
// the real load takes e5 cache hits. In-process because the caller is its
// own future loader (identical cache identity by construction) and because
// a sandboxed app cannot spawn a copy of its own binary: libsecinit traps
// the child bootstrapping a second app sandbox (SYSCALL_SET_USERLAND_PROFILE).
// Programs prime one at a time -- ANECompilerService serializes compiles
// globally. Runs on iOS too: the per-program compile peak is the same one
// the post-download build would pay minutes later (the increased-memory-
// limit entitlements cover it); priming only moves it under the download.
// Purely additive: any failure just leaves the cache cold.

@MainActor public final class Primer {
    private var queue: [URL] = []
    private var inFlight = false
    private var lastTop = ""

    public init() {}

    // Feed the in-flight repo path; a change of its top-level component
    // marks the previous component complete (HubFetch downloads the tree in
    // listing order, in place at the final cache-keyed paths).
    public func observe(_ file: String, set: URL) {
        let top = String(file.split(separator: "/").first ?? "")
        if top != lastTop, lastTop.hasSuffix(".mlmodelc") {
            queue.append(set.appendingPathComponent(lastTop))
        }
        lastTop = top
        pump()
    }

    // Download over (or the model switched): drop what has not started. An
    // in-flight program cannot be aborted; it finishes writing its cache
    // entry -- at worst the main load queues briefly behind that one
    // compile.
    public func cancel() {
        queue.removeAll()
        lastTop = ""
    }

    private func pump() {
        if !inFlight, !queue.isEmpty {
            inFlight = true
            let url = queue.removeFirst()
            Task.detached(priority: .utility) { [weak self] in
                for fn in PrimePlan.plan(at: url) {
                    let cfg = MLModelConfiguration()
                    cfg.computeUnits = .cpuAndNeuralEngine
                    if let fn { cfg.functionName = fn }
                    _ = try? MLModel(contentsOf: url, configuration: cfg)
                }
                await self?.primed()
            }
        }
    }

    private func primed() {
        inFlight = false
        pump()
    }
}

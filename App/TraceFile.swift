import Foundation
import LLM
import os

// Plain-text mirror of the session trace: every TraceEvent appends as a
// timestamped line with its payload indented under it, so the file reads as
// the session's full transcript (rendered deltas, raw decodes, tool results)
// and can be pasted instead of screenshots. Overwritten each launch -- it is
// always "the session being looked at", not an archive. The resolved path is
// the first line this subsystem logs, so `log stream` shows where to look.

final class TraceFile {
    private let log = Logger(subsystem: "io.github.leok7v.gadeon",
                             category: "trace")
    private let url: URL
    private let fmt: DateFormatter
    private var handle: FileHandle?

    init() {
        // Current run at transcript.log/current.txt (a stable, pullable name),
        // previous runs archived + pruned ~24h -- parallel to diag.log/.
        url = Diag.startRun(folder: "transcript.log")
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "HH:mm:ss.SSS"
        fmt = f
        log.info("transcript \(self.url.path, privacy: .public)")
    }

    var path: String { url.path }

    func note(_ text: String) {
        write(text + "\n")
    }

    func append(_ e: TraceEvent) {
        write(line(e) + "\n")
    }

    // "HH:mm:ss.SSS [kind] N tok ctx N 1.23s | summary" + indented payload.
    private func line(_ e: TraceEvent) -> String {
        let dur = e.t1.timeIntervalSince(e.t0)
        var head = fmt.string(from: e.t1) + " [" + e.kind.rawValue + "]"
        if e.tokens > 0 { head += " \(e.tokens) tok" }
        if e.ctx >= 0 { head += " ctx \(e.ctx)" }
        if dur >= 0.001 { head += String(format: " %.2fs", dur) }
        head += " | " + e.summary
        if !e.text.isEmpty {
            let body = e.text
                .split(separator: "\n", omittingEmptySubsequences: false)
                .map { part in "  " + part }
                .joined(separator: "\n")
            head += "\n" + body
        }
        return head
    }

    // Lazily (re)creates the file on first write of the launch, truncating
    // whatever the previous launch left.
    private func write(_ s: String) {
        let fm = FileManager.default
        if handle == nil {
            try? fm.createDirectory(at: url.deletingLastPathComponent(),
                                    withIntermediateDirectories: true)
            fm.createFile(atPath: url.path, contents: Data())
            handle = try? FileHandle(forWritingTo: url)
        }
        if let handle {
            try? handle.write(contentsOf: Data(s.utf8))
        }
    }
}

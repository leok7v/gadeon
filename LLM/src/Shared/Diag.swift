import Foundation
import os

// Unified diagnostics sink shared by App and LLM. Every report() lands three
// places: stderr (Xcode console), the unified log (Console.app), and an appended
// per-run file under Caches/<bundle>/diag.log/, tagged with a timestamp and the
// caller's file:line. One file per launch, pruned after ~24h, so a whole day of
// runs -- iOS and macOS alike -- reads as one folder with no copy/paste.
// rollingFile is reused for the transcript folder.

public final class Diag: @unchecked Sendable {

    public static let shared = Diag()

    private static let retention: TimeInterval = 24 * 3600

    private let log = Logger(subsystem: "io.github.leok7v.gadeon",
                             category: "diag")
    private let lock = NSLock()
    private let handle: FileHandle?
    private let stamp: DateFormatter
    // Where this run is writing, so the debug view can offer the file itself
    // instead of a path a phone user cannot act on anyway.
    public let path: String

    private init() {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "HH:mm:ss.SSS"
        stamp = f
        let url = Diag.startRun(folder: "diag.log")
        FileManager.default.createFile(atPath: url.path, contents: Data())
        handle = try? FileHandle(forWritingTo: url)
        path = url.path
    }

    // A memory-footprint reporter, filled in by the App layer (which owns the
    // platform split for os_proc_available_memory). LLM code cannot reach
    // App, so it asks through here -- the same shape MarkdownDiag already
    // uses. nil in the CLI and in tests, where there is nothing to report to.
    nonisolated(unsafe)
    public static var memory: (@Sendable (String) -> Void)?

    // The fine-grained twin of `memory`: reports that draw a CURVE (one per
    // prefill chunk, one per downloaded file) rather than mark a moment.
    // Split at the CALL SITE, so the character of a report is declared by the
    // code that knows it instead of guessed from its wording by whoever wires
    // the hook. Left nil when the channel is off, which costs nothing.
    nonisolated(unsafe)
    public static var memoryDetail: (@Sendable (String) -> Void)?

    // The caller's file:line comes free via the default arguments, captured at
    // the call site (a thin forwarder passes them through).
    public func report(_ s: String, file: String = #fileID, line: Int = #line) {
        let name = file.split(separator: "/").last.map(String.init) ?? file
        let out = "\(stamp.string(from: Date())) \(name):\(line) \(s)"
        log.notice("\(s, privacy: .public)")
        // write(contentsOf:), not write(Data:) -- the latter raises an
        // uncatchable ObjC exception on iOS.
        try? FileHandle.standardError.write(contentsOf: Data("\(out)\n".utf8))
        lock.lock()
        try? handle?.write(contentsOf: Data("\(out)\n".utf8))
        lock.unlock()
    }

    // The current run always writes to <folder>/current.txt -- a STABLE name a
    // devicectl file-copy can pull (directory copies from the device are
    // unreliable). The PREVIOUS run's current.txt is archived to a timestamped
    // file, and archives older than the retention window are pruned, so a day of
    // runs is preserved while the latest is always at a known path.
    public static func startRun(folder: String) -> URL {
        let fm = FileManager.default
        let dir = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? "app")
            .appendingPathComponent(folder, isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let current = dir.appendingPathComponent("current.txt")
        if fm.fileExists(atPath: current.path) {
            let mtime = (try? current.resourceValues(
                forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? Date()
            let f = DateFormatter()
            f.locale = Locale(identifier: "en_US_POSIX")
            f.dateFormat = "yyyy-MM-dd_HH-mm-ss"
            let archive = dir.appendingPathComponent(
                "\(f.string(from: mtime)).txt")
            try? fm.moveItem(at: current, to: archive)
        }
        prune(dir)
        return current
    }

    // Drop archived runs older than the retention window; current.txt is left.
    private static func prune(_ dir: URL) {
        let fm = FileManager.default
        let now = Date()
        let files = (try? fm.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        for f in files where f.lastPathComponent != "current.txt" {
            let mtime = (try? f.resourceValues(
                forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? now
            if now.timeIntervalSince(mtime) > retention {
                try? fm.removeItem(at: f)
            }
        }
    }
}

// Pulling these logs off a connected iPhone WITHOUT copy/paste (why the file
// name is a STABLE current.txt: `devicectl` file copies work, directory copies
// do not). This sink writes diag.log/; the session trace writes transcript.log/
// -- parallel folders under the app's Caches:
//   Library/Caches/io.github.leok7v.gadeon/diag.log/current.txt
//   Library/Caches/io.github.leok7v.gadeon/transcript.log/current.txt
// Previous runs archive beside current.txt as <yyyy-MM-dd_HH-mm-ss>.txt, pruned
// after the 24h retention above.
//
//   # 1. find the device (State == connected; the UDID rotates, never hardcode)
//   xcrun devicectl list devices
//
//   # 2. pull the current run (a FILE copy -- reliable)
//   xcrun devicectl device copy from --device <UDID> \
//     --domain-type appDataContainer \
//     --domain-identifier io.github.leok7v.gadeon \
//     --source "Library/Caches/io.github.leok7v.gadeon/diag.log/current.txt" \
//     --destination /tmp/current.txt
//
//   # 3. enumerate the archives, then copy one by its exact name (step 2 form)
//   xcrun devicectl device info files --device <UDID> \
//     --domain-type appDataContainer \
//     --domain-identifier io.github.leok7v.gadeon \
//     --subdirectory "Library/Caches/io.github.leok7v.gadeon/diag.log"
//
// The app-GROUP container (if one is ever reintroduced) is reached with
// --domain-type appGroupDataContainer --domain-identifier <group-id> instead.

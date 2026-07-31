import CryptoKit
import Foundation

// Fetches a compiled CoreML model set from the Hugging Face Hub with no
// third-party dependency: two plain HTTPS endpoints, URLSession +
// JSONSerialization for the rest.
//
//     list   GET {host}/api/models/{repo}/tree/{sha}?recursive=true
//     fetch  GET {host}/{repo}/resolve/{sha}/{path}      (302 -> CDN)
//
// Public repo: no auth. The revision is pinned to a commit sha BEFORE the
// first byte: pulling file-by-file from "main" could straddle a push and mix
// files from two commits (loads cleanly, generates garbage). Files download
// IN PLACE at their final {sha}/ paths: each blob streams to a temp file, is
// digest-checked against the tree API's oid, and only then moves (atomic
// rename) to its destination, so a landed file is always whole and verified.
// Completeness is the .complete sentinel, NEVER file presence --
// ModelCatalog.isComplete gates on it, so a partial tree is never treated as
// a set. An interrupted download RESUMES: survivors that re-verify are kept,
// only the rest re-download. In-place (not a staged .part tree renamed at the
// end) is also what lets the ANE compile cache be primed per-program while
// later files still stream -- the e5 cache keys on the FINAL path.
//
// Resume within a launch uses URLError.downloadTaskResumeData; surviving app
// suspension needs a background URLSession (wire in if a cold download
// matters).
public enum HubError: Error {
    case http(Int, String)          // status, url
    case malformed(String)          // endpoint that did not parse
    case notFound(String)           // no files under this prefix
    case digest(String)             // oid mismatch on this path
}

public struct HubFetch: Sendable {
    // Progress is BYTE-based (done/total are bytes, not a file count), so a
    // multi-file set with wildly unequal sizes -- and a single huge file --
    // report an honest fraction and a speed-derived ETA. `file` is the file in
    // flight, for display.
    public struct Status: Sendable {
        public let file: String
        public let done: Int64        // bytes downloaded so far
        public let total: Int64       // total bytes to download
    }

    struct Entry: Sendable {
        let path: String
        let size: Int64
        let oid: String
        let lfs: Bool
    }

    static let host = "https://huggingface.co"
    static let sentinel = ".complete"
    static let retries = 5
    static let chunk = 1 << 20

    // Download everything under `prefix` at `revision` into `dest`, return the
    // set dir (dest/{sha}, or dest/{sha}/{prefix}). Already staged at that
    // commit -> no-op, so cheap to call every launch. Prunes sibling {oldsha}
    // dirs after install; excludeFromBackup keeps the set out of
    // iCloud/backup.
    public static func fetch(
        repo: String,
        prefix: String = "",
        into dest: URL,
        revision: String = "main",
        files: [String]? = nil,
        excludeFromBackup: Bool = false,
        report: @escaping @Sendable (Status) -> Void = { _ in }
    ) async throws -> URL {
        let fm = FileManager.default
        let sha = try await commit(repo, revision)
        let root = dest.appendingPathComponent(sha)
        let mark = root.appendingPathComponent(sentinel)
        if !fm.fileExists(atPath: mark.path) {
            let all = try await tree(repo, sha)
            // `files` is an exact allowlist (self-contained GGUF); else `prefix`
            // selects a subtree; else the whole repo.
            let want: [Entry]
            if let files {
                want = all.filter { e in files.contains(e.path) }
            } else if prefix.isEmpty {
                want = all
            } else {
                want = all.filter { e in e.path.hasPrefix(prefix + "/") }
            }
            if want.isEmpty {
                throw HubError.notFound(files?.joined(separator: ",") ?? prefix)
            }
            try await install(repo, sha, want, root, report)
            fm.createFile(atPath: mark.path, contents: nil)
            prune(dest, keep: sha)
        }
        if excludeFromBackup {
            exclude(fromBackup: root)
        }
        return prefix.isEmpty ? root : root.appendingPathComponent(prefix)
    }

    // Remove sibling {oldsha} staging trees (and any stale .part) so only the
    // freshly verified commit survives in `dest`.
    static func prune(_ dest: URL, keep sha: String) {
        let fm = FileManager.default
        let kids = (try? fm.contentsOfDirectory(at: dest,
            includingPropertiesForKeys: nil)) ?? []
        for k in kids where k.lastPathComponent != sha {
            try? fm.removeItem(at: k)
        }
    }

    // Tag the set dir excluded-from-backup. Application Support keeps it from
    // being purged; the flag keeps it out of iCloud/backup. Best-effort.
    static func exclude(fromBackup url: URL) {
        var u = url
        var v = URLResourceValues()
        v.isExcludedFromBackup = true
        try? u.setResourceValues(v)
    }

    static func commit(_ repo: String, _ rev: String) async throws -> String {
        let url = "\(host)/api/models/\(repo)/revision/\(esc(rev))"
        let (data, _) = try await page(url)
        let obj = try? JSONSerialization.jsonObject(with: data)
        let dict = obj as? [String: Any]
        return try need(dict?["sha"] as? String, url)
    }

    // The tree endpoint pages; follow rel="next" rather than assume one call
    // covers the whole set.
    static func tree(_ repo: String, _ sha: String) async throws -> [Entry] {
        var next: String? = "\(host)/api/models/\(repo)/tree/\(sha)"
            + "?recursive=true"
        var out: [Entry] = []
        while let url = next {
            let (data, link) = try await page(url)
            let obj = try? JSONSerialization.jsonObject(with: data)
            let arr = try need(obj as? [Any], url)
            for any in arr {
                if let e = entry(any) {
                    out.append(e)
                }
            }
            next = nextLink(link)
        }
        return out
    }

    // Directory nodes carry no blob. LFS nodes carry lfs.oid (sha256 of the
    // content); plain nodes carry the git blob oid, which is sha1 over
    // "blob <size>\0" + content -- not over the content alone.
    static func entry(_ any: Any) -> Entry? {
        var out: Entry? = nil
        if let d = any as? [String: Any],
           d["type"] as? String == "file",
           let path = d["path"] as? String {
            let size = (d["size"] as? NSNumber)?.int64Value ?? 0
            let lfs = d["lfs"] as? [String: Any]
            let oid = (lfs?["oid"] as? String) ?? (d["oid"] as? String)
            if let oid {
                out = Entry(path: path, size: size, oid: oid, lfs: lfs != nil)
            }
        }
        return out
    }

    static func install(
        _ repo: String,
        _ sha: String,
        _ want: [Entry],
        _ root: URL,
        _ report: @escaping @Sendable (Status) -> Void
    ) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        // Byte-based progress: total is the sum of file sizes; the meter adds
        // each in-flight file's live bytes to the completed base and reports
        // (throttled) so the ETA is derived from real throughput.
        let bytesTotal = want.reduce(Int64(0)) { $0 + $1.size }
        let meter = ByteMeter(total: bytesTotal, report: report)
        // ONE session for the whole set. A session per file gave URLSession a
        // fresh connection pool for each of the set's ~84 blobs, so the CDN
        // saw ~84 TLS/QUIC handshakes instead of one warm pool, and each
        // abandoned HTTP/3 connection made CFNetwork log
        // "nw_connection_copy_protocol_metadata on unconnected nw_connection"
        // from its own teardown path.
        let pump = Pump()
        defer { pump.done() }
        for e in want {
            let dst = root.appendingPathComponent(e.path)
            try fm.createDirectory(at: dst.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            meter.setFile(e.path)
            // A survivor of an interrupted download that still matches its
            // oid is kept -- resume re-verifies instead of restarting.
            let kept = fm.fileExists(atPath: dst.path)
                && (try? verify(dst, e)) != nil
            if !kept {
                try await pull(repo, sha, e, dst, pump) { written in
                    meter.live(written)
                }
            }
            meter.complete(e.size)
        }
    }

    static func pull(
        _ repo: String, _ sha: String, _ e: Entry, _ dst: URL, _ pump: Pump,
        _ onBytes: @escaping @Sendable (Int64) -> Void
    ) async throws {
        let fm = FileManager.default
        let raw = "\(host)/\(repo)/resolve/\(sha)/\(esc(e.path))"
        let url = try need(URL(string: raw), raw)
        var attempt = 0
        var resume: Data? = nil
        var verified: URL? = nil
        var last: Error? = nil
        while attempt < retries && verified == nil {
            do {
                let tmp = try await pump.body(url, resume, onBytes)
                do {
                    try verify(tmp, e)
                    verified = tmp
                } catch {
                    // One corrupted CDN read must not fail the whole fetch:
                    // throw the blob away and re-pull from scratch (no
                    // resume -- the bytes are bad, not missing).
                    try? fm.removeItem(at: tmp)
                    resume = nil
                    last = error
                }
            } catch let err as URLError {
                resume = err.downloadTaskResumeData
                last = err
            } catch {
                last = error
            }
            attempt += 1
        }
        let file = try need(verified, e.path, last)
        try? fm.removeItem(at: dst)
        try fm.moveItem(at: file, to: dst)
    }

    static func verify(_ file: URL, _ e: Entry) throws {
        let got = e.lfs ? try sha256(file) : try blobSHA1(file)
        if got != e.oid {
            throw HubError.digest(e.path)
        }
    }

    static func sha256(_ file: URL) throws -> String {
        let h = try FileHandle(forReadingFrom: file)
        var d = SHA256()
        var part = try h.read(upToCount: chunk)
        while let c = part, !c.isEmpty {
            d.update(data: c)
            part = try h.read(upToCount: chunk)
        }
        try h.close()
        return hex(d.finalize())
    }

    static func blobSHA1(_ file: URL) throws -> String {
        let fm = FileManager.default
        let attrs = try fm.attributesOfItem(atPath: file.path)
        let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        let h = try FileHandle(forReadingFrom: file)
        var d = Insecure.SHA1()
        d.update(data: Data("blob \(size)\0".utf8))
        var part = try h.read(upToCount: chunk)
        while let c = part, !c.isEmpty {
            d.update(data: c)
            part = try h.read(upToCount: chunk)
        }
        try h.close()
        return hex(d.finalize())
    }

    static func page(_ url: String) async throws -> (Data, String?) {
        let req = URLRequest(url: try need(URL(string: url), url))
        let (data, resp) = try await URLSession.shared.data(for: req)
        let http = resp as? HTTPURLResponse
        let code = http?.statusCode ?? 0
        if code != 200 {
            throw HubError.http(code, url)
        }
        return (data, http?.value(forHTTPHeaderField: "Link"))
    }

    // RFC 5988: Link: <https://...>; rel="next"
    static func nextLink(_ header: String?) -> String? {
        var out: String? = nil
        if let h = header, h.contains("rel=\"next\"") {
            let open = h.split(separator: "<", maxSplits: 1)
            if open.count == 2 {
                let shut = open[1].split(separator: ">", maxSplits: 1)
                if !shut.isEmpty {
                    out = String(shut[0])
                }
            }
        }
        return out
    }

    static func need<T>(_ v: T?, _ what: String,
                        _ cause: Error? = nil) throws -> T {
        if v == nil {
            throw cause ?? HubError.malformed(what)
        }
        return v!
    }

    static func hex<D: Sequence>(_ d: D) -> String
        where D.Element == UInt8 {
        d.map { b in String(format: "%02x", b) }.joined()
    }

    static func esc(_ s: String) -> String {
        s.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? s
    }
}

// Byte-progress meter for a multi-file fetch: the completed-file base plus the
// in-flight file's live bytes, throttled to ~1 MB steps so the UI is not
// flooded. Delegate callbacks are serial but a lock keeps it Sendable-clean.
private final class ByteMeter: @unchecked Sendable {
    private let total: Int64
    private let report: @Sendable (HubFetch.Status) -> Void
    private let lock = NSLock()
    private var base: Int64 = 0
    private var last: Int64 = 0
    private var file = ""

    init(total: Int64,
         report: @escaping @Sendable (HubFetch.Status) -> Void) {
        self.total = total
        self.report = report
    }

    func setFile(_ f: String) { lock.lock(); file = f; lock.unlock() }

    // The in-flight file's cumulative bytes; emits at ~1 MB granularity.
    func live(_ fileBytes: Int64) {
        lock.lock()
        let now = base + fileBytes
        let emit = now - last >= 1 << 20
        if emit { last = now }
        let f = file
        lock.unlock()
        if emit { report(HubFetch.Status(file: f, done: now, total: total)) }
    }

    func complete(_ size: Int64) {
        lock.lock()
        base += size
        last = base
        let n = base, f = file
        lock.unlock()
        report(HubFetch.Status(file: f, done: n, total: total))
    }
}

// One URLSession for a whole set install, plus the bridge from its download
// callbacks to async/await. The session outlives every individual blob so the
// CDN connection pool stays warm across the set, which means the delegate is
// shared too: each in-flight task's continuation and byte sink are keyed by
// taskIdentifier rather than held as one pair of fields. didFinish and
// didComplete can both fire for the same task, so a waiter is REMOVED as it
// is taken and the continuation is consumed exactly once.
final class Pump: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private struct Waiter {
        let cont: CheckedContinuation<URL, Error>
        let onBytes: @Sendable (Int64) -> Void
    }

    private let lock = NSLock()
    private var waiters: [Int: Waiter] = [:]
    private var live: URLSession? = nil

    // Built on first use, not in init: the session retains its delegate, so
    // handing `self` to URLSession from inside init would publish a
    // half-constructed object.
    private func session() -> URLSession {
        lock.lock()
        let s = live ?? URLSession(configuration: .default, delegate: self,
                                   delegateQueue: nil)
        live = s
        lock.unlock()
        return s
    }

    // Ends the session so it releases its delegate (self) and drops the
    // connection pool. The retain cycle session <-> delegate lives until here
    // by design, so this must run on every path out of the install.
    func done() {
        lock.lock()
        let s = live
        live = nil
        lock.unlock()
        s?.finishTasksAndInvalidate()
    }

    private func take(_ id: Int) -> Waiter? {
        lock.lock()
        let w = waiters.removeValue(forKey: id)
        lock.unlock()
        return w
    }

    // Streams straight to a temp file and follows the 302 to the CDN (a
    // multi-GB blob never lands in memory), reporting live bytesWritten for
    // the progress meter and preserving resume-on-failure (the error carries
    // downloadTaskResumeData for the retry loop).
    func body(_ url: URL, _ resume: Data?,
              _ onBytes: @escaping @Sendable (Int64) -> Void)
        async throws -> URL {
        try await withCheckedThrowingContinuation { cont in
            let s = session()
            let task = resume.map { s.downloadTask(withResumeData: $0) }
                ?? s.downloadTask(with: url)
            lock.lock()
            waiters[task.taskIdentifier] = Waiter(cont: cont,
                                                  onBytes: onBytes)
            lock.unlock()
            task.resume()
        }
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        let dst = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                ProcessInfo.processInfo.globallyUniqueString)
        let code = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0
        let w = take(downloadTask.taskIdentifier)
        if code != 200 && code != 206 {
            w?.cont.resume(throwing: HubError.http(
                code, downloadTask.originalRequest?.url?.absoluteString ?? ""))
        } else {
            do {
                try FileManager.default.moveItem(at: location, to: dst)
                w?.cont.resume(returning: dst)
            } catch {
                w?.cont.resume(throwing: error)
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let error {
            take(task.taskIdentifier)?.cont.resume(throwing: error)
        }
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        lock.lock()
        let w = waiters[downloadTask.taskIdentifier]
        lock.unlock()
        w?.onBytes(totalBytesWritten)
    }
}

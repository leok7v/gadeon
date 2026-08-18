import Darwin
import Foundation
import LLM

final class AneCache: @unchecked Sendable {

    static let shared = AneCache()

    enum CacheMode: String { case off, hardlink }
    static let cacheMode: CacheMode = {
        ProcessInfo.processInfo.environment["ANE_CACHE_MODE"] == "disable"
            ? .off : .hardlink
    }()
    private static let minOrphanAge: TimeInterval = 3600

    private func report(_ s: String, file: String = #fileID, line: Int = #line) {
        Diag.shared.report(s, file: file, line: line)
    }

    private let lock = NSLock()
    private var launched = false
    private var preDownload: Set<String>?
    private var migrated: Bool?

    private struct Claims: Codable {
        var sets: [String: [String]] = [:]
    }

    private struct Seen: Codable {
        var first: [String: Double] = [:]
    }

    func buildBegan() -> Set<String> {
        lock.lock()
        let first = !launched
        launched = true
        let early = preDownload
        preDownload = nil
        lock.unlock()
        if first { launchSurvey() }
        return early ?? entries()
    }

    func containerMigrated() -> Bool {
        lock.lock()
        if migrated == nil {
            let url = stateDir().appendingPathComponent("last-root.txt")
            let current = cacheRoot().path
            let previous = try? String(contentsOf: url, encoding: .utf8)
            migrated = previous != nil && previous != current
            try? current.write(to: url, atomically: true, encoding: .utf8)
        }
        let out = migrated ?? false
        lock.unlock()
        return out
    }

    func downloadBegan() {
        lock.lock()
        let first = !launched
        launched = true
        lock.unlock()
        if first { launchSurvey() }
        let now = entries()
        lock.lock()
        preDownload = now
        lock.unlock()
    }

    func buildEnded(setDir: URL, before: Set<String>) {
        let key = setKey(setDir)
        let after = entries()
        let fresh = after.subtracting(before)
        if let build = osBuild() {
            lock.lock()
            var claims = loadClaims(build)
            let mine = Set(claims.sets[key] ?? [])
            let elsewhere = Set(claims.sets
                .filter { claim in claim.key != key }
                .values.flatMap { names in names })
            let unclaimed = after.subtracting(elsewhere)
            let merged = mine.intersection(after).union(fresh)
                .union(unclaimed)
            claims.sets[key] = merged.sorted()
            saveClaims(claims, build)
            lock.unlock()
            report("claim \(key): +\(fresh.count) fresh, \(merged.count) "
                + "claimed of \(after.count) live")
            if AneCache.cacheMode == .hardlink { shadowLive(build) }
        }
    }

    private func launchSurvey() {
        let root = cacheRoot()
        report("mode=\(AneCache.cacheMode.rawValue) root \(root.path)")
        report("state dir \(stateDir().path)")
        if AneCache.cacheMode == .hardlink {
            report("shadow root \(shadowRoot().path)")
        }
        if let build = osBuild() {
            let dir = root.appendingPathComponent(build)
            if isOS && containerMigrated() {
                let mb = directoryBytes(dir) >> 20
                try? FileManager.default.removeItem(at: dir)
                try? FileManager.default.removeItem(at: shadowDir(build))
                try? FileManager.default.removeItem(at: claimsURL(build))
                report("migration: dropped \(mb) MB dead cache + shadow "
                    + "(re-keyed to a new container path)")
            }
            let names = entries(dir)
            let fresh = updateSeen(names, build)
            let mb = directoryBytes(dir) >> 20
            report("""
                survey \(build): \(names.count) entries \(mb) MB, \
                \(fresh) new since last launch
                """)
            if AneCache.cacheMode == .hardlink {
                let sdir = shadowDir(build)
                let sn = entries(sdir).count
                report("shadow \(build): \(sn) entries "
                    + "\(directoryBytes(sdir) >> 20) MB at \(sdir.path)")
            }
            pruneDeadClaims(build)
            collect(dir, build, names.union(entries(shadowDir(build))))
            if AneCache.cacheMode == .hardlink { pruneStaleShadows(build) }
        } else {
            report("OS build not parsed; cache keeper IDLE")
        }
    }

    private func pruneDeadClaims(_ build: String) {
        let fm = FileManager.default
        let store = Bundle.modelStore()
        let installed = (try? fm.contentsOfDirectory(atPath: store.path)) ?? []
        if !installed.isEmpty {
            lock.lock()
            var claims = loadClaims(build)
            let gone = claims.sets.keys.filter { key in
                !fm.fileExists(
                    atPath: store.appendingPathComponent(key).path)
            }
            for key in gone {
                claims.sets.removeValue(forKey: key)
            }
            if !gone.isEmpty {
                saveClaims(claims, build)
            }
            lock.unlock()
            if !gone.isEmpty {
                report("claims: dropped \(gone.count) for uninstalled set(s) "
                    + gone.sorted().joined(separator: ", "))
            }
        }
    }

    private func pruneStaleShadows(_ current: String) {
        let fm = FileManager.default
        let kids = (try? fm.contentsOfDirectory(
            at: shadowRoot(), includingPropertiesForKeys: nil)) ?? []
        for kid in kids where kid.lastPathComponent.hasPrefix("shadow-")
            && kid.lastPathComponent != "shadow-\(current)" {
            try? fm.removeItem(at: kid)
        }
    }

    private func collect(_ dir: URL, _ build: String, _ names: Set<String>) {
        lock.lock()
        let claims = loadClaims(build)
        let seen = loadSeen(build)
        lock.unlock()
        let orphans = names.subtracting(
            Set(claims.sets.values.flatMap { names in names })).count
        if let pending = modelAwaitingClaim(claims) {
            report("GC held: no claim yet for \(pending); "
                + "\(orphans) unclaimed of \(names.count)")
        } else {
            let claimed = Set(claims.sets.values.flatMap { names in names })
            let fm = FileManager.default
            let now = Date().timeIntervalSince1970
            var deleted = 0
            var bytes: Int64 = 0
            for name in names.subtracting(claimed) {
                let copies = both(name, build)
                let born = seen.first[name] ?? modified(copies)
                if now - born > AneCache.minOrphanAge,
                   now - modified(copies) > AneCache.minOrphanAge {
                    bytes += entryBytes(copies)
                    for copy in copies { try? fm.removeItem(at: copy) }
                    deleted += 1
                }
            }
            report("GC: deleted \(deleted) of \(orphans) unclaimed entries, "
                + "\(bytes >> 20) MB freed")
        }
    }

    private func modelAwaitingClaim(_ claims: Claims) -> String? {
        var result: String? = nil
        let store = Bundle.modelStore()
        for name in Models.all
        where result == nil && !ModelCatalog.isGguf(name) {
            if let set = ModelCatalog.localSet(name, in: store),
               ModelCatalog.isComplete(set),
               (claims.sets[setKey(set)] ?? []).isEmpty {
                result = name
            }
        }
        return result
    }

    private func shadowLive(_ build: String) {
        let live = cacheRoot().appendingPathComponent(build)
        let shadow = shadowDir(build)
        let t0 = Date()
        let linked = mirror(from: live, to: shadow, relinkStale: true,
                            label: "shadow")
        excludeFromBackup(shadow)
        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        let shadowMB = directoryBytes(shadow) >> 20
        let liveMB = directoryBytes(live) >> 20
        report("shadow saved: +\(linked) file(s) in \(ms) ms "
            + "(shadow \(shadowMB) MB vs live \(liveMB) MB) "
            + "from \(live.path) to \(shadow.path)")
        sampleShadowLink(live: live, shadow: shadow)
    }

    private func sampleShadowLink(live: URL, shadow: URL) {
        let fm = FileManager.default
        var sample: URL? = nil
        let walk = fm.enumerator(at: shadow,
            includingPropertiesForKeys: [.isRegularFileKey])
        for case let url as URL in walk ?? .init() where sample == nil {
            let regular = (try? url.resourceValues(
                forKeys: [.isRegularFileKey]))?.isRegularFile == true
            if regular { sample = url }
        }
        if let file = sample {
            let rel = String(file.standardizedFileURL.path
                .dropFirst(shadow.standardizedFileURL.path.count))
            let liveFile = URL(fileURLWithPath: live.path + rel)
            report("shadow sample shadow=\(file.path) inode \(inode(file)) "
                + "\(fileBytes(file))B | live=\(liveFile.path) inode "
                + "\(inode(liveFile)) \(fileBytes(liveFile))B")
        }
    }

    private func fileBytes(_ url: URL) -> Int {
        (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
    }

    private func both(_ name: String, _ build: String) -> [URL] {
        [cacheRoot().appendingPathComponent(build)
            .appendingPathComponent(name),
         shadowDir(build).appendingPathComponent(name)]
    }

    private func entryBytes(_ copies: [URL]) -> Int64 {
        let fm = FileManager.default
        var out: Int64 = 0
        for url in copies where out == 0 && fm.fileExists(atPath: url.path) {
            out = directoryBytes(url)
        }
        return out
    }

    private func modified(_ copies: [URL]) -> TimeInterval {
        let fm = FileManager.default
        var newest: TimeInterval = 0
        for url in copies {
            newest = max(newest, (try? fm.attributesOfItem(atPath: url.path)[
                .modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
        }
        return newest
    }

    func releaseSet(_ setDir: URL) {
        if let build = osBuild() {
            let key = setKey(setDir)
            lock.lock()
            var claims = loadClaims(build)
            let mine = Set(claims.sets.removeValue(forKey: key) ?? [])
            saveClaims(claims, build)
            let held = Set(claims.sets.values.flatMap { names in names })
            lock.unlock()
            let fm = FileManager.default
            let dropped = mine.subtracting(held)
            var bytes: Int64 = 0
            for name in dropped {
                let copies = both(name, build)
                bytes += entryBytes(copies)
                for copy in copies { try? fm.removeItem(at: copy) }
            }
            report("release \(key): dropped \(dropped.count) of "
                + "\(mine.count) claimed, \(bytes >> 20) MB freed")
        }
    }

    func purgeSet(_ setDir: URL) {
        if let build = osBuild() {
            let key = setKey(setDir)
            lock.lock()
            let entries = loadClaims(build).sets[key] ?? []
            lock.unlock()
            let root = cacheRoot().appendingPathComponent(build)
            let shadow = shadowDir(build)
            let fm = FileManager.default
            if entries.isEmpty {
                try? fm.removeItem(at: root)
                try? fm.removeItem(at: shadow)
            } else {
                for entry in entries {
                    try? fm.removeItem(at: root.appendingPathComponent(entry))
                    try? fm.removeItem(
                        at: shadow.appendingPathComponent(entry))
                }
            }
            report("""
                purge: dropped \(entries.isEmpty ? "all" : "\(entries.count)") \
                cache+shadow entries for \(key) (load failed; will recompile)
                """)
        }
    }

    func restoreFromShadow() {
        if AneCache.cacheMode == .hardlink, let build = osBuild() {
            relinkIfCold(cacheRoot().appendingPathComponent(build), build)
        }
    }

    private func relinkIfCold(_ live: URL, _ build: String) {
        let shadow = shadowDir(build)
        if FileManager.default.fileExists(atPath: shadow.path) {
            let shadowMB = directoryBytes(shadow) >> 20
            let restored = mirror(from: shadow, to: live, label: "relink")
            report("relink: shadow \(shadowMB) MB, restored \(restored) "
                + "file(s) from \(shadow.path) to \(live.path)")
        }
    }

    private func mirror(from: URL, to: URL, relinkStale: Bool = false,
                        label: String = "") -> Int {
        let fm = FileManager.default
        var count = 0
        // Standardize both, or the slice mangles and links land beside `to`.
        let base = from.standardizedFileURL.path
        let walk = fm.enumerator(at: from,
            includingPropertiesForKeys: [.isRegularFileKey])
        for case let src as URL in walk ?? .init() {
            let regular = (try? src.resourceValues(
                forKeys: [.isRegularFileKey]))?.isRegularFile == true
            let rel = String(src.standardizedFileURL.path.dropFirst(base.count))
            let dst = URL(fileURLWithPath: to.path + rel)
            let present = fm.fileExists(atPath: dst.path)
            let stale = relinkStale && present && inode(src) != inode(dst)
            let truncated = !relinkStale && present
                && fileBytes(dst) < fileBytes(src)
            if regular, !present || stale || truncated {
                let why = !present ? "missing"
                    : (truncated ? "truncated" : "stale")
                if stale || truncated { try? fm.removeItem(at: dst) }
                try? fm.createDirectory(at: dst.deletingLastPathComponent(),
                                        withIntermediateDirectories: true)
                // link(2): a copy or clone would not keep inode and blocks.
                if link(src.path, dst.path) == 0 {
                    count += 1
                    if !label.isEmpty {
                        let si = inode(src), di = inode(dst)
                        report("\(label) \(why) \(rel): src inode \(si) "
                            + "\(fileBytes(src))B -> dst inode \(di) "
                            + "\(fileBytes(dst))B shared=\(si == di)")
                    }
                }
            }
        }
        return count
    }

    private func inode(_ url: URL) -> UInt64 {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
    }

    private func cacheRoot() -> URL {
        let fm = FileManager.default
        let caches = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let bundled = caches
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? "app")
            .appendingPathComponent("com.apple.e5rt.e5bundlecache")
        let bare = caches.appendingPathComponent("com.apple.e5rt.e5bundlecache")
        let useBare = !fm.fileExists(atPath: bundled.path)
            && fm.fileExists(atPath: bare.path)
        return useBare ? bare : bundled
    }

    private func osBuild() -> String? {
        let s = ProcessInfo.processInfo.operatingSystemVersionString
        var result: String? = nil
        if let at = s.range(of: "Build "),
           let close = s[at.upperBound...].firstIndex(of: ")") {
            result = String(s[at.upperBound ..< close])
        }
        return result
    }

    private func stateDir() -> URL {
        let fm = FileManager.default
        let base = fm.urls(for: .applicationSupportDirectory,
                           in: .userDomainMask)[0]
            .appendingPathComponent("anecache", isDirectory: true)
        try? fm.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private func shadowRoot() -> URL {
        stateDir().appendingPathComponent("anecache-shadow", isDirectory: true)
    }

    private func shadowDir(_ build: String) -> URL {
        shadowRoot().appendingPathComponent("shadow-\(build)",
                                            isDirectory: true)
    }

    private func setKey(_ setDir: URL) -> String {
        setDir.deletingLastPathComponent().lastPathComponent
            + "/" + setDir.lastPathComponent
    }

    private func entries() -> Set<String> {
        osBuild().map { build in
            entries(cacheRoot().appendingPathComponent(build))
        } ?? []
    }

    private func entries(_ dir: URL) -> Set<String> {
        Set((try? FileManager.default
            .contentsOfDirectory(atPath: dir.path)) ?? [])
    }

    private func updateSeen(_ names: Set<String>, _ build: String) -> Int {
        lock.lock()
        var seen = loadSeen(build)
        let prior = Set(seen.first.keys)
        let now = Date().timeIntervalSince1970
        var fresh = 0
        var next: [String: Double] = [:]
        for name in names {
            if let born = seen.first[name] {
                next[name] = born
            } else {
                next[name] = now
                fresh += 1
            }
        }
        seen.first = next
        saveSeen(seen, build)
        lock.unlock()
        let vanished = prior.subtracting(names).sorted()
        if !vanished.isEmpty {
            report("survey: \(vanished.count) entries VANISHED since last "
                + "launch (evicted): \(vanished.joined(separator: ", "))")
        }
        return fresh
    }

    private func claimsURL(_ build: String) -> URL {
        stateDir().appendingPathComponent("claims-\(build).json")
    }

    private func seenURL(_ build: String) -> URL {
        stateDir().appendingPathComponent("seen-\(build).json")
    }

    private func loadClaims(_ build: String) -> Claims {
        (try? JSONDecoder().decode(
            Claims.self, from: Data(contentsOf: claimsURL(build)))) ?? Claims()
    }

    private func saveClaims(_ claims: Claims, _ build: String) {
        try? JSONEncoder().encode(claims).write(to: claimsURL(build))
    }

    private func loadSeen(_ build: String) -> Seen {
        (try? JSONDecoder().decode(
            Seen.self, from: Data(contentsOf: seenURL(build)))) ?? Seen()
    }

    private func saveSeen(_ seen: Seen, _ build: String) {
        try? JSONEncoder().encode(seen).write(to: seenURL(build))
    }

    private func directoryBytes(_ dir: URL) -> Int64 {
        let fm = FileManager.default
        var total: Int64 = 0
        if let walk = fm.enumerator(at: dir,
                                    includingPropertiesForKeys: [.fileSizeKey]) {
            for case let url as URL in walk {
                let size = (try? url.resourceValues(
                    forKeys: [.fileSizeKey]))?.fileSize ?? 0
                total += Int64(size)
            }
        }
        return total
    }

    private func excludeFromBackup(_ url: URL) {
        var u = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? u.setResourceValues(values)
    }

}

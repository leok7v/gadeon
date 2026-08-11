import Darwin
import Foundation
import LLM
import os

final class AneCache: @unchecked Sendable {

    static let shared = AneCache()

    // The daemon writes the cache bundle APFS-PURGEABLE, so another Neural
    // Engine client's memory pressure can reclaim it. A warm reload is trusted
    // ONLY when the on-disk bundle is the exact inode+blocks the daemon wrote,
    // and a hardlink is the one operation preserving both (a copy, a clonefile
    // and an in-place content rewrite each recompile).
    //  .hardlink  (DEFAULT) shadow every fresh entry with a hardlink in
    //             non-purgeable app data on compile; relink from the shadow
    //             before the first load when a purge unlinked the cache entry.
    //  .off       no shadow (set ANE_CACHE_MODE=disable to opt out).
    enum CacheMode: String { case off, hardlink }
    static let cacheMode: CacheMode = {
        ProcessInfo.processInfo.environment["ANE_CACHE_MODE"] == "disable"
            ? .off : .hardlink
    }()
    // Entries younger than this are never deleted: a primer compile may
    // still be writing them.
    private static let minAge: TimeInterval = 3600

    private let log = Logger(subsystem: "io.github.leok7v.gadeon",
                             category: "anecache")

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

    // A warm hit leaves no filesystem trace, so ownership is tracked by
    // diffing the entry list around each set's build. Call on the build task
    // BEFORE the engine's first MLModel touch.

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

    // The e5 cache keys bind to a per-container identity nothing app-side can
    // pin, so a migration -- every Xcode install, every App Store update --
    // means the coming build recompiles.

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

    // The primer compiles the set's programs while it still streams, so the
    // ownership baseline has to be captured here rather than at buildBegan.

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

    // Everything that appeared since `before` is this set's claim; members
    // that vanished are dropped.

    func buildEnded(setDir: URL, before: Set<String>) {
        let key = setKey(setDir)
        let after = entries()
        let fresh = after.subtracting(before)
        if let build = osBuild() {
            lock.lock()
            var claims = loadClaims(build)
            // A completed build also owns every live entry no OTHER set
            // claims: `fresh` is empty on a warm launch, so anything compiled
            // by a run that DIED before buildEnded would stay unclaimed
            // forever and GC would delete it one minAge later. `old` and
            // `fresh` stay in the union: they hold a claim on an entry a
            // sibling set shares.
            let old = Set(claims.sets[key] ?? [])
            let others = Set(claims.sets.filter { $0.key != key }
                .values.flatMap { names in names })
            let merged = old.intersection(after).union(fresh)
                .union(after.subtracting(others))
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
            // iOS carries the old cache forward across an install, but its
            // entries are keyed to the old absolute path and the daemon
            // re-keys and recompiles regardless, so on a migration the cache
            // and its shadow are provably dead. macOS containers never
            // migrate and are never wiped here.
            if isOS && containerMigrated() {
                let mb = directoryBytes(dir) >> 20
                try? FileManager.default.removeItem(at: dir)
                try? FileManager.default.removeItem(at: shadowDir(build))
                // Application Support survives the install while the cache
                // re-keys, so the claim would survive as names that can never
                // exist again: non-empty, it opens the GC gate over a cache
                // being rebuilt and points purgeSet at phantom entries.
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
            collect(dir, build, names)
            if AneCache.cacheMode == .hardlink { pruneStaleShadows(build) }
        } else {
            report("OS build not parsed; cache keeper IDLE")
        }
    }

    // GC unions EVERY claim, installed or not, so a claim outliving its set
    // pins its entries permanently. Guarded on a NON-EMPTY store: a store that
    // is momentarily unreadable must prune nothing rather than unclaim the
    // whole cache.

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

    // The cache re-keys wholesale on an OS update, so a previous build's
    // shadow can only pin dead blocks.

    private func pruneStaleShadows(_ current: String) {
        let fm = FileManager.default
        let kids = (try? fm.contentsOfDirectory(
            at: shadowRoot(), includingPropertiesForKeys: nil)) ?? []
        for k in kids where k.lastPathComponent.hasPrefix("shadow-")
            && k.lastPathComponent != "shadow-\(current)" {
            try? fm.removeItem(at: k)
        }
    }

    private func collect(_ dir: URL, _ build: String, _ names: Set<String>) {
        lock.lock()
        let claims = loadClaims(build)
        let seen = loadSeen(build)
        lock.unlock()
        let orphans = names.subtracting(
            Set(claims.sets.values.flatMap { names in names })).count
        if let hold = gateClosed(claims) {
            report("GC held: \(hold); \(orphans) unclaimed of \(names.count)")
        } else {
            let claimed = Set(claims.sets.values.flatMap { names in names })
            let fm = FileManager.default
            let now = Date().timeIntervalSince1970
            var deleted = 0
            var bytes: Int64 = 0
            for name in names.subtracting(claimed) {
                let url = dir.appendingPathComponent(name)
                let born = seen.first[name] ?? now
                let mtime = (try? fm.attributesOfItem(atPath: url.path)[
                    .modificationDate] as? Date)?.timeIntervalSince1970 ?? now
                if now - born > AneCache.minAge,
                   now - mtime > AneCache.minAge {
                    bytes += directoryBytes(url)
                    try? fm.removeItem(at: url)
                    // The shadow must not outlive its entry: it would hold the
                    // blocks after the last real copy is gone.
                    try? fm.removeItem(
                        at: shadowDir(build).appendingPathComponent(name))
                    deleted += 1
                }
            }
            report("GC: deleted \(deleted) of \(orphans) unclaimed entries, "
                + "\(bytes >> 20) MB freed")
        }
    }

    // Every installed, complete, non-GGUF set must hold a non-empty claim
    // before GC may run: a warm build diffs to nothing and proves no
    // ownership, so its set holds the gate closed until it compiles cold once.

    private func gateClosed(_ claims: Claims) -> String? {
        var result: String? = nil
        let store = Bundle.modelStore()
        for name in Models.all
        where result == nil && !ModelCatalog.isGguf(name) {
            if let set = ModelCatalog.localSet(name, in: store),
               ModelCatalog.isComplete(set),
               (claims.sets[setKey(set)] ?? []).isEmpty {
                result = "no claim yet for \(name)"
            }
        }
        return result
    }

    // Hardlink every live entry not yet shadowed, AFTER the build finishes, so
    // a cache compiled before the mode was on is seeded in the same pass as
    // this build's fresh entries.

    private func shadowLive(_ build: String) {
        let live = cacheRoot().appendingPathComponent(build)
        let shadow = shadowDir(build)
        let t0 = Date()
        // refresh: a recompile writes a NEW inode at an existing path, and a
        // link-if-missing mirror would keep the stale one forever.
        let linked = mirror(from: live, to: shadow, refresh: true,
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
        if let s = sample {
            let rel = String(s.standardizedFileURL.path
                .dropFirst(shadow.standardizedFileURL.path.count))
            let liveFile = URL(fileURLWithPath: live.path + rel)
            report("shadow sample shadow=\(s.path) inode \(inode(s)) "
                + "\(fileBytes(s))B | live=\(liveFile.path) inode "
                + "\(inode(liveFile)) \(fileBytes(liveFile))B")
        }
    }

    private func fileBytes(_ url: URL) -> Int {
        (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
    }

    // The set's claim targets only its entries; an unclaimed set (never
    // seeded) has nothing to target and falls back to the whole OS-build dir.

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
                for e in entries {
                    try? fm.removeItem(at: root.appendingPathComponent(e))
                    try? fm.removeItem(at: shadow.appendingPathComponent(e))
                }
            }
            report("""
                purge: dropped \(entries.isEmpty ? "all" : "\(entries.count)") \
                cache+shadow entries for \(key) (load failed; will recompile)
                """)
        }
    }

    // Call from the build task AFTER a priming compile has made the daemon
    // create the cache dir (a fresh install has nothing to restore into until
    // then), and BEFORE the full model load.

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

    // Hardlink every regular file under `from` whose counterpart under `to` is
    // MISSING (or, with refresh, points at a DIFFERENT inode), creating parent
    // dirs; returns the count linked. refresh=false (restore direction) never
    // clobbers a present live file with a stale shadow; refresh=true (shadow
    // direction) heals a drifted inode.

    private func mirror(from: URL, to: URL, refresh: Bool = false,
                        label: String = "") -> Int {
        let fm = FileManager.default
        var count = 0
        // The enumerator yields symlink-STANDARDIZED child URLs (iOS resolves
        // /var -> /private/var) while `from` may be the raw form: slice both
        // in the same standardized form, or every relative path mangles and
        // the links scatter to a sibling of `to`.
        let base = from.standardizedFileURL.path
        let walk = fm.enumerator(at: from,
            includingPropertiesForKeys: [.isRegularFileKey])
        for case let src as URL in walk ?? .init() {
            let regular = (try? src.resourceValues(
                forKeys: [.isRegularFileKey]))?.isRegularFile == true
            let rel = String(src.standardizedFileURL.path.dropFirst(base.count))
            let dst = URL(fileURLWithPath: to.path + rel)
            let present = fm.fileExists(atPath: dst.path)
            let stale = refresh && present && inode(src) != inode(dst)
            // An in-place purge can leave the live file PRESENT but emptied
            // rather than unlinking it. A shared inode reads equal sizes, so
            // this fires only on real drift and never clobbers a
            // content-bearing live file, which is authoritative.
            let truncated = !refresh && present
                && fileBytes(dst) < fileBytes(src)
            if regular, !present || stale || truncated {
                let why = !present ? "missing"
                    : (truncated ? "truncated" : "stale")
                if stale || truncated { try? fm.removeItem(at: dst) }
                try? fm.createDirectory(at: dst.deletingLastPathComponent(),
                                        withIntermediateDirectories: true)
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

    // 0 when unreadable.

    private func inode(_ url: URL) -> UInt64 {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
    }

    // iOS and the sandboxed macOS app use the bundle-id-nested cache dir.

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

    // "23F84" from "Version 26.x (Build 23F84)"; nil disables the keeper.

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

    // Same volume as the e5 cache it hardlinks, so link(2) never crosses
    // volumes, and not purged like Caches.

    private func shadowRoot() -> URL {
        stateDir().appendingPathComponent("anecache-shadow", isDirectory: true)
    }

    private func shadowDir(_ build: String) -> URL {
        shadowRoot().appendingPathComponent("shadow-\(build)",
                                            isDirectory: true)
    }

    // Claim key: <model dir>/<revision leaf> -- stable across launches.

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

    // Vanished entries drop out, so the file cannot grow without bound.

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

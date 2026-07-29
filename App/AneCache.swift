import Darwin
import Foundation
import LLM
import os

// Keeper of the ANE compile cache (Library/Caches/<bundle-id>/
// com.apple.e5rt.e5bundlecache). The system never garbage-collects inside an
// OS-build namespace, so superseded model revisions and interrupted compiles
// accumulate gigabytes of dead compiled programs. The cache's keys are
// opaque -- model.milhash is not a hash of any file we hold, and warm hits
// leave no filesystem trace -- so ownership is tracked by OBSERVATION: diff
// the entry list around each set's build and persist the claim per set and
// OS build. GC deletes only unclaimed, aged entries, and only once every
// installed (non-GGUF) set holds a non-empty claim, so it can never delete
// state it has not accounted for. The launch survey log doubles as the
// instrumentation for the per-install cold-compile investigation: it prints
// the cache root (exposing the data-container UUID) and the entry delta
// since the previous launch.
final class AneCache: @unchecked Sendable {

    static let shared = AneCache()

    // Purge-survival strategy, selected by ANE_CACHE_MODE (set it in the Xcode
    // scheme to A/B without a rebuild). The daemon writes the bundle APFS-
    // PURGEABLE, so another Neural Engine client's memory pressure reclaims it
    // and the next launch recompiles (the 4GB-phone "Optimizing after Maps"
    // report). A warm reload is trusted ONLY when the on-disk bundle is the
    // exact inode+blocks the daemon wrote: a copy, a clonefile, and even an
    // in-place content rewrite each recompile (measured). A HARDLINK is the
    // one operation that preserves both inode number and blocks -- and it
    // warm-loads even after the cache dir entry is removed and relinked.
    //  .hardlink  (DEFAULT) shadow every fresh entry with a hardlink in
    //             non-purgeable app data on compile; relink from the shadow
    //             before the first load when a purge unlinked the cache entry.
    //             Defeats an UNLINK-style reclaim (the inode survives via the
    //             shadow); a data-block purge shares the shadow's blocks and
    //             degrades to a normal recompile, no worse than off.
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

    // The key cache events, made impossible to miss: os_log at notice with the
    // whole line public (no <private> redaction), AND a stderr write so it
    // lands in Xcode's debug console regardless of Console.app level filters.
    // stderr write is the iOS-safe contentsOf form (the Data: form raises an
    // uncatchable ObjC exception on iOS).
    private func report(_ s: String, file: String = #fileID, line: Int = #line) {
        Diag.shared.report(s, file: file, line: line)
    }

    private let lock = NSLock()
    private var launched = false
    // Entry list captured when a download begins, so programs the primer
    // compiles WHILE the set still streams land inside the ownership diff.
    private var preDownload: Set<String>?
    private var migrated: Bool?

    private struct Claims: Codable {
        var sets: [String: [String]] = [:]
    }

    private struct Seen: Codable {
        var first: [String: Double] = [:]
    }

    // Entry list before the set's compiles start; the once-per-launch survey
    // (restore experiment + GC + log) runs first. Call on the build task
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

    // Whether the data container moved since the previous launch -- true
    // after every Xcode install and every App Store update. The e5 cache
    // keys bind to a per-container identity nothing app-side can pin
    // (model paths, the cache location, and the executable path were all
    // stabilized and installs STILL compiled cold), so a migration means
    // the coming build recompiles: the app shows the honest Optimizing
    // screen instead of a stuck-looking "Loading".
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

    // A download is starting: capture the entry list now, before the primer
    // compiles any of the streaming set's programs.
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

    // The set finished its full build: everything that appeared since
    // `before` is this set's claim; members that vanished are dropped.
    func buildEnded(setDir: URL, before: Set<String>) {
        let key = setKey(setDir)
        let after = entries()
        let fresh = after.subtracting(before)
        if let build = osBuild() {
            lock.lock()
            var claims = loadClaims(build)
            // A cold build (fresh non-empty) REPLACES the claim: every Xcode
            // install migrates the data container, the path-keyed entries
            // re-key wholesale, and unioning would keep each install's dead
            // predecessors claimed forever. A warm build proves nothing new
            // and keeps the surviving claim. A PARTIAL recompile briefly
            // orphans the set's still-warm remainder; GC reclaims it and
            // the next build recompiles those programs once -- self-healing.
            let old = Set(claims.sets[key] ?? [])
            let merged = fresh.isEmpty ? old.intersection(after) : fresh
            claims.sets[key] = merged.sorted()
            saveClaims(claims, build)
            lock.unlock()
            log.info("""
                claim \(key, privacy: .public): +\(fresh.count) \
                (total \(merged.count))
                """)
            if AneCache.cacheMode == .hardlink { shadowLive(build) }
        }
    }

    // ---- once-per-launch survey: restore experiment, GC, instrumentation --

    private func launchSurvey() {
        let root = cacheRoot()
        // Verifiable at a glance each launch (see report()).
        report("mode=\(AneCache.cacheMode.rawValue) root \(root.path)")
        report("state dir \(stateDir().path)")
        if AneCache.cacheMode == .hardlink {
            report("shadow root \(shadowRoot().path)")
        }
        if let build = osBuild() {
            let dir = root.appendingPathComponent(build)
            // iOS regenerates the Data container on every install and carries
            // the old cache forward, but its entries are keyed to the old
            // absolute path -- the daemon re-keys and recompiles regardless.
            // Left alone they pile up (measured 42 -> 153 entries / 1.5 GB, and
            // the daemon's per-compile scan slowed to 24s). On a migration the
            // whole cache AND its shadow are provably dead, so drop them before
            // compiling; the compile is cold either way. macOS containers never
            // migrate and are never wiped here.
            if isOS && containerMigrated() {
                let mb = directoryBytes(dir) >> 20
                try? FileManager.default.removeItem(at: dir)
                try? FileManager.default.removeItem(at: shadowDir(build))
                report("migration: dropped \(mb) MB dead cache + shadow "
                    + "(re-keyed to a new container path)")
            }
            // The relink is NOT here: on a fresh install the cache dir does not
            // exist yet, so restoreFromShadow runs later -- after a priming
            // compile has made the daemon create+own it.
            let names = entries(dir)
            let fresh = updateSeen(names, build)
            let mb = directoryBytes(dir) >> 20
            report("""
                survey \(build): \(names.count) entries \(mb) MB, \
                \(fresh) new since last launch
                """)
            // Shadow state independent of a build, so a purge shows as live
            // shrinking while the shadow still holds the blocks (or not).
            if AneCache.cacheMode == .hardlink {
                let sdir = shadowDir(build)
                let sn = entries(sdir).count
                report("shadow \(build): \(sn) entries "
                    + "\(directoryBytes(sdir) >> 20) MB at \(sdir.path)")
            }
            collect(dir, build, names)
            if AneCache.cacheMode == .hardlink { pruneStaleShadows(build) }
        } else {
            log.warning("OS build not parsed; cache keeper idle")
        }
    }

    // Drop shadow-<build> dirs for every OS build but the current one: the
    // cache re-keys wholesale on an OS update, so an old build's shadow can
    // only pin dead blocks. Mirrors the cache's own stale-OS-build self-prune.
    private func pruneStaleShadows(_ current: String) {
        let fm = FileManager.default
        let kids = (try? fm.contentsOfDirectory(
            at: shadowRoot(), includingPropertiesForKeys: nil)) ?? []
        for k in kids where k.lastPathComponent.hasPrefix("shadow-")
            && k.lastPathComponent != "shadow-\(current)" {
            try? fm.removeItem(at: k)
        }
    }

    // Delete entries that are unclaimed by every installed set, previously
    // seen (not born this hour), and old enough that no compile can still be
    // writing them -- and only when the claim gate is open.
    private func collect(_ dir: URL, _ build: String, _ names: Set<String>) {
        lock.lock()
        let claims = loadClaims(build)
        let seen = loadSeen(build)
        lock.unlock()
        if let hold = gateClosed(claims) {
            log.info("GC held: \(hold, privacy: .public)")
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
                    // Drop the entry's hardlink shadow with it, so the shadow
                    // never outlives the cache entry and holds its blocks after
                    // the last real copy is gone.
                    try? fm.removeItem(
                        at: shadowDir(build).appendingPathComponent(name))
                    deleted += 1
                }
            }
            log.info("""
                GC: deleted \(deleted) unclaimed entries, \
                \(bytes >> 20) MB freed
                """)
        }
    }

    // The gate: every installed, complete, non-GGUF catalog set must hold a
    // non-empty claim (a warm build diffs to nothing and proves no
    // ownership, so its set keeps the gate closed until it compiles cold
    // once). Returns the reason the gate is closed, or nil when open.
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

    // ---- hardlink shadow: survive an unlink-style purge ------------------

    // Shadow the COMPLETE cache AFTER the build finishes: hardlink every live
    // entry file not yet in the shadow. Running it here (not before the load)
    // both seeds a model compiled before the mode was on and captures this
    // build's fresh entries, in one place, against a finished cache. `mirror`
    // links only the missing files, so a warm rebuild that added nothing is a
    // no-op.
    private func shadowLive(_ build: String) {
        let live = cacheRoot().appendingPathComponent(build)
        let shadow = shadowDir(build)
        let t0 = Date()
        // refresh: a recompile writes a NEW inode at an existing path (a purge
        // that emptied the shadow, an OS re-pin), and a link-if-missing mirror
        // would keep the stale inode forever -- so re-link where the inodes
        // differ, healing the shadow to the daemon's current file.
        let linked = mirror(from: live, to: shadow, refresh: true,
                            label: "shadow")
        excludeFromBackup(shadow)
        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        let shadowMB = directoryBytes(shadow) >> 20
        let liveMB = directoryBytes(live) >> 20
        // shadow == live proves the hardlinks carry the compiled blocks; shadow
        // << live means the cross-container link is landing empty.
        report("shadow saved: +\(linked) file(s) in \(ms) ms "
            + "(shadow \(shadowMB) MB vs live \(liveMB) MB) "
            + "from \(live.path) to \(shadow.path)")
        sampleShadowLink(live: live, shadow: shadow)
    }

    // Log one shadow file against its live counterpart -- inode + size on both
    // sides. A real hardlink shares the inode AND the size; a mismatched inode
    // or a 0-byte sample means the cross-container link is not sharing blocks.
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

    // Drop a set's cache entries AND their shadows so the next build recompiles
    // clean -- the recovery for a load that failed on a corrupt or stale
    // relinked bundle. Uses the set's claim to target only its entries; an
    // unclaimed set (never seeded) falls back to the whole OS-build dir.
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

    // Relink the shadow's preserved entries into the cache dir. Called from the
    // build task AFTER a priming compile has made the daemon create+own the dir
    // (a fresh install has nothing to restore into until then), and BEFORE the
    // full model load, so the daemon warm-hits the restored inodes on lookup.
    func restoreFromShadow() {
        if AneCache.cacheMode == .hardlink, let build = osBuild() {
            relinkIfCold(cacheRoot().appendingPathComponent(build), build)
        }
    }

    // Restore any cache file the daemon has not (re)written by relinking the
    // surviving inode from the shadow. A warm launch finds the files present
    // and relinks nothing; a shadow file emptied by a data-block purge relinks
    // empty and the daemon recompiles it, exactly as with no shadow.
    private func relinkIfCold(_ live: URL, _ build: String) {
        let shadow = shadowDir(build)
        if FileManager.default.fileExists(atPath: shadow.path) {
            let shadowMB = directoryBytes(shadow) >> 20
            let restored = mirror(from: shadow, to: live, label: "relink")
            // Always report: shadow size tells whether it held real blocks to
            // restore (0 MB = the hardlinks never carried content), and the
            // restore count whether they landed in the fresh cache.
            report("relink: shadow \(shadowMB) MB, restored \(restored) "
                + "file(s) from \(shadow.path) to \(live.path)")
        }
    }

    // Hardlink every regular file under `from` whose counterpart under `to` is
    // MISSING (or, with refresh, points at a DIFFERENT inode), creating parent
    // dirs; returns the count linked. link(2) shares the inode, so this never
    // copies bytes and is idempotent (matching counterparts are skipped).
    // refresh=false (restore direction) never clobbers a present live file with
    // a stale shadow; refresh=true (shadow direction) heals a drifted inode.
    private func mirror(from: URL, to: URL, refresh: Bool = false,
                        label: String = "") -> Int {
        let fm = FileManager.default
        var count = 0
        // The enumerator yields symlink-STANDARDIZED child URLs (iOS resolves
        // /var -> /private/var), while `from` may be the raw /var form from
        // cacheRoot(); slicing the raw-length prefix off a standardized child
        // mangles every relative path and scatters the links to a sibling of
        // `to`. Slice both in the same standardized form so the prefix matches.
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
            // Restore recovery: an in-place purge can leave the live file
            // PRESENT but emptied (smaller than the shadow's full copy) rather
            // than unlinking it; replace it from the shadow. A content-bearing
            // live file is never clobbered (the daemon's is authoritative), and
            // a shared inode reads equal sizes, so it fires only on real drift.
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
                    // Per-file forensics: WHY it relinked (missing/truncated/
                    // stale), and a real hardlink makes dst share src's inode
                    // AND size; a shared==false or 0-byte dst means the link
                    // did not carry the compiled blocks.
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

    // The file's inode number (0 if unreadable), for the refresh comparison.
    private func inode(_ url: URL) -> UInt64 {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
    }

    // ---- paths + state ---------------------------------------------------

    // Prefer the bundle-id-nested cache dir (what iOS and the sandboxed
    // macOS app actually use); the bare sibling only when it alone exists.
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

    // The shadow lives in the Data container's Application Support, same volume
    // as the e5 cache it hardlinks (so link(2) never crosses volumes) and not
    // purged like Caches. Its whole job is surviving an IN-PLACE cache purge
    // (same install, same path), so it need not outlive the container -- and
    // the Data container is preserved across upgrade installs anyway. Was the
    // app-group container, dropped: iOS 16/26 spuriously regenerates it, and
    // nothing here needs a group container.
    private func shadowRoot() -> URL {
        stateDir().appendingPathComponent("anecache-shadow", isDirectory: true)
    }

    private func shadowDir(_ build: String) -> URL {
        shadowRoot().appendingPathComponent("shadow-\(build)",
                                            isDirectory: true)
    }

    // Set identity for claims: <model dir>/<revision leaf> -- unique across
    // catalog sets and stable across launches.
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

    // Record first-seen stamps; returns how many entries are new since the
    // previous launch. Vanished entries drop out so the file cannot grow
    // without bound.
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
        // Entries present last launch but gone now = evicted (an in-place
        // purge, the case the shadow defends). Name them: the relink lines that
        // follow should restore exactly these.
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

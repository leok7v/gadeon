import Foundation
import Testing
@testable import LLM

// Network-gated (set GADEON_HUB_TEST=1). Proves the Hub API parsing (commit sha ->
// tree -> oid extraction) and the git-blob-sha1 digest against the live 0.8B
// repo WITHOUT pulling the ~561MB set: it fetches only the tree and the ~20KB
// tokenizer_config.json, whose pull() verifies the digest and throws on mismatch.
// The sha256/LFS path is exercised by a real full download (CLI/app).
struct HubFetchTests {
    @Test func hubSmallFileVerifies() async throws {
        if ProcessInfo.processInfo.environment["GADEON_HUB_TEST"] == "1" {
            let repo = "leok7v/Qwen3.5-0.8B-coreml"
            let sha = try await HubFetch.commit(repo, "main")
            #expect(sha.count == 40)
            let tree = try await HubFetch.tree(repo, sha)
            let cfg = tree.first { e in e.path == "tokenizer_config.json" }
            #expect(cfg != nil)
            if let cfg {
                #expect(cfg.lfs == false)
                let dst = FileManager.default.temporaryDirectory
                    .appendingPathComponent("hubtest-\(sha).json")
                try? FileManager.default.removeItem(at: dst)
                let pump = Pump()
                try await HubFetch.pull(repo, sha, cfg, dst, pump,
                                        false) { _ in }
                pump.done()
                #expect(FileManager.default.fileExists(atPath: dst.path))
                try? FileManager.default.removeItem(at: dst)
            }
        }
    }

    private func scratch() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("assemble-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("blob.gguf.part")
    }

    private func write(_ url: URL, _ text: String) {
        try? Data(text.utf8).write(to: url)
    }

    @Test func assembleTakesTheContiguousPrefixOnly() throws {
        let part = scratch()
        write(part, "")
        write(HubFetch.piece(part, 0), "hello ")
        write(HubFetch.piece(part, 6), "world")
        write(HubFetch.piece(part, 99), "orphan")
        let have = try HubFetch.assemble(part)
        #expect(have == 11)
        #expect(try String(decoding: Data(contentsOf: part), as: UTF8.self)
                == "hello world")
        let fm = FileManager.default
        #expect(!fm.fileExists(atPath: HubFetch.piece(part, 0).path))
        #expect(!fm.fileExists(atPath: HubFetch.piece(part, 6).path))
        #expect(fm.fileExists(atPath: HubFetch.piece(part, 99).path))
        HubFetch.sweep(part)
        #expect(!fm.fileExists(atPath: HubFetch.piece(part, 99).path))
        try? fm.removeItem(at: part.deletingLastPathComponent())
    }

    @Test func assembleNeverWritesAHole() throws {
        let part = scratch()
        write(part, "abc")
        write(HubFetch.piece(part, 99), "way past the end")
        let have = try HubFetch.assemble(part)
        #expect(have == 3)
        #expect(try Data(contentsOf: part).count == 3)
        #expect(FileManager.default.fileExists(
            atPath: HubFetch.piece(part, 99).path))
        try? FileManager.default.removeItem(
            at: part.deletingLastPathComponent())
    }
}

struct HubDrainTests {
    @Test func drainPullsARealFileThroughTheBackgroundSession() async throws {
        if ProcessInfo.processInfo.environment["GADEON_HUB_TEST"] == "1" {
            let repo = "leok7v/Qwen3.5-0.8B-coreml"
            let sha = try await HubFetch.commit(repo, "main")
            let tree = try await HubFetch.tree(repo, sha)
            let e = try #require(tree.first { t in
                t.path.hasSuffix("vision.mlmodelc/weights/weight.bin") })
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("drain-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
            let part = dir.appendingPathComponent("weight.bin.part")
            let raw = "\(HubFetch.host)/\(repo)/resolve/\(sha)/"
                + HubFetch.esc(e.path)
            let url = try #require(URL(string: raw))
            let t0 = Date()
            try await HubFetch.drain(url, e, part, { n in
                _ = n
            }, parked: { true })
            let secs = Date().timeIntervalSince(t0)
            let got = HubFetch.size(part)
            print("drained \(got) of \(e.size) in \(secs)s")
            #expect(got == e.size)
            try? FileManager.default.removeItem(at: dir)
        }
    }
}

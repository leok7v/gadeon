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
                try await HubFetch.pull(repo, sha, cfg, dst) { _ in }
                #expect(FileManager.default.fileExists(atPath: dst.path))
                try? FileManager.default.removeItem(at: dst)
            }
        }
    }
}

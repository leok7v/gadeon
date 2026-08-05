import Foundation

// The HF source for each model the picker offers -- multi-file CoreML/ANE sets
// AND single-file ternary GGUFs served by the unified LLM engine (Metal/SIMD).
// Both download through the same HubFetch flow into modelStore/<name>/<sha>/;
// the revision is a PINNED commit sha so a later Hub push cannot swap weights
// under a released binary. `bytes` is the on-disk size, shown in the consent
// prompt. This lives in Shared/ (next to HubFetch), not CoreML/, because it is
// backend-agnostic.
public enum ModelCatalog {
    public struct Source: Sendable {
        public let repo: String
        public let revision: String        // pinned commit sha
        public let bytes: Int64
        // Exact files to fetch (nil = the whole repo). A self-contained GGUF
        // needs only its one weight file, not the repo's vision projector +
        // docs; a CoreML set needs every file (nil).
        public let files: [String]?

        public init(repo: String, revision: String, bytes: Int64,
                    files: [String]? = nil) {
            self.repo = repo
            self.revision = revision
            self.bytes = bytes
            self.files = files
        }
    }

    public static let sources: [String: Source] = [
        // Revisions pin the commit carrying each set's generation_config.json
        // with the per-mode sampling matrix (_sampling_presets) the runtime
        // selects on the fly.
        // 0.8B + 2B vision towers are pal8 (8-bit palettized) as of these pins:
        // ~half the tower, OCR identical to fp16 (measured A/B). 4B/9B
        // stay fp16.
        // These pins carry the ORIGIN-form config.json (rope_theta nested at
        // text_config.rope_parameters; the engine cross-verifies its I/O-derived
        // geometry against it at load). The 0.8B pin also adds the Apache-2.0
        // LICENSE the origin ships.
        "Qwen3.5-0.8B": Source(
            repo: "leok7v/Qwen3.5-0.8B-coreml",
            revision: "2a965bb43c7ee13a90fdad98dbc867a8c2a630b9",
            bytes: 663_000_000),
        "QwenPaw-Flash-2B": Source(
            repo: "leok7v/QwenPaw-Flash-2B-coreml",
            revision: "9cf1086cdf9bc9ca653df488061c75986679a974",
            bytes: 1_640_000_000),
        // The 4B and 9B come from the -MTP repos: their trunks carry a third
        // "verify" function in the same weight blob, so self-speculative
        // decode costs only the drafter (4B +86 MB, 9B +174 MB) rather than a
        // second copy of the weights. Both drafters are baked from the origin
        // Qwen3.5 safetensors, and the 4B's shared-blob verify makes its
        // speculative output identical to its own greedy decode.
        "QwenPaw-Flash-4B": Source(
            repo: "leok7v/QwenPaw-Flash-4B-MTP-coreml",
            revision: "d20d2c120980b4802dfcd1686865fc1a969276af",
            bytes: 3_951_000_000),
        "QwenPaw-Flash-9B": Source(
            repo: "leok7v/QwenPaw-Flash-9B-MTP-coreml",
            revision: "84e90005d20dcdf0d5fcbb53564176373651a1b5",
            bytes: 7_854_000_000),
        // The ternary 27B GGUF served by the unified engine: the Q2_0 weight
        // file, the Q8_0 vision tower (measured transparent vs BF16, cos
        // ~0.99995 -- ViT.swift consumes it), and generation_config.json for
        // the sampling matrix (MetalChat reads it from beside the weight).
        "Ternary-Bonsai-27B": Source(
            repo: "leok7v/Ternary-Bonsai-27B-gguf",
            revision: "53a78b8ff8e8792fc65ced23afe99684e5403bba",
            bytes: 7_794_368_480,
            files: ["Ternary-Bonsai-27B-Q2_0.gguf",
                    "Ternary-Bonsai-27B-mmproj-Q8_0.gguf",
                    "generation_config.json"]),
        // The dense 1.7B (Qwen3, text-only): a Metal/GPU ternary companion for
        // RAM-constrained iOS. The one Q2_0 weight + its sampling matrix.
        "Ternary-Bonsai-1.7B": Source(
            repo: "leok7v/Ternary-Bonsai-1.7B-gguf",
            revision: "877587692a0c131a4c271d593f7a26903ae1e2f1",
            bytes: 463_292_000,
            files: ["Ternary-Bonsai-1.7B-Q2_0.gguf",
                    "generation_config.json"]),
        // gemma-4-E2B (QAT), the one model here that hears and watches as well
        // as reads. Self-contained: the sampling matrix, the chat template and
        // BOTH towers (vision + audio) live in the single file, so unlike the
        // Bonsai sets it needs no sibling mmproj or generation_config.
        "gemma-4-E2B": Source(
            repo: "leok7v/gemma-4-e2b-it-qat",
            revision: "47c144d18e31fb4d6cdde6f593b0c3db986610e6",
            bytes: 2_665_414_656,
            files: ["gemma-4-e2b-it-qat.gguf"]),
        // The same architecture one size up: 42 layers at 2560 wide against
        // E2B's 35 at 1536, grouped-query attention rather than multi-query,
        // and every MLP left at 4 bits where E2B drops its wide half to 2.
        // The vision and audio towers are the same weights, so only the text
        // tower and the two projections into it grow.
        "gemma-4-E4B": Source(
            repo: "leok7v/gemma-4-e4b-it-qat",
            revision: "ac03211d7016197b97d1abc8f4f9f36632e1e974",
            bytes: 3_794_550_784,
            files: ["gemma-4-e4b-it-qat.gguf"]),
        // gemma-4-12B, the UNIFIED architecture rather than a larger E4B: no
        // per-layer embeddings, no shared KV, and multimodality with no tower
        // at all -- an image is raw pixel patches and audio is raw waveform
        // frames, each through one projection into the embedding space. Its
        // full-attention layers keep a single 512-wide kv head whose key and
        // value share one projection, so its KV per token is far below what
        // 48 layers at 3840 would suggest.
        "gemma-4-12B": Source(
            repo: "leok7v/gemma-4-12b-it-qat",
            revision: "b22113f7912356882273609f1a7802bfc1292f04",
            bytes: 6_822_641_664,
            files: ["gemma-4-12b-it-qat.gguf"]),
    ]

    public static func source(_ name: String) -> Source? { sources[name] }

    // The GGUF weight file inside each single-file (unified-engine) model's set
    // dir. A name here is a GGUF model (isGguf); absent is a CoreML/ANE set.
    public static let ggufFiles: [String: String] = [
        "Ternary-Bonsai-27B": "Ternary-Bonsai-27B-Q2_0.gguf",
        "Ternary-Bonsai-1.7B": "Ternary-Bonsai-1.7B-Q2_0.gguf",
        "gemma-4-E2B": "gemma-4-e2b-it-qat.gguf",
        "gemma-4-E4B": "gemma-4-e4b-it-qat.gguf",
        "gemma-4-12B": "gemma-4-12b-it-qat.gguf",
    ]

    public static func isGguf(_ name: String) -> Bool {
        ggufFiles[name] != nil
    }

    // The on-disk path of a GGUF model's weight file (inside its downloaded
    // set dir modelStore/<name>/<sha>/), or nil if `name` is not a GGUF model.
    public static func ggufPath(_ name: String, in dest: URL) -> String? {
        var result: String? = nil
        if let file = ggufFiles[name], let set = localSet(name, in: dest) {
            result = set.appendingPathComponent(file).path
        }
        return result
    }

    // The verified set lands at dest/<pinned-sha>/ (HubFetch stages by
    // commit). Because the pin IS the sha, the app names that dir offline and
    // checks completeness with no network round-trip.
    public static func localSet(_ name: String, in dest: URL) -> URL? {
        var result: URL? = nil
        if let src = sources[name] {
            result = dest.appendingPathComponent(name, isDirectory: true)
                .appendingPathComponent(src.revision, isDirectory: true)
        }
        return result
    }

    // Present ONLY when HubFetch's .complete sentinel is in the dir (a fully
    // verified tree): a half-download would make Engine.loadEssential take a
    // wrong file-presence branch, so presence must mean complete.
    public static func isComplete(_ setDir: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: setDir.appendingPathComponent(HubFetch.sentinel).path)
    }
}

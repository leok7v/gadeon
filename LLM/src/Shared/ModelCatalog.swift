import Foundation

public enum ModelCatalog {
    public struct Source: Sendable {
        public let repo: String
        public let revision: String        // pinned commit sha
        public let bytes: Int64
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
        "Qwen3.5-4B": Source(
            repo: "leok7v/Qwen3.5-4B",
            revision: "70ec6653daabea753fe9ca3466cfd618bb70cdfb",
            bytes: 3_584_533_984,
            files: ["Qwen3.5-4B-UD-Q4_K_XL.ggxf"]),
        "Qwen3.5-9B": Source(
            repo: "leok7v/Qwen3.5-9B",
            revision: "5c9b8db474384f238c88745492401fea0a3813be",
            bytes: 6_884_262_304,
            files: ["Qwen3.5-9B-UD-Q4_K_XL.ggxf"]),
        "Qwen3.8-27B-IQ1_S": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "a047f6e4e0454c4218635f546d749a6041647e6d",
            bytes: 7_119_830_240,
            files: ["Qwen3.8-27B-IQ1_S.ggxf"]),
        "Qwen3.8-27B-IQ2_XXS": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "a047f6e4e0454c4218635f546d749a6041647e6d",
            bytes: 8_459_696_256,
            files: ["Qwen3.8-27B-IQ2_XXS.ggxf"]),
        "Qwen3.8-27B-IQ3_XXS": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "a047f6e4e0454c4218635f546d749a6041647e6d",
            bytes: 11_862_468_736,
            files: ["Qwen3.8-27B-IQ3_XXS.ggxf"]),
        "Qwen3.8-27B-IQ4_XS": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "a047f6e4e0454c4218635f546d749a6041647e6d",
            bytes: 15_180_454_016,
            files: ["Qwen3.8-27B-IQ4_XS.ggxf"]),
        "Qwen3.8-27B-Q4_K_S": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "a047f6e4e0454c4218635f546d749a6041647e6d",
            bytes: 16_285_821_056,
            files: ["Qwen3.8-27B-Q4_K_S.ggxf"]),
        "Ternary-Bonsai-27B": Source(
            repo: "leok7v/Ternary-Bonsai-27B-gguf",
            revision: "0e9bbf1b159573dcca23f59c82b365ac58422e00",
            bytes: 7_794_369_344,
            files: ["Ternary-Bonsai-27B-Q2_0.ggxf"]),
        "Ternary-Bonsai-1.7B": Source(
            repo: "leok7v/Ternary-Bonsai-1.7B-gguf",
            revision: "9363219fb1ca4b27524b53e0b782cad8b2b34867",
            bytes: 463_292_000,
            files: ["Ternary-Bonsai-1.7B-Q2_0.ggxf"]),
        "gemma-4-E2B": Source(
            repo: "leok7v/gemma-4-e2b-it-qat",
            revision: "20bad044a76eac59e0a897d2607541229c48efae",
            bytes: 2_665_414_656,
            files: ["gemma-4-e2b-it-qat.ggxf"]),
        // The same architecture one size up: 42 layers at 2560 wide against
        // E2B's 35 at 1536, grouped-query attention rather than multi-query,
        // and every MLP left at 4 bits where E2B drops its wide half to 2.
        // The vision and audio towers are the same weights, so only the text
        // tower and the two projections into it grow.
        "gemma-4-E4B": Source(
            repo: "leok7v/gemma-4-e4b-it-qat",
            revision: "25bca6860fab6acc3988e494f8c7e58baf2441c4",
            bytes: 3_794_550_784,
            files: ["gemma-4-e4b-it-qat.ggxf"]),
        // gemma-4-12B, the UNIFIED architecture rather than a larger E4B: no
        // per-layer embeddings, no shared KV, and multimodality with no tower
        // at all -- an image is raw pixel patches and audio is raw waveform
        // frames, each through one projection into the embedding space. Its
        // full-attention layers keep a single 512-wide kv head whose key and
        // value share one projection, so its KV per token is far below what
        // 48 layers at 3840 would suggest.
        "gemma-4-12B": Source(
            repo: "leok7v/gemma-4-12b-it-qat",
            revision: "2a39ef64662cd82129ea2421193d9b7712464ad9",
            bytes: 7_060_897_792,
            files: ["gemma-4-12b-it-qat.ggxf"]),
    ]

    public static func source(_ name: String) -> Source? {
        sources[name]
    }

    public static let ggufFiles: [String: String] = [
        "Qwen3.5-4B": "Qwen3.5-4B-UD-Q4_K_XL.ggxf",
        "Qwen3.5-9B": "Qwen3.5-9B-UD-Q4_K_XL.ggxf",
        "Qwen3.8-27B-IQ1_S": "Qwen3.8-27B-IQ1_S.ggxf",
        "Qwen3.8-27B-IQ2_XXS": "Qwen3.8-27B-IQ2_XXS.ggxf",
        "Qwen3.8-27B-IQ3_XXS": "Qwen3.8-27B-IQ3_XXS.ggxf",
        "Qwen3.8-27B-IQ4_XS": "Qwen3.8-27B-IQ4_XS.ggxf",
        "Qwen3.8-27B-Q4_K_S": "Qwen3.8-27B-Q4_K_S.ggxf",
        "Ternary-Bonsai-27B": "Ternary-Bonsai-27B-Q2_0.ggxf",
        "Ternary-Bonsai-1.7B": "Ternary-Bonsai-1.7B-Q2_0.ggxf",
        "gemma-4-E2B": "gemma-4-e2b-it-qat.ggxf",
        "gemma-4-E4B": "gemma-4-e4b-it-qat.ggxf",
        "gemma-4-12B": "gemma-4-12b-it-qat.ggxf",
    ]

    public static func isGguf(_ name: String) -> Bool {
        ggufFiles[name] != nil
    }

    // The on-disk path of a GGUF model's weight file (inside its downloaded
    // set dir modelStore/<name>/<sha>/), or nil if `name` is not a GGUF model.
    public static func ggufPath(_ name: String, in dest: URL) -> String? {
        var result: String? = nil
        if let file = ggufFiles[name], let set = localSet(name, in: dest) {
            let url = set.appendingPathComponent(file)
            adoptLegacy(url)
            result = url.path
        }
        return result
    }

    private static func adoptLegacy(_ url: URL) {
        let fm = FileManager.default
        let was = url.deletingPathExtension().appendingPathExtension("gguf")
        if !fm.fileExists(atPath: url.path),
           fm.fileExists(atPath: was.path) {
            try? fm.moveItem(at: was, to: url)
        }
    }

    // The verified set lands at dest/<pinned-sha>/ (HubFetch stages by
    // commit). Because the pin IS the sha, the app names that dir offline and
    // checks completeness with no network round-trip.
    public static func localSet(_ name: String, in dest: URL) -> URL? {
        var result: URL? = nil
        if let src = source(name) {
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

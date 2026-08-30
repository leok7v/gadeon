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
            revision: "9b5937927636b99e3fc0b9b115d6df520cba260d",
            bytes: 3_584_533_984,
            files: ["Qwen3.5-4B-UD-Q4_K_XL.ggxf"]),
        "Qwen3.5-9B": Source(
            repo: "leok7v/Qwen3.5-9B",
            revision: "2bb6530cee7044ed90f3bb5d5d4a349d405024f4",
            bytes: 6_884_262_304,
            files: ["Qwen3.5-9B-UD-Q4_K_XL.ggxf"]),
        "Qwen3.8-27B-IQ1_S": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "8562611e196c867b4603ada2742e2e1e83a29dc7",
            bytes: 7_119_830_240,
            files: ["Qwen3.8-27B-IQ1_S.ggxf"]),
        "Qwen3.8-27B-IQ2_XXS": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "8562611e196c867b4603ada2742e2e1e83a29dc7",
            bytes: 8_459_696_256,
            files: ["Qwen3.8-27B-IQ2_XXS.ggxf"]),
        "Qwen3.8-27B-IQ3_XXS": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "8562611e196c867b4603ada2742e2e1e83a29dc7",
            bytes: 11_862_468_736,
            files: ["Qwen3.8-27B-IQ3_XXS.ggxf"]),
        "Qwen3.8-27B-IQ4_XS": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "8562611e196c867b4603ada2742e2e1e83a29dc7",
            bytes: 15_180_454_016,
            files: ["Qwen3.8-27B-IQ4_XS.ggxf"]),
        "Qwen3.8-27B-Q4_K_S": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "8562611e196c867b4603ada2742e2e1e83a29dc7",
            bytes: 16_285_821_056,
            files: ["Qwen3.8-27B-Q4_K_S.ggxf"]),
        "Ternary-Bonsai-27B": Source(
            repo: "leok7v/Ternary-Bonsai-27B-gguf",
            revision: "369aa33df3d8a8255a35a160305b6632eeb3f1a1",
            bytes: 7_794_369_344,
            files: ["Ternary-Bonsai-27B-Q2_0.ggxf"]),
        "Ternary-Bonsai-1.7B": Source(
            repo: "leok7v/Ternary-Bonsai-1.7B-gguf",
            revision: "0fb751628b00550b9656984c5870e4cfdea392a7",
            bytes: 463_292_000,
            files: ["Ternary-Bonsai-1.7B-Q2_0.ggxf"]),
        "gemma-4-E2B": Source(
            repo: "leok7v/gemma-4-e2b-it-qat",
            revision: "a98d8daed9141ec98f68f54fafa7a08efa8ca580",
            bytes: 2_665_414_656,
            files: ["gemma-4-e2b-it-qat.ggxf"]),
        // The same architecture one size up: 42 layers at 2560 wide against
        // E2B's 35 at 1536, grouped-query attention rather than multi-query,
        // and every MLP left at 4 bits where E2B drops its wide half to 2.
        // The vision and audio towers are the same weights, so only the text
        // tower and the two projections into it grow.
        "gemma-4-E4B": Source(
            repo: "leok7v/gemma-4-e4b-it-qat",
            revision: "712cfb35aad315ed3f8fc3f92c0c04b83ee2023f",
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
            revision: "63c8311ff0d884228a7e3c415f4516d40daf755c",
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
            dropStale(name, in: dest)
            result = set.appendingPathComponent(file).path
        }
        return result
    }

    public static func dropStale(_ name: String, in dest: URL) {
        let fm = FileManager.default
        if let set = localSet(name, in: dest) {
            let home = dest.appendingPathComponent(name, isDirectory: true)
            let kids = (try? fm.contentsOfDirectory(
                at: home, includingPropertiesForKeys: nil)) ?? []
            for kid in kids
            where kid.lastPathComponent != set.lastPathComponent {
                try? fm.removeItem(at: kid)
            }
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

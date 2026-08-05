import Foundation

// What a loaded model IS, for a surface that has to say something before any
// turn has produced numbers. Every field is read from the model's own files --
// a metadata key, a tensor's size, a compiled program on disk -- so a new
// checkpoint describes itself and nothing here is keyed by model name.

public struct ModelShape: Sendable {

    // A weight group and what it weighs. A multimodal file's gigabytes are
    // not all language, and which sense they bought is the interesting half.
    public struct Tower: Sendable {
        public let name: String
        public let bytes: Int

        public init(name: String, bytes: Int) {
            self.name = name
            self.bytes = bytes
        }
    }

    public let towers: [Tower]
    // Positions the checkpoint was TRAINED to address, which is not what a
    // device can afford to allocate for one. 0 when no file says.
    public let trainedContext: Int
    public let embedding: Int

    public init(towers: [Tower], trainedContext: Int, embedding: Int) {
        self.towers = towers
        self.trainedContext = trainedContext
        self.embedding = embedding
    }

    static func grouped(_ text: Int, _ vision: Int,
                        _ audio: Int) -> [Tower] {
        var out = [Tower(name: "text", bytes: text)]
        if vision > 0 { out.append(Tower(name: "vision", bytes: vision)) }
        if audio > 0 { out.append(Tower(name: "audio", bytes: audio)) }
        return out
    }

    private static func json(_ url: URL) -> [String: Any] {
        let parsed = (try? Data(contentsOf: url)).flatMap { data in
            try? JSONSerialization.jsonObject(with: data)
        }
        return parsed as? [String: Any] ?? [:]
    }

    // A directory's whole contents, or a plain file's own size.
    private static func size(_ url: URL) -> Int {
        let fm = FileManager.default
        var total = 0
        var isDir: ObjCBool = false
        if fm.fileExists(atPath: url.path, isDirectory: &isDir) {
            if isDir.boolValue {
                let walk = fm.enumerator(
                    at: url, includingPropertiesForKeys: [.fileSizeKey])
                while let item = walk?.nextObject() as? URL {
                    total += (try? item.resourceValues(
                        forKeys: [.fileSizeKey]))?.fileSize ?? 0
                }
            } else {
                total = ((try? fm.attributesOfItem(atPath: url.path))?[.size]
                    as? Int) ?? 0
            }
        }
        return total
    }
}

extension ModelShape {

    // A GGUF describes itself completely. The tower a tensor serves is in the
    // name the emit gives it -- vision under `v.` / `mm.vision`, audio under
    // `a.` / `mm.audio` -- and both scalars are keyed by the file's own
    // architecture. `sidecars` carries a tower that ships as a separate file.

    init(gguf g: GGUF, sidecars: [Tower] = []) {
        let arch = g.string("general.architecture") ?? ""
        var text = 0
        var vision = 0
        var audio = 0
        for (name, t) in g.tensors {
            if name.hasPrefix("v.") || name.hasPrefix("mm.vision") {
                vision += t.byteCount
            } else if name.hasPrefix("a.") || name.hasPrefix("mm.audio") {
                audio += t.byteCount
            } else {
                text += t.byteCount
            }
        }
        self.init(
            towers: ModelShape.grouped(text, vision, audio) + sidecars,
            trainedContext: g.int(arch + ".context_length")
                ?? ModelShape.sourceContext(g),
            embedding: g.int(arch + ".embedding_length") ?? 0)
    }

    // The gemma repacks carry their origin config.json verbatim, and the
    // earlier ones state the trained context only in there.

    private static func sourceContext(_ g: GGUF) -> Int {
        var out = 0
        if let raw = g.string("gemma4.source.config_json"),
           let data = raw.data(using: .utf8),
           let root = (try? JSONSerialization.jsonObject(with: data))
               as? [String: Any],
           let text = root["text_config"] as? [String: Any],
           let n = (text["max_position_embeddings"] as? NSNumber)?.intValue {
            out = n
        }
        return out
    }
}

public extension ModelShape {

    // A CoreML set is a DIRECTORY of compiled programs, so its towers are
    // measured rather than declared: vision.mlmodelc is the eye, every other
    // program plus the embedding sidecar the language trunk. The two scalars
    // come from the config.json the set ships, which Engine cross-checks
    // against the programs' own I/O at load -- so reading it here is reading
    // them.

    init(coremlSet dir: URL) {
        let names = (try? FileManager.default
            .contentsOfDirectory(atPath: dir.path)) ?? []
        var text = 0
        var vision = 0
        for name in names
        where name.hasSuffix(".mlmodelc") || name.hasSuffix(".bin") {
            let n = ModelShape.size(dir.appendingPathComponent(name))
            if name == "vision.mlmodelc" { vision += n } else { text += n }
        }
        let cfg = ModelShape.json(dir.appendingPathComponent("config.json"))
        let sub = cfg["text_config"] as? [String: Any] ?? cfg
        self.init(
            towers: ModelShape.grouped(text, vision, 0),
            trainedContext:
                (sub["max_position_embeddings"] as? NSNumber)?.intValue ?? 0,
            embedding: (sub["hidden_size"] as? NSNumber)?.intValue ?? 0)
    }
}

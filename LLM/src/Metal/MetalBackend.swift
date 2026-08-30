// Adapts the GPU MetalEngine to the AgentBackend seam, so ChatSession / Agent
// drive it exactly like the SIMD BonsaiBackend and the CoreML EngineBackend.
// A reference type (@unchecked Sendable): ChatSession serializes every call, so
// the engine's mutable state is never touched concurrently.
import CoreML
import Foundation

public final class MetalBackend: AgentBackend, @unchecked Sendable {
    let engine: MetalEngine
    let tokenizer: Tokenizer
    // Path of the sibling Qwen3-VL mmproj; nil = text-only.
    private let mmprojPath: String?
    // The GPU tower, built on the first image turn and kept RESIDENT across
    // turns: a per-turn rebuild cost ~2s of reload + dequant, and the f16
    // weights (~0.5GB) sit comfortably beside the LM on the 16GB-gated hosts
    // the 27B ships to.
    private var vit: MetalViT?
    private var savedMark: MetalEngine.Bookmark?
    // Tower wall time of the last vision prefill (load + tile encodes), for
    // the session's .vision trace event.
    private var towerSeconds = 0.0

    public init(engine: MetalEngine, tokenizer: Tokenizer,
                mmprojPath: String? = nil) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.mmprojPath = mmprojPath
    }

    public var eos: Int32 { tokenizer.eosId }
    public var eosIds: Set<Int32> { tokenizer.eosIds }
    public var position: Int { get async { engine.pos } }

    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text, addSpecial: true)
    }
    public func tokenBytes(_ id: Int32) -> [UInt8] { tokenizer.decodeBytes([id]) }
    public func text(_ ids: [Int32]) -> String { tokenizer.decode(ids) }

    public func reset() async { engine.reset() }
    public func useSampler(_ s: Sampler?) async { engine.sampler = s }
    public func extend(_ ids: [Int32]) async throws -> Int32 {
        // A prefill-phase Stop leaves the batched loop early; surface it as
        // EngineError.stopped so ChatSession rolls the turn back (parity with
        // the CoreML Engine's ingestBatched).
        let out = engine.extend(ids)
        if engine.shouldStop() { throw EngineError.stopped }
        return out
    }
    public func decode(_ token: Int32) async throws -> Int32 {
        engine.decode(token)
    }

    // The engine's nonisolated lock-backed stop, the reliable lever for the
    // synchronous Metal forward: the app raises it while a forward holds the
    // ChatSession actor, prefill honors it, the decode loop polls shouldStop.
    public func requestStop() { engine.requestStop() }
    public func shouldStop() -> Bool { engine.shouldStop() }

    public func mark() async throws { savedMark = engine.bookmark() }
    public func rewind() async throws {
        if let m = savedMark { engine.restore(m) }
    }

    // The tower's GEMM and the vision prefill's batched chunk are both
    // simdgroup-matrix kernels, so a GPU without matrix units has no vision
    // path at all -- report none, and the app never offers the attach UI.
    public func supportsVision() async -> Bool {
        mmprojPath != nil && engine.ctx.matrixUnits
    }

    public func visionEncodeSeconds() async -> Double { towerSeconds }

    // The engine is internal; the app reaches its spec counters here.
    public func drainSpecTurn() -> SpecTurn? {
        engine.drainSpecTurn()
    }

    public func visionGrid() async -> VisionGrid? {
        var result: VisionGrid? = nil
        if let c = vit?.cfg {
            result = VisionGrid(perSide: c.side)
        } else if let mmprojPath,
                  let c = try? ViTConfig(mmprojPath: mmprojPath) {
            result = VisionGrid(perSide: c.side)
        }
        return result
    }

    public func prefillVision(_ ids: [Int32], tiles: VisionTiles,
                              imageStarts: [Int], gridH: Int,
                              gridW: Int) async throws -> Int32 {
        engine.reset()
        return try await extendVision(ids, tiles: tiles,
                                      imageStarts: imageStarts,
                                      gridH: gridH, gridW: gridW)
    }

    public func extendVision(_ ids: [Int32], tiles: VisionTiles,
                             imageStarts: [Int], gridH: Int,
                             gridW: Int) async throws -> Int32 {
        if mmprojPath == nil || !engine.ctx.matrixUnits {
            throw EngineError.missingModel("mmproj")
        }
        // The first image turn pays the one-time tower build; later turns go
        // straight to the resident GPU encode.
        let t0 = Date()
        if vit == nil { vit = try MetalViT(path: mmprojPath!) }
        let tower = vit!
        var feats: [[Float]] = []
        for tile in tiles.tiles {
            feats.append(tower.forward(
                pixels: MetalBackend.pixels(from: tile, cfg: tower.cfg)))
        }
        towerSeconds = Date().timeIntervalSince(t0)
        let out = engine.extendVision(ids, feats: feats,
                                      starts: imageStarts,
                                      gridH: gridH, gridW: gridW)
        if engine.shouldStop() { throw EngineError.stopped }
        return out
    }

    // Invert VisionPreprocess.patchTensor's family layout (merge-block rows;
    // cols channel-major with the temporal duplicate) back to HWC pixels:
    // the seam carries family patch tensors, the ViT eats pixels, and the
    // normalizations agree (family /127.5-1 == this tower's mean/std 0.5).
    static func pixels(from tile: MLMultiArray, cfg: ViTConfig) -> [Float] {
        let patch = VisionPreprocess.patch
        let temporal = VisionPreprocess.temporal
        let merge = VisionPreprocess.merge
        let side = cfg.imageSize
        let gwB = cfg.side / merge
        let dim = 3 * temporal * patch * patch
        let rows = tile.shape[0].intValue
        var px = [Float](repeating: 0, count: side * side * 3)
        tile.withUnsafeBytes { raw in
            let sp = raw.bindMemory(to: Float16.self)
            for r in 0..<rows {
                let mw = r % merge
                let mh = (r / merge) % merge
                let gwb = (r / (merge * merge)) % gwB
                let ghb = r / (merge * merge * gwB)
                for k in 0..<dim
                where (k / (patch * patch)) % temporal == 0 {
                    let pw = k % patch
                    let ph = (k / patch) % patch
                    let ch = k / (patch * patch * temporal)
                    let y = (ghb * merge + mh) * patch + ph
                    let x = (gwb * merge + mw) * patch + pw
                    px[(y * side + x) * 3 + ch] = Float(sp[r * dim + k])
                }
            }
        }
        return px
    }

    struct State: BackendState { let bookmark: MetalEngine.Bookmark }
    public func saveState() async throws -> any BackendState {
        State(bookmark: engine.bookmark())
    }
    public func loadState(_ state: any BackendState) async throws {
        if let s = state as? State { engine.restore(s.bookmark) }
    }

    // A pre-turn rollback point carries the state AND savedMark, so a rollback
    // (prefill cancel, or a title turn) leaves the next turn's rewind pointing
    // where it did rather than at the rolled-back turn -- matching
    // EngineBackend.Turn (the default checkpoint, saveState alone, drops it).
    struct Turn: BackendState {
        let bookmark: MetalEngine.Bookmark
        let mark: MetalEngine.Bookmark?
    }
    public func checkpoint() async throws -> any BackendState {
        Turn(bookmark: engine.bookmark(), mark: savedMark)
    }
    public func rollback(_ state: any BackendState) async throws {
        if let t = state as? Turn {
            engine.restore(t.bookmark)
            savedMark = t.mark
        }
    }

    public func serializeState(_ state: any BackendState) async -> Data {
        var out = Data()
        if let s = state as? State { out = engine.serialize(s.bookmark) }
        return out
    }

    public func deserializeState(_ data: Data) async throws
        -> any BackendState {
        State(bookmark: engine.deserialize(data))
    }
}

// A loaded Bonsai model on the GPU: the Metal engine + a GGUF-driven tokenizer /
// chat template / sampler recs. The GPU counterpart of BonsaiChat; loads
// EVERYTHING from the one GGUF.
public struct MetalChat {
    public let engine: MetalEngine
    public let tokenizer: Tokenizer
    public let chatTemplate: String
    public let samplingPresets: SamplingPresets
    public let mmprojPath: String?
    public let shape: ModelShape

    public init(ggufPath: String, pageP: Int = 512) throws {
        let m = try BonsaiModel(path: ggufPath)
        engine = try MetalEngine(m, pageP: pageP)
        var tok = try Tokenizer(gguf: m.gguf)
        tok.addStops(Tokenizer.stopIds(besideSet: URL(
            fileURLWithPath: ggufPath).deletingLastPathComponent()))
        tokenizer = tok
        chatTemplate = m.gguf.string("tokenizer.chat_template")
            ?? BonsaiChat.fallbackTemplate
        samplingPresets = SamplingPresets.require(gguf: m.gguf,
                                                  path: ggufPath)
        let inside = m.gguf.int("clip.vision.block_count") != nil
        let mmproj = inside ? ggufPath : MetalChat.mmprojBeside(ggufPath)
        mmprojPath = mmproj
        shape = ModelShape(gguf: m.gguf,
                           sidecars: inside ? MetalChat.eye(m.gguf)
                                            : MetalChat.eye(mmproj))
    }

    static func eye(_ gguf: GGUF) -> [ModelShape.Tower] {
        let bytes = gguf.tensors.values
            .filter { t in t.name.hasPrefix("v.") || t.name.hasPrefix("mm.") }
            .reduce(0) { sum, t in sum + t.byteCount }
        var out: [ModelShape.Tower] = []
        if bytes > 0 {
            out.append(ModelShape.Tower(name: "vision", bytes: bytes))
        }
        return out
    }

    // The vision tower ships as its own file here, and everything in an
    // mmproj serves the eye -- so the file's size IS the tower's weight.

    static func eye(_ mmproj: String?) -> [ModelShape.Tower] {
        var out: [ModelShape.Tower] = []
        if let mmproj,
           let size = (try? FileManager.default
               .attributesOfItem(atPath: mmproj))?[.size] as? Int {
            out.append(ModelShape.Tower(name: "vision", bytes: size))
        }
        return out
    }

    // A sibling "*mmproj*.gguf" in the weight's directory is the model's
    // Qwen3-VL vision tower; present -> the backend advertises vision.
    static func mmprojBeside(_ path: String) -> String? {
        let dir = (path as NSString).deletingLastPathComponent
        let names = (try? FileManager.default
            .contentsOfDirectory(atPath: dir)) ?? []
        return names
            .first { n in n.contains("mmproj")
                && (n.hasSuffix(".ggxf") || n.hasSuffix(".gguf")) }
            .map { n in dir + "/" + n }
    }

    public var mtpDrafts: Int {
        let raw = ProcessInfo.processInfo.environment["LLM_MTP_DRAFTS"]
        var out = 0
        if let raw, let n = Int(raw) {
            out = n
        } else {
            let gb = Double(engine.model.gguf.mapSize) / 1_073_741_824
            let trunk = engine.model.layers[0].ffnDown.type
            out = gb < 3 || Blocks.codebook(trunk) ? 0 : 2
        }
        return out
    }

    public func backend() -> MetalBackend {
        MetalBackend(engine: engine, tokenizer: tokenizer,
                     mmprojPath: mmprojPath)
    }
}

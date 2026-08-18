// Adapts the pure-Swift ternary BonsaiEngine (SIMD/CPU) to the AgentBackend
// seam, so ChatSession / Agent drive it exactly like the CoreML EngineBackend.
// A reference type (@unchecked Sendable): the ChatSession serializes every call,
// so the engine's mutable state is never touched concurrently.
import Foundation

public final class BonsaiBackend: AgentBackend, @unchecked Sendable {
    let engine: BonsaiEngine
    let tokenizer: Tokenizer
    private var savedMark: BonsaiEngine.Bookmark?

    public init(engine: BonsaiEngine, tokenizer: Tokenizer) {
        self.engine = engine
        self.tokenizer = tokenizer
    }

    public var eos: Int32 { tokenizer.eosId }
    public var position: Int { get async { engine.pos } }

    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text, addSpecial: true)
    }
    public func tokenBytes(_ id: Int32) -> [UInt8] { tokenizer.decodeBytes([id]) }
    public func text(_ ids: [Int32]) -> String { tokenizer.decode(ids) }

    public func reset() async { engine.reset() }
    public func useSampler(_ s: Sampler?) async { engine.sampler = s }
    public func extend(_ ids: [Int32]) async throws -> Int32 { engine.extend(ids) }
    public func decode(_ token: Int32) async throws -> Int32 { engine.decode(token) }

    public func mark() async throws { savedMark = engine.bookmark() }
    public func rewind() async throws {
        if let m = savedMark { engine.restore(m) }
    }

    struct State: BackendState { let bookmark: BonsaiEngine.Bookmark }
    public func saveState() async throws -> any BackendState {
        State(bookmark: engine.bookmark())
    }
    public func loadState(_ state: any BackendState) async throws {
        if let s = state as? State { engine.restore(s.bookmark) }
    }

    // Bytes, not a forward pass.

    public func serializeState(_ state: any BackendState) async -> Data {
        var out = Data()
        if let s = state as? State { out = engine.serialize(s.bookmark) }
        return out
    }

    public func deserializeState(_ data: Data) async throws
        -> any BackendState {
        let b = engine.deserialize(data)
        if b == nil {
            throw GGUFErr.parse("parked state is not this build's format")
        }
        return State(bookmark: b!)
    }

    // A pre-turn rollback point carries the state AND savedMark, so a rollback
    // (prefill cancel, or a title turn) leaves the next turn's rewind pointing
    // where it did rather than at the rolled-back turn -- matching
    // EngineBackend.Turn (the default checkpoint, saveState alone, drops it).
    struct Turn: BackendState {
        let bookmark: BonsaiEngine.Bookmark
        let mark: BonsaiEngine.Bookmark?
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
}

// A loaded Bonsai model: the ternary engine + a GGUF-driven tokenizer / chat
// template / sampler recs -- the self-contained GGUF counterpart of AneChat's
// CoreML directory (tokenizer.json + chat_template.jinja + generation_config).
public struct BonsaiChat {
    public let engine: BonsaiEngine
    public let tokenizer: Tokenizer
    public let chatTemplate: String
    public let samplingPresets: SamplingPresets

    // Minimal ChatML fallback when a GGUF ships no chat_template (rare).
    static let fallbackTemplate = """
        {% for message in messages %}<|im_start|>{{ message.role }}
        {{ message.content }}<|im_end|>
        {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
        {% endif %}
        """

    public init(ggufPath: String) throws {
        let m = try BonsaiModel(path: ggufPath)
        engine = BonsaiEngine(m)
        tokenizer = try Tokenizer(gguf: m.gguf)
        chatTemplate = m.gguf.string("tokenizer.chat_template")
            ?? BonsaiChat.fallbackTemplate
        samplingPresets = SamplingPresets.require(gguf: m.gguf,
                                                  path: ggufPath)
    }

    public func backend() -> BonsaiBackend {
        BonsaiBackend(engine: engine, tokenizer: tokenizer)
    }
}

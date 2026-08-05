// Adapts the pure-Swift Gemma4Engine (SIMD/CPU) to the AgentBackend seam, so
// ChatSession drives gemma-4 exactly as it drives the ternary Bonsai lineage
// and the CoreML EngineBackend.
//
// Two things differ from BonsaiBackend, both gemma facts rather than style:
// the tokenizer is the SentencePiece one (GemmaTokenizer, a separate code
// path -- F24), and a turn ends on a SET of ids rather than one (F23), which
// the seam carries as eosIds.
//
// A reference type (@unchecked Sendable): the ChatSession serializes every
// call, so the engine's mutable state is never touched concurrently.
import Foundation

public final class Gemma4Backend: AgentBackend, @unchecked Sendable {
    let engine: Gemma4Engine
    let tokenizer: GemmaTokenizer
    private var savedMark: Gemma4Engine.Bookmark?

    init(engine: Gemma4Engine, tokenizer: GemmaTokenizer) {
        self.engine = engine
        self.tokenizer = tokenizer
    }

    public var eos: Int32 { tokenizer.eosId }

    public var bosToken: String { tokenizer.bosToken }

    // The authoritative stop set: gemma ends a turn on three ids, and a loop
    // comparing against the scalar alone runs past <end_of_turn> (F23).
    public var eosIds: Set<Int32> { tokenizer.eosIds }

    public var position: Int { get async { engine.pos } }

    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text, addSpecial: true)
    }

    public func tokenBytes(_ id: Int32) -> [UInt8] {
        tokenizer.tokenBytes(id)
    }

    public func text(_ ids: [Int32]) -> String {
        tokenizer.decode(ids)
    }

    public func reset() async {
        engine.reset()
    }

    public func useSampler(_ s: Sampler?) async {
        engine.sampler = s
    }

    // A prefill-phase Stop leaves the loop early; surface it as
    // EngineError.stopped so ChatSession rolls the turn back, matching the
    // ternary and CoreML backends.
    public func extend(_ ids: [Int32]) async throws -> Int32 {
        let out = engine.extend(ids)
        if engine.shouldStop() { throw EngineError.stopped }
        return out
    }

    public func requestStop() { engine.requestStop() }
    public func shouldStop() -> Bool { engine.shouldStop() }

    public func supportsSoftTokens() async -> Bool { true }

    public func extendSoft(_ ids: [Int32],
                           spans: [SoftSpan]) async throws -> Int32 {
        let feed = SoftFeed(spans)
        let out = engine.extend(ids, softAt: { id in feed.row(id) })
        if engine.shouldStop() { throw EngineError.stopped }
        return out
    }

    public func decode(_ token: Int32) async throws -> Int32 {
        engine.decode(token)
    }

    public func mark() async throws {
        savedMark = engine.bookmark()
    }

    public func rewind() async throws {
        if let m = savedMark { engine.restore(m) }
    }

    struct State: BackendState {
        let bookmark: Gemma4Engine.Bookmark
    }

    public func saveState() async throws -> any BackendState {
        State(bookmark: engine.bookmark())
    }

    public func loadState(_ state: any BackendState) async throws {
        if let s = state as? State { engine.restore(s.bookmark) }
    }

    // A parked conversation restores from BYTES, never from a forward pass.
    // The protocol default returns empty Data, which reads as a warm cache
    // and is a cold one every time. [[never-reprefill]]

    public func serializeState(_ state: any BackendState) async -> Data {
        var out = Data()
        if let s = state as? State { out = engine.serialize(s.bookmark) }
        return out
    }

    public func deserializeState(_ data: Data) async throws
        -> any BackendState {
        guard let b = engine.deserialize(data) else {
            throw GGUFErr.parse("parked state is not this build's format")
        }
        return State(bookmark: b)
    }

    // A pre-turn rollback point carries the state AND savedMark, so a
    // rollback leaves the next turn's rewind pointing where it did rather
    // than at the rolled-back turn -- matching EngineBackend.Turn.
    struct Turn: BackendState {
        let bookmark: Gemma4Engine.Bookmark
        let mark: Gemma4Engine.Bookmark?
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

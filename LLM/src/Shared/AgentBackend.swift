import CoreML
import Foundation

// The backend seam the turn loops render and decode against, plus the message /
// metrics model they carry. `ChatSession` (and every compute backend --
// EngineBackend, MetalBackend, BonsaiBackend) speak this protocol; the concrete
// engine stays behind it. Async where the real backend is actor-isolated.

// A part of a multimodal user turn. A message carrying contentParts renders as
// template list-content (each .image emits the template's own vision triple);
// without them, its `content` string renders verbatim.
public enum ContentPart: Sendable {
    case text(String)
    case image
    case audio
    case video

    // For logs and for the default question; not markup, which is the
    // template's business.
    public var noun: String {
        switch self {
        case .text: "text"
        case .image: "image"
        case .audio: "audio"
        case .video: "video"
        }
    }
}

// One message in the conversation the template renders. `toolCalls` is set
// only on an assistant message that invoked tools; `reasoning` stays nil in
// history (the template strips past-turn thinking, so committing it would
// diverge the re-rendered prefix).
public struct AgentMessage: Sendable {
    public var role: String
    public var content: String
    public var contentParts: [ContentPart]?
    public var toolCalls: [AgentToolCall]
    public var reasoning: String?
    // Which tool a `tool` message answers. Qwen's template wraps the result
    // in a role-agnostic <tool_response> block and never asks; gemma's names
    // the tool in the block itself (`response:NAME{...}`) and falls back to
    // the literal "unknown" without this.
    public var name: String?

    public init(role: String, content: String,
                contentParts: [ContentPart]? = nil,
                toolCalls: [AgentToolCall] = [], reasoning: String? = nil,
                name: String? = nil) {
        self.role = role
        self.content = content
        self.contentParts = contentParts
        self.toolCalls = toolCalls
        self.reasoning = reasoning
        self.name = name
    }
}

// A tool call as carried in assistant history (name + arguments), the shape
// the chat_template serializes back into <tool_call> markup.
public struct AgentToolCall: Sendable {
    public var name: String
    public var arguments: [ToolArg]

    public init(name: String, arguments: [ToolArg]) {
        self.name = name
        self.arguments = arguments
    }
}

// Per-turn telemetry surfaced after a turn: the KV token count and how the last
// reply split between <think> reasoning and visible content.
public struct TurnMetrics: Sendable {
    public let ctx: Int
    public let thinkTokens: Int
    public let contentTokens: Int
    // Throughput for the status bar: prompt-prefill and generation tokens per
    // second. Default 0 for a path that does not time its runs.
    public let pp: Double
    public let tg: Double
    // WHY the final decode ended ("eos", "loop-breaker", "cancelled", ...),
    // so the app can explain an empty turn instead of showing silence. ""
    // while a turn is still streaming.
    public let endReason: String
    public init(ctx: Int, thinkTokens: Int, contentTokens: Int,
                pp: Double = 0, tg: Double = 0, endReason: String = "") {
        self.ctx = ctx
        self.thinkTokens = thinkTokens
        self.contentTokens = contentTokens
        self.pp = pp
        self.tg = tg
        self.endReason = endReason
    }
}

// The exact operations the turn loop needs from an inference backend, async
// where the real one is actor-isolated. `extend` appends onto the CURRENT
// state (no reset); `mark`/`rewind` snapshot and roll back the KV to the last
// mark; `position` is the live KV token count. A mock conforms to drive the
// loop with no model; EngineBackend adapts the real AneChat.
public protocol AgentBackend: Sendable {
    func encode(_ text: String) -> [Int32]
    func tokenBytes(_ id: Int32) -> [UInt8]
    func text(_ ids: [Int32]) -> String
    var eos: Int32 { get }
    // The FULL stop set. Gemma-4 ends a turn on three distinct ids
    // (<eos>, <end_of_turn>, and a channel terminator), so a loop comparing
    // against the scalar alone runs straight past the turn boundary into the
    // next one. Defaults to just `eos` for the single-stop lineages.
    var eosIds: Set<Int32> { get }
    func reset() async
    func useSampler(_ s: Sampler?) async
    func extend(_ ids: [Int32]) async throws -> Int32
    func mark() async throws
    func rewind() async throws
    var position: Int { get async }
    func decode(_ token: Int32) async throws -> Int32
    // Cooperative stop for a synchronous, non-suspending backend (Metal): the
    // app raises requestStop from the main actor while a forward holds the
    // ChatSession actor with no await point, so Task/AsyncStream cancellation
    // does not reach it. Prefill honors it; the decode loop polls shouldStop.
    // Default no-op / false -- actor backends (CoreML) already stop at their
    // await points via Task cancellation.
    func requestStop()
    func shouldStop() -> Bool
    // Committed-but-undelivered tokens (MTP speculative decode): tokens the
    // backend's state has already advanced through but decode() has not yet
    // handed out. The turn loop must not inject markup (</think>) while any
    // are pending -- it would land after tokens the transcript never saw.
    // Default 0 for non-speculative backends.
    func queuedCount() async -> Int
    // Fresh vision prefill: reset, then prefill `ids` (with each image's
    // <|image_pad|> already expanded to the tower's merged-token count) while
    // splicing the tower feats over each image span, returning the next-token
    // argmax. A text-only backend has no vision path (the default throws);
    // EngineBackend routes to Engine.prefillVision.
    func prefillVision(_ ids: [Int32], tiles: VisionTiles,
                       imageStarts: [Int], gridH: Int,
                       gridW: Int) async throws -> Int32
    // Vision prefill onto the CURRENT state (no reset): a later-turn image in
    // the continuation delta, its feats spliced at the running position so
    // prior turns stay in the KV. The default throws; EngineBackend routes to
    // Engine.prefillVisionExtend.
    func extendVision(_ ids: [Int32], tiles: VisionTiles,
                      imageStarts: [Int], gridH: Int,
                      gridW: Int) async throws -> Int32
    // Prefill onto the CURRENT state with attachments spliced in: `ids` is the
    // turn with every span's placeholder ALREADY expanded to its block, and
    // `spans` carries the tower rows to lay over those placeholder positions.
    //
    // SEPARATE from extendVision rather than a generalization of it, because
    // the two cannot express each other. extendVision speaks MLMultiArray
    // tiles on a square patch grid at ONE token rate; a native-resolution
    // tower emits a different count per image (following the aspect ratio,
    // known only after it runs), a video frame gets a smaller budget than a
    // still, and a turn may carry a second modality whose placeholder is a
    // different id entirely. The default throws; Gemma4Backend routes to
    // Gemma4Engine.extend(_:softAt:).
    func extendSoft(_ ids: [Int32], spans: [SoftSpan]) async throws -> Int32
    // Whether this backend can splice tower features at all. The app gates
    // the audio / video attach UI on it, so a backend without the path never
    // dead-ends a send.
    func supportsSoftTokens() async -> Bool
    // The begin-of-sequence SPELLING this model's template asks for by name.
    // Empty (the default) for a lineage whose template never mentions it;
    // gemma-4 opens every conversation with `{{- bos_token -}}` and drops a
    // token from position 0 without it.
    var bosToken: String { get }
    // Whether this backend can see images at all, and (at send time) the
    // tower's grid for tile sizing. Defaults: no. The app gates the whole
    // attach UI on supportsVision, so a text-only backend (today's Metal
    // 27B) never dead-ends a send.
    func supportsVision() async -> Bool
    func visionGrid() async -> VisionGrid?
    // Wall seconds the LAST vision prefill spent encoding tiles in the tower
    // (the phase before the LM sees a token), so the session can trace tower
    // time apart from prefill. 0 for a text-only backend.
    func visionEncodeSeconds() async -> Double
    // Snapshot / restore the WHOLE generation state (KV + recurrence +
    // position), distinct from the per-turn mark/rewind: this is the
    // multicontext primitive -- park one conversation's state and resume
    // another over one loaded model.
    func saveState() async throws -> any BackendState
    func loadState(_ state: any BackendState) async throws
    // A pre-turn rollback point: the whole generation state PLUS the internal
    // rewind mark, so a cancelled turn's prefill is undone cleanly (state and
    // the mark that the next turn's rewind uses both restore). Default backs it
    // with saveState/loadState; EngineBackend adds the mark.
    func checkpoint() async throws -> any BackendState
    func rollback(_ state: any BackendState) async throws
    // Serialize / restore a state snapshot to bytes for on-disk persistence
    // (the precooked-prompt cache). Default no-op; EngineBackend backs it with
    // Engine.serialize / deserialize.
    func serializeState(_ state: any BackendState) async -> Data
    func deserializeState(_ data: Data) async throws -> any BackendState
}

// An opaque, backend-owned snapshot of the full generation state, held by a
// ChatContext. EngineBackend backs it with an Engine.Bookmark.
public protocol BackendState: Sendable {}

// A placeholder for backends without real persistence (mocks): the default
// deserializeState returns it, and loadState ignores an unrecognized state.
public struct NullBackendState: BackendState {}

public extension AgentBackend {
    var eosIds: Set<Int32> { [eos] }
    func requestStop() {}
    func shouldStop() -> Bool { false }
    func queuedCount() async -> Int { 0 }
    func supportsVision() async -> Bool { false }
    func supportsSoftTokens() async -> Bool { false }
    var bosToken: String { "" }
    func extendSoft(_ ids: [Int32], spans: [SoftSpan]) async throws -> Int32 {
        throw EngineError.missingModel("soft tokens")
    }
    func visionGrid() async -> VisionGrid? { nil }
    func visionEncodeSeconds() async -> Double { 0 }
    func serializeState(_ state: any BackendState) async -> Data { Data() }
    func deserializeState(_ data: Data) async throws -> any BackendState {
        NullBackendState()
    }
    func prefillVision(_ ids: [Int32], tiles: VisionTiles,
                       imageStarts: [Int], gridH: Int,
                       gridW: Int) async throws -> Int32 {
        throw EngineError.missingModel("vision")
    }
    func extendVision(_ ids: [Int32], tiles: VisionTiles,
                      imageStarts: [Int], gridH: Int,
                      gridW: Int) async throws -> Int32 {
        throw EngineError.missingModel("vision")
    }
    func checkpoint() async throws -> any BackendState {
        try await saveState()
    }
    func rollback(_ state: any BackendState) async throws {
        try await loadState(state)
    }
}

// Adapts a AneChat (Engine actor + Tokenizer) to AgentBackend. Engine ops hop
// the actor; tokenizer ops are pure/sync. A reference type so its rewind
// bookmark survives the serialized awaits; @unchecked (that bookmark is the
// only mutable state, touched only under those awaits).
public final class EngineBackend: AgentBackend, @unchecked Sendable {
    private let chat: AneChat
    private var savedMark: Engine.Bookmark?

    public init(_ chat: AneChat) {
        self.chat = chat
    }

    public var eos: Int32 { chat.tokenizer.eosId }

    public var position: Int {
        get async { await chat.engine.position() }
    }

    public func encode(_ text: String) -> [Int32] {
        chat.tokenizer.encode(text, addSpecial: true)
    }

    public func tokenBytes(_ id: Int32) -> [UInt8] {
        chat.tokenizer.decodeBytes([id])
    }

    public func text(_ ids: [Int32]) -> String {
        chat.tokenizer.decode(ids)
    }

    public func reset() async {
        await chat.engine.reset()
    }

    public func useSampler(_ s: Sampler?) async {
        await chat.engine.useSampler(s)
    }

    public func extend(_ ids: [Int32]) async throws -> Int32 {
        try await chat.engine.ingestBatched(ids)
    }

    public func mark() async throws {
        savedMark = try await chat.engine.bookmark()
    }

    public func rewind() async throws {
        if let mark = savedMark { try await chat.engine.restore(mark) }
    }

    public func decode(_ token: Int32) async throws -> Int32 {
        try await chat.engine.decode(token)
    }

    // The CoreML Engine's nonisolated lock-backed stop (raised while a
    // synchronous prefill holds the actor). Decode already stops at the await
    // hop, so shouldStop mainly serves prefill parity with the Metal backend.
    public func requestStop() { chat.engine.requestStop() }
    public func shouldStop() -> Bool { chat.engine.shouldStop() }

    public func queuedCount() async -> Int {
        await chat.engine.specQueueCount()
    }

    public func supportsVision() async -> Bool {
        await chat.engine.supportsVision()
    }

    public func visionGrid() async -> VisionGrid? {
        await chat.engine.visionGrid()
    }

    public func visionEncodeSeconds() async -> Double {
        await chat.engine.visionEncodeSeconds()
    }

    public func prefillVision(_ ids: [Int32], tiles: VisionTiles,
                              imageStarts: [Int], gridH: Int,
                              gridW: Int) async throws -> Int32 {
        try await chat.engine.prefillVision(
            ids, tiles: tiles, imageStarts: imageStarts,
            gridH: gridH, gridW: gridW)
    }

    public func extendVision(_ ids: [Int32], tiles: VisionTiles,
                             imageStarts: [Int], gridH: Int,
                             gridW: Int) async throws -> Int32 {
        try await chat.engine.prefillVisionExtend(
            ids, tiles: tiles, imageStarts: imageStarts,
            gridH: gridH, gridW: gridW)
    }

    // A full-state snapshot backed by the oracle-proven
    // Engine.bookmark/restore.
    struct State: BackendState { let bookmark: Engine.Bookmark }

    public func saveState() async throws -> any BackendState {
        State(bookmark: try await chat.engine.bookmark())
    }

    public func loadState(_ state: any BackendState) async throws {
        if let s = state as? State {
            try await chat.engine.restore(s.bookmark)
        }
    }

    // A pre-turn rollback point carries BOTH the engine state and savedMark:
    // a cancelled turn may have already re-marked (rewind past this turn's raw
    // <think>), so restoring the state without the mark would leave the next
    // turn's rewind pointing into rolled-back territory.
    struct Turn: BackendState {
        let bookmark: Engine.Bookmark
        let mark: Engine.Bookmark?
    }

    public func checkpoint() async throws -> any BackendState {
        Turn(bookmark: try await chat.engine.bookmark(), mark: savedMark)
    }

    public func rollback(_ state: any BackendState) async throws {
        if let t = state as? Turn {
            try await chat.engine.restore(t.bookmark)
            savedMark = t.mark
        }
    }

    public func serializeState(_ state: any BackendState) async -> Data {
        var out = Data()
        if let s = state as? State {
            out = await chat.engine.serialize(s.bookmark)
        }
        return out
    }

    public func deserializeState(_ data: Data) async throws
        -> any BackendState {
        State(bookmark: try await chat.engine.deserialize(data))
    }
}

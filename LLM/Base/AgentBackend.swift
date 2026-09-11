import Foundation


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
    public let overrun: Int
    public let stopToken: Int32?
    public init(ctx: Int, thinkTokens: Int, contentTokens: Int,
                pp: Double = 0, tg: Double = 0, endReason: String = "",
                overrun: Int = 0, stopToken: Int32? = nil) {
        self.ctx = ctx
        self.thinkTokens = thinkTokens
        self.contentTokens = contentTokens
        self.pp = pp
        self.tg = tg
        self.endReason = endReason
        self.overrun = overrun
        self.stopToken = stopToken
    }
}

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
    func requestStop()
    func shouldStop() -> Bool
    // Committed-but-undelivered tokens (MTP speculative decode): tokens the
    // backend's state has already advanced through but decode() has not yet
    // handed out. The turn loop must not inject markup (</think>) while any
    // are pending -- it would land after tokens the transcript never saw.
    // Default 0 for non-speculative backends.
    func queuedCount() async -> Int
    func drainSpecTurn() -> SpecTurn?
    func useSpeculation(_ on: Bool)
    // Prefill onto the CURRENT state with attachments spliced in: `ids` is the
    // turn with every span's placeholder ALREADY expanded to its block, and
    // `spans` carries the tower rows to lay over those placeholder positions.
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
    func supportsVision() async -> Bool
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
    func drainSpecTurn() -> SpecTurn? { nil }
    func useSpeculation(_ on: Bool) {}
    func supportsVision() async -> Bool { false }
    func supportsSoftTokens() async -> Bool { false }
    var bosToken: String { "" }
    func extendSoft(_ ids: [Int32], spans: [SoftSpan]) async throws -> Int32 {
        throw EngineError.missingModel("soft tokens")
    }
    func serializeState(_ state: any BackendState) async -> Data { Data() }
    func deserializeState(_ data: Data) async throws -> any BackendState {
        NullBackendState()
    }
    func checkpoint() async throws -> any BackendState {
        try await saveState()
    }
    func rollback(_ state: any BackendState) async throws {
        try await loadState(state)
    }
}

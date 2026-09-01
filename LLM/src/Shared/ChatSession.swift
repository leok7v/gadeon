import CoreML
import CryptoKit
import Foundation
import os

private let chatLog = Logger(subsystem: "io.github.leok7v.gadeon",
                             category: "chat")

// Holds the tool-call structure grammar while it is armed. The sampler's mask
// closure and the decode loop both touch it, but only under the ChatSession
// actor's serialized awaits (the mask runs inside backend.decode, the advance
// after it returns), so the shared reference is @unchecked rather than locked.
final class GrammarGate: @unchecked Sendable {
    private let vocab: GrammarVocab
    // Mask only the structural tags (skip the whole-vocab walk at free-run name
    // / value positions), so the cost is the cheap tag bytes, not the values.
    private let structuralOnly: Bool
    var grammar: Grammar?
    // Armed IS "a live grammar exists": arming sets it, death and disarm
    // nil it, so no parallel flag can drift from it.
    var armed: Bool { grammar != nil }

    init(vocab: GrammarVocab, structuralOnly: Bool) {
        self.vocab = vocab
        self.structuralOnly = structuralOnly
    }

    func disarm() {
        grammar = nil
    }

    func mask(_ logits: inout [Float]) {
        if let g = grammar {
            g.maskLogits(vocab, &logits, structuralOnly: structuralOnly)
        }
    }
}

// One tool round surfaced to a UI: fired once when the call is detected
// (result nil, the tool is about to run) and once more with the result --
// the exact string laid into the KV as the tool response, so what the UI
// shows IS what the model saw. For the budget-nudge round (the per-turn
// tool cap was spent) the single event carries the nudge text as its
// result. `resolved` is the tool that actually ran; nil means the emitted
// name grounded as unknown.
public struct ToolRoundEvent: Sendable {
    public let round: Int
    public let name: String
    public let resolved: String?
    public let params: [ToolArg]
    public let result: String?
}

// The multi-turn chat loop, shared by the app and gadeon-cli: render the whole
// conversation through the chat_template, bring the engine to the generation
// point by re-prefilling ONLY the delta since the last turn, and stream the
// reply. It is the Agent's render + O(delta) bookmark continuation MINUS the
// tool loop and grammar mask.
//
// The template STRIPS <think> from PAST turns, so the raw tokens left in the
// KV (still carrying that turn's <think>) diverge from how the template now
// re-renders it. Each turn marks the KV at the assistant-open; the next
// rewinds there and re-prefills the prior STRIPPED answer + the new user turn
// + gen prompt. Every answer is re-processed once, so total work is
// O(conversation), not O(n^2). The backend is injected (real Engine / mock).
public actor ChatSession {
    private let backend: any AgentBackend
    private let template: String
    // The model's per-mode sampling matrix (thinking on/off x text/vision).
    // installTurnSampler selects the cell for each turn -- reasoning-effort and
    // whether the turn is a vision turn both steer it, per the model card.
    private let presets: SamplingPresets
    private let vocabSize: Int
    // Reasoning-effort on/off as the TEMPLATE spells it. Whether flipping it
    // live reaches the model is the template's choice, not ours: Qwen renders
    // it into every generation prompt, so the next turn takes it; gemma-4
    // spends it on a marker in the FIRST system turn, which is already in the
    // KV and is subtracted from every later delta, so a flip there changes
    // nothing until the conversation restarts.
    private var enableThinking: Bool
    // Reasoning the template opened, which this turn must spend none of --
    // what reaches a conversation already under way when enableThinking
    // cannot. The loop closes the channel at its first token and the sampler
    // takes the instruct cell, so the turn decodes prose the way a
    // never-opened one does instead of at reasoning temperature.
    private var suppressReasoning = false
    private var reasoningEffort: String?
    // Per-turn safety cap on decoded tokens (a small model can loop without
    // ever emitting EOS). Default unbounded -- the app stops a run by
    // cancelling the consuming task, which the decode loop below observes via
    // Task.isCancelled.
    private let maxTokens: Int
    // HARD cap on tokens spent INSIDE <think> (0 = unbounded). On overflow the
    // loop injects </think> IMMEDIATELY, so a no-EOS thinking runaway cannot
    // hang the turn (the guarantee a soft cap can't make -- it waits for a
    // newline). var: the app derives both caps from the model's MEASURED
    // decode rate (a time budget), which arrives after init and refines per
    // turn -- see setReasoningCaps.
    private var maxReasoning: Int
    // SOFT cap on <think> tokens (0 = off). Once reasoning passes it, the loop
    // ends <think> at the NEXT paragraph break (a trailing newline) rather
    // than mid-sentence -- a cleaner cut than the hard cap. Rides the same
    // inject- </think> path as Quick Answer.
    private var softReasoningCap: Int
    // 0 keeps the Sampler's own default; the turn restarts the stream either way.
    private let samplerSeed: UInt64
    // Overthink penalty (off unless lambda > 0): a per-token logit bias on the
    // curated single-token branch-openers, installed only while thinking to
    // shorten the chain of thought. See Sampler.overthinkMarkers.
    private let overthinkTokens: Set<Int32>
    private let overthinkLambda: Float
    // Single-token wire specials, exempt from the sampler's repetition
    // penalties: a tool round re-emits them verbatim, and penalizing them
    // rotted the 4B's markup round over round (spaced tags, arg soup, EOS
    // mid-call by round 5). See Sampler.penaltyExempt.
    private let wireTokens: Set<Int32>
    // Optional light-tool support (wikipedia_query, ...): the runner dispatches
    // tool calls and its specs render into the fresh turn's system prompt. nil
    // (airplane mode, or no tool model) disables tools -- no specs, no detection.
    private var runner: (any ToolRunner)?
    // The advertised specs ARE the runner's (ToolRunner.tools is a pure
    // computed list), so the pair cannot fall out of lockstep across
    // setTools.
    private var toolSpecs: [ToolSpec] { runner?.tools ?? [] }
    // Tool-call structure grammar: while a <tool_call> is open it masks the
    // sampler so only well-formed <function=NAME>...</function> markup can be
    // emitted. It does NOT pin the function name -- the name free-runs, and
    // resolveTool fuzzy-matches it afterwards, so a call to an ABSENT tool
    // survives as an unknown-tool grounding rather than being steered onto a
    // wrong real tool. STRUCTURAL is the default (masks only the tag
    // positions, measured == off speed) -- see "WHY THE TOOL GRAMMAR
    // DEFAULTS TO STRUCTURAL" at the bottom of this file.
    // LLM_TOOL_GRAMMAR=off disarms; "full" also walks the whole vocab at
    // free-run value bytes (~7x decode, debugging only).
    enum GrammarMode { case off, full, structural }
    private static let grammarMode: GrammarMode = {
        switch ProcessInfo.processInfo.environment["LLM_TOOL_GRAMMAR"] {
        case "full", "1": return .full
        case "off", "0": return .off
        default: return .structural
        }
    }()
    static let digitsExempt =
        ProcessInfo.processInfo.environment["LLM_DIGIT_EXEMPT"] != "off"
    private var gate: GrammarGate?
    private var grammarVocab: GrammarVocab?
    // The tool-call dialect the model's template speaks. XML (<function=NAME>
    // ...) for QwenPaw/3.5/3.6; JSON ({"name":.., "arguments":..}) for older
    // Qwen3 (the dense 1.7B). PARSING is dialect-agnostic (Tools.parse sniffs
    // the body), so only the STRUCTURAL grammar keys off this: its mask forces
    // <function=> tags, which would forbid the '{' a JSON call opens with, so
    // it arms only for XML. The template is the authority (an arch could ship
    // either), detected by the literal it renders calls with.
    private let toolDialectXML: Bool
    // The markup this model's template speaks, derived from it (ChatWire).
    private let wire: ChatWire
    // Whether the template numbers attachments itself (see
    // numbersAttachments); when it does, ChatSession must not do it too.
    private let templateNumbers: Bool
    // The turn's base sampler (no grammar mask). The masked variant is swapped
    // in only while a <tool_call> is open -- a logitMask forces the engine to
    // materialize the full logit vector on CPU every token, so it must not ride
    // the whole turn.
    private var turnSampler: Sampler?
    // A per-turn cap on tool rounds -- a runaway guard only; an agentic
    // answer legitimately chains several lookups (search -> fetch -> fetch).
    static let maxToolRounds = 10
    // Cap on the BYTES of one still-open <tool_call> body. While a call is
    // open NOTHING streams (both channels hold back), the loop breaker never
    // trips on varied content, and the app's maxTokens is unbounded -- so a
    // model free-running a huge value (observed: the 4B writing an endless
    // SVG inside a hallucinated `code` call) is an invisible forever-decode.
    // Real values here are an expression / query / url; the largest observed
    // recoverable call body was ~4.5KB. Distinct from fetch_url's 16KB page
    // limit: that bounds a tool RESULT, which is prefilled, never decoded --
    // this bounds only what the model TYPES inside a call, and its real cost
    // is the silent stall before it fires (~2 minutes at 8KB on the 4B), so
    // it should stay small; a future big-value tool (write_file-style
    // content) would need a per-tool cap instead.
    static let maxOpenCallBytes = 8192
    // Laid as the tool response when the round budget is spent, so a looping
    // model produces a final answer from what it has instead of an empty turn.
    static let toolBudgetNudge =
        "You have reached the tool-call limit for this turn. Do not call any "
        + "more tools; answer the user now using the information you already "
        + "have."
    private var history: [AgentMessage]
    // An APPEND-ONLY mirror of the tokens resident in the KV up to the current
    // turn mark (a clean turn boundary, before the generation prompt). Each turn
    // rewinds the engine to that mark and APPENDS the delta -- the previous
    // stripped answer + the new user turn -- rendered in isolation; this array
    // grows by that delta. Empty == a fresh conversation. It is NOT diffed
    // against a whole re-render; it exists
    // only so park/resume/serialize can round-trip the KV token sequence.
    private var committed: [Int32]
    // The <|image_pad|> token the chat template emits per image, expanded to the
    // tower's merged-token run at prefill; resolved once at init.
    private let imagePadId: Int32
    // An image conversation: some history message carries contentParts (only
    // vision turns do), so text follow-ups keep selecting the VL sampling
    // presets. Derived from history so park/resume and a restored context
    // agree with the conversation they actually hold -- a stored latch did
    // not survive resume().
    private var visionContext: Bool {
        history.contains { m in m.contentParts != nil }
    }
    // Set by requestQuickAnswer (the app's Quick Answer button): on the next
    // decode step, if STILL inside <think>, inject </think> to jump to the
    // answer. Rides the same inThink-guarded path as the maxReasoning cap, so
    // a press that lands after the model already left reasoning is a no-op.
    // Cleared each turn.
    private var forceEndThink = false
    private var metaTurn = false
    // Whether the last generation prompt leaves decoding INSIDE the reasoning
    // region. Set per turn from the rendered gen prompt, because the template
    // is the authority and the three shipped answers differ: Qwen3.5 opens the
    // marker and leaves it open; the dense Qwen3 bakes a CLOSED empty block
    // regardless of enable_thinking (the flag alone would misroute its whole
    // answer to the reasoning channel); gemma-4 says nothing and opens its own
    // channel as its first decoded output. See ChatWire.startsInReasoning.
    private var genStartsThink = false
    // NO markup is assumed anywhere in this file beyond the model's OWN
    // reasoning / tool-call wire (<think>, <tool_call>): everything laid
    // into the KV -- openers, think prefixes, tool responses -- comes out
    // of the template, by rendering with and without add_generation_prompt
    // and taking suffixes. The template is the single authority on chat
    // markup.
    // How the LAST turn ended. Both rolled-back cases restore the state and
    // drop the user turn from history identically; they differ only in what
    // the app owes the user. `.stopped` produced nothing, so its two
    // transcript bubbles go with it. `.answerless` spent tool rounds the user
    // WATCHED and then reached no final answer -- deleting that erases
    // seconds of visible work and the question behind it, so those bubbles
    // stay and the transcript says what happened. Cleared at every turn.
    public enum TurnOutcome: String, Sendable {
        case answered, stopped, answerless
    }

    public private(set) var turnOutcome: TurnOutcome = .answered

    public var turnRolledBack: Bool { turnOutcome != .answered }
    public private(set) var lastMetrics: TurnMetrics
    // The system message split for the precooked-prefix cache: `systemStable`
    // is everything whose bytes survive across sessions (persona, tools tier,
    // locale) and is what precook persists; `systemTail` is the short dynamic
    // suffix (date/time) that would poison the cache stamp, laid separately
    // after a restore. The rendered system message is stable + tail.
    private let systemStable: String
    private let systemTail: String
    // Structured session trace (the app's debug view + transcript log):
    // every render / prefill / decode / tool round / rewind surfaces as a
    // TraceEvent through this sink. nil (the default) costs nothing.
    private var traceSink: (@Sendable (TraceEvent) -> Void)?
    // The in-flight prefix prime/cook. Turns AWAIT it before touching the
    // engine: actor reentrancy interleaves long operations at their awaits,
    // so without this gate a send during the cook runs its prefill
    // concurrently with the cook's over the same engine (two interleaved
    // prefills, a torn KV).
    private var priming: Task<Void, Never>?

    public init(backend: any AgentBackend, template: String, system: String,
                systemTail: String = "", vocabSize: Int,
                presets: SamplingPresets = .greedy,
                enableThinking: Bool = false,
                reasoningEffort: String? = nil, maxTokens: Int = .max,
                maxReasoning: Int = 0, softReasoningCap: Int = 0,
                overthink: Float = 0, seed: UInt64 = 0,
                runner: (any ToolRunner)? = nil) {
        self.backend = backend
        self.template = template
        let wire = ChatWire.derive(template)
        self.wire = wire
        self.openTag = Array(wire.toolCallOpen.utf8)
        self.closeTag = Array(wire.toolCallClose.utf8)
        self.thinkOpen = Array(wire.reasoningOpen.utf8)
        self.thinkClose = Array(wire.reasoningClose.utf8)
        self.toolDialectXML = template.contains("<function=")
        self.templateNumbers = ChatSession.numbersAttachments(template)
        self.systemStable = system
        self.systemTail = systemTail
        self.runner = runner
        self.presets = presets
        self.vocabSize = max(vocabSize, 1)
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.maxTokens = maxTokens
        self.maxReasoning = maxReasoning
        self.softReasoningCap = softReasoningCap
        self.overthinkLambda = overthink
        self.samplerSeed = seed
        // Build the per-model marker set: encode each curated marker (bare and
        // space-prefixed, the two BPE shapes) and keep only single-token hits
        // -- a multi-token marker cannot be biased by a per-token subtraction.
        var markers: Set<Int32> = []
        if overthink != 0 {
            for marker in Sampler.overthinkMarkers {
                for variant in [marker, " " + marker] {
                    let ids = backend.encode(variant)
                    if ids.count == 1 { markers.insert(ids[0]) }
                }
            }
        }
        self.overthinkTokens = markers
        var specials: Set<Int32> = []
        for marker in wire.penaltyExemptCandidates {
            let ids = backend.encode(marker)
            if ids.count == 1 { specials.insert(ids[0]) }
        }
        // Digits ride along: the Qwen pretokenizer splits every numeral
        // into single-digit tokens, so copying a tool result into prose
        // RE-EMITS the digit tokens the call and result just used --
        // presence 1.5-2.0 then steers the copy off by a digit (tool said
        // 3753.75, the 0.8B wrote $3,753.74; 2925.26 drifted to $2,938).
        // NOT the comma: exempting it freed a ",000,000,..." group cycle
        // to run to the loop breaker (observed), and a comma appears once
        // per group, so penalties never flip it the way they flip digits.
        for ch in "0123456789." where ChatSession.digitsExempt {
            let ids = backend.encode(String(ch))
            if ids.count == 1 { specials.insert(ids[0]) }
        }
        self.wireTokens = specials
        // Only a marker the tokenizer carries as ONE token can be the pad.
        // A model without it (gemma-4 spells its image marker <|image|>)
        // encodes the spelling as a RUN of ordinary pieces, and taking the
        // first id would make expandPads rewrite every occurrence of that
        // ordinary token -- with count 0, deleting it from the prompt.
        let padIds = backend.encode("<|image_pad|>")
        self.imagePadId = padIds.count == 1 ? padIds[0] : -1
        self.history = [AgentMessage(role: "system",
                                     content: system + systemTail)]
        self.committed = []
        self.lastMetrics = TurnMetrics(ctx: 0, thinkTokens: 0,
                                       contentTokens: 0)
    }

    // Flip reasoning-effort for the next and later turns (the Thinking
    // button).
    public func setThinking(_ on: Bool) {
        enableThinking = on
    }

    // Spend no reasoning from the next turn on, whatever the template already
    // laid down. Takes effect mid-conversation, which setThinking cannot on a
    // template that carries the flag in its system block.
    public func setSuppressReasoning(_ on: Bool) {
        suppressReasoning = on
    }

    public func setReasoningEffort(_ level: String?) {
        reasoningEffort = level
    }

    // Swap the tool runner live (the Web-access / airplane toggle); takes effect
    // on the next turn. nil disables tools (no specs rendered, no detection).
    public func setTools(_ runner: (any ToolRunner)?) {
        self.runner = runner
    }

    // Install (or clear) the trace sink. Fires on the session's actor; the
    // app hops each event to its own context.
    public func setTrace(_ sink: (@Sendable (TraceEvent) -> Void)?) {
        traceSink = sink
    }

    // Vision capability, straight from the backend: the app gates the whole
    // image-attach UI on supportsVision and sizes tiles from visionGrid at
    // send, so it never touches a concrete engine.
    public func supportsVision() async -> Bool {
        await backend.supportsVision()
    }

    public func visionGrid() async -> VisionGrid? {
        await backend.visionGrid()
    }

    // Whether this model takes tower features directly (image / clip / video
    // as soft tokens), which is a different capability from supportsVision:
    // that one promises the MLMultiArray tile path, this one the SoftSpan
    // path, and no backend today offers both.
    public func supportsSoftTokens() async -> Bool {
        await backend.supportsSoftTokens()
    }

    private func trace(_ kind: TraceEvent.Kind, from t0: Date? = nil,
                       until t1: Date? = nil, ctx: Int = -1, tokens: Int = 0,
                       summary: String, text: String = "",
                       image: Data? = nil) {
        if let sink = traceSink {
            let now = Date()
            sink(TraceEvent(kind: kind, t0: t0 ?? now, t1: t1 ?? now, ctx: ctx,
                            tokens: tokens, summary: summary, text: text,
                            image: image))
        }
    }

    // Quick Answer: end reasoning at the next decoded token. Effective only
    // while still inside <think> (the loop re-checks at the injection point),
    // so a late call is a no-op.
    public func requestQuickAnswer() {
        forceEndThink = true
    }

    // Retune the thinking budget: soft = end <think> at the next paragraph
    // break, hard = inject immediately (the runaway backstop). Takes effect
    // on the next decode iteration, so a change mid-conversation applies to
    // the running and following turns.
    public func setReasoningCaps(soft: Int, hard: Int) {
        softReasoningCap = soft
        maxReasoning = hard
    }

    // Raise the backend's stop flag -- the reliable lever for a synchronous,
    // non-suspending backend (Metal) whose forward does not observe Task
    // cancellation. nonisolated so the app raises it WHILE the actor is busy in
    // a forward; the backend clears it at the next extend. Prefill honors it
    // (rolls the turn back); the decode loop polls it via backend.shouldStop().
    public nonisolated func requestStop() {
        backend.requestStop()
    }

    // Drop the conversation back to just the system prompt and clear the
    // engine, for a New Chat. The next reply prefills from scratch.
    public func reset() async {
        history = Array(history.prefix(1))
        committed = []
        attachmentCounts = [:]
        await backend.reset()
        lastMetrics = TurnMetrics(ctx: 0, thinkTokens: 0, contentTokens: 0)
        trace(.reset, ctx: 0, summary: "new chat")
    }

    // A parked conversation: the backend's full generation state (KV +
    // recurrence + position) plus the resumable conversation (history + the
    // tokens resident in the KV). Switch between conversations over ONE loaded
    // model by parking the current one and resuming another; neither loses its
    // KV/SSM/GDN memory. Also the unit persisted by the precooked-prompt
    // cache.
    public struct ChatContext: Sendable {
        let state: any BackendState
        let history: [AgentMessage]
        let committed: [Int32]
        let attachments: [String: Int]
    }

    // Snapshot the live conversation so another can run, then be resumed.
    public func park() async throws -> ChatContext {
        ChatContext(state: try await backend.saveState(),
                    history: history, committed: committed,
                    attachments: attachmentCounts)
    }

    // Make a parked conversation live again: restore its engine state and its
    // history/committed, so the next reply continues exactly where it left
    // off.
    public func resume(_ context: ChatContext) async throws {
        try await backend.loadState(context.state)
        history = context.history
        committed = context.committed
        attachmentCounts = context.attachments
    }

    // A codable envelope of the resumable conversation; the engine state rides
    // alongside as an appended blob (json-length prefix, json, then state
    // bytes).
    private struct ContextMeta: Codable {
        let stamp: String
        let committed: [Int32]
        let roles: [String]
        let contents: [String]
        // Without this a resumed conversation restarts at Picture 1 and
        // collides with numbers already sitting in its own KV.
        let attachments: [String: Int]
    }

    // Persist the live conversation to `url`, tagged with `stamp` (a content
    // hash of the fixed prefix). The precooked-prompt cache saves the state
    // after prefilling the system + tools prompt once, so a later launch
    // RESTORES it instead of re-prefilling hundreds of tokens.
    public func saveContext(to url: URL, stamp: String) async throws {
        let context = try await park()
        let stateData = await backend.serializeState(context.state)
        let meta = ContextMeta(
            stamp: stamp, committed: context.committed,
            roles: context.history.map { $0.role },
            contents: context.history.map { $0.content },
            attachments: context.attachments)
        var out = Data()
        let json = try JSONEncoder().encode(meta)
        var len = Int64(json.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(json)
        out.append(stateData)
        try out.write(to: url)
    }

    // Restore a persisted conversation from `url` when its stamp matches;
    // returns whether it loaded (false -> the caller should prefill fresh,
    // e.g. because the prefix changed and the stamp no longer matches).
    public func loadContext(from url: URL, stamp: String) async throws -> Bool {
        let data = try Data(contentsOf: url)
        let base = data.startIndex
        var len: Int64 = 0
        withUnsafeMutableBytes(of: &len) { dst in
            _ = data.copyBytes(to: dst, from: base ..< base + 8)
        }
        let jlen = Int(Int64(littleEndian: len))
        let meta = try JSONDecoder().decode(
            ContextMeta.self,
            from: data.subdata(in: base + 8 ..< base + 8 + jlen))
        var loaded = false
        if meta.stamp == stamp {
            let state = try await backend.deserializeState(
                data.subdata(in: base + 8 + jlen ..< data.endIndex))
            try await backend.loadState(state)
            var restored: [AgentMessage] = []
            for i in 0 ..< meta.roles.count {
                restored.append(AgentMessage(role: meta.roles[i],
                                             content: meta.contents[i]))
            }
            history = restored
            committed = meta.committed
            attachmentCounts = meta.attachments
            loaded = true
        }
        return loaded
    }

    // ---- precooked system-prefix cache (TTFT) ---------------------------
    // The rendered system + tools prefix is byte-identical for every
    // conversation of a session configuration, and prefilling it is most of
    // the time-to-first-token on a big model (the Metal 27B pays ~50s).
    // precook() runs that prefill ONCE on an empty session and persists the
    // engine state via saveContext; prime() restores it and lays only the
    // dynamic tail. Both leave the session in the identical post-system
    // state (whole system block laid, mark at its end), so the first turn
    // appends only its own user delta.

    // The closed render of the full system block, and its stable prefix
    // (the render cut where the dynamic tail's bytes begin). The Qwen
    // template raises on any render without a user message, so the block is
    // derived by SUBTRACTION like toolContinuation: render [system, probe]
    // and drop the probe's own render as a suffix. Empty on a template where
    // that subtraction does not hold -- precook/prime then no-op. An empty
    // tail makes the whole block the prefix (the date bakes at cook time).
    // What the template emits before any message: gemma-4 opens a system turn
    // whenever enable_thinking is set, even with no messages at all. Every
    // Qwen-shaped template renders nothing here.
    private func leadingBlock(thinking: Bool) -> String {
        (try? renderPrompt(
            template: template, messages: [], tools: [],
            addGenerationPrompt: false,
            enableThinking: thinking,
            reasoningEffort: thinking ? reasoningEffort : nil,
            bosToken: backend.bosToken)) ?? ""
    }

    private func systemRenders() -> (full: String, prefix: String) {
        let probe = AgentMessage(role: "user", content: "x")
        let both = (try? renderPrompt(
            template: template, messages: [history[0], probe],
            tools: toolSpecs, addGenerationPrompt: false,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
        // NO tools and NO reasoning, so these bytes are the user TURN alone:
        // a probe carrying either is not a suffix of `both` on a template that
        // moves them (Qwen3.8), and the cache switches itself off.
        var probeText = (try? renderPrompt(
            template: template, messages: [probe], tools: [],
            addGenerationPrompt: false, enableThinking: false,
            reasoningEffort: nil,
            bosToken: backend.bosToken)) ?? ""
        // A lead surviving thinking=false belongs to `full`, not to the turn.
        let lead = leadingBlock(thinking: false)
        if !lead.isEmpty, probeText.hasPrefix(lead) {
            probeText = String(probeText.dropFirst(lead.count))
        }
        var full = ""
        if !both.isEmpty, !probeText.isEmpty, both.hasSuffix(probeText) {
            full = String(both.dropLast(probeText.count))
        }
        if full.isEmpty {
            chatLog.error("no precook prefix under this template")
        }
        var prefix = full
        if !systemTail.isEmpty,
           let at = full.range(of: systemTail, options: .backwards) {
            prefix = String(full[..<at.lowerBound])
        }
        return (full, prefix)
    }

    private static func stamp(_ s: String) -> String {
        SHA256.hash(data: Data(s.utf8))
            .map { b in String(format: "%02x", b) }.joined()
    }

    // The identity of what precook would persist; empty when there is none.
    public var precookStamp: String {
        let prefix = systemRenders().prefix
        return prefix.isEmpty ? "" : ChatSession.stamp(prefix)
    }

    // Prefill the stable prefix on an EMPTY session and persist its state,
    // then complete the system block (dynamic tail) and set the mark at its
    // end. The token-boundary check guards the BPE seam between prefix and
    // tail; a mismatch lays the whole block in one piece instead (the file
    // stays valid -- prime re-verifies against its own render).
    public func precook(to url: URL) async throws {
        let r = systemRenders()
        if committed.isEmpty && !r.prefix.isEmpty {
            let t0 = Date()
            let prefixIds = backend.encode(r.prefix)
            let fullIds = backend.encode(r.full)
            await backend.reset()
            _ = try await backend.extend(prefixIds)
            committed = prefixIds
            try await saveContext(to: url, stamp: ChatSession.stamp(r.prefix))
            if fullIds.count >= prefixIds.count,
               Array(fullIds.prefix(prefixIds.count)) == prefixIds {
                let tail = Array(fullIds[prefixIds.count...])
                if !tail.isEmpty { _ = try await backend.extend(tail) }
            } else {
                await backend.reset()
                _ = try await backend.extend(fullIds)
            }
            committed = fullIds
            try await backend.mark()
            trace(.prefill, from: t0, ctx: await backend.position,
                  tokens: fullIds.count,
                  summary: String(format: "precooked system prefix (%.1fs)",
                                  Date().timeIntervalSince(t0)))
        }
    }

    // The app's one entry: prime from the cache, cook on a miss, optionally
    // clearing the engine first (New Chat). Runs as the `priming` task that
    // every turn awaits, so an early send waits the cook out instead of
    // interleaving with it.
    public func primeOrCook(at url: URL, resetFirst: Bool = false) {
        // Chain on any cook already in flight rather than replacing it: the
        // two would drive the ONE engine at once, and the second's reset would
        // free state the first is still computing over.
        let running = priming
        priming = Task {
            await running?.value
            if resetFirst { await reset() }
            if await prime(from: url) == false {
                try? await precook(to: url)
            }
        }
    }

    // Take this session off the engine: stop an in-flight prime/cook and wait
    // for it to unwind. Sessions SHARE one backend, so a session being
    // replaced must be off the engine before its successor resets it --
    // otherwise the reset frees KV pages the outgoing cook's command buffer is
    // still reading through the bindless page table, and the GPU faults. The
    // stop flag is what actually lands (a Metal forward is synchronous and
    // passes no cancellation point); the await is what makes "off the engine"
    // true rather than merely asked for.
    public func endPriming() async {
        backend.requestStop()
        priming?.cancel()
        await priming?.value
        priming = nil
    }

    // The GPU work the APP does BEFORE a turn -- the vision and audio towers
    // an attachment is encoded through -- drives the SAME engine as the prime
    // but does not travel the turn path, so it never reaches the gate inside
    // runSoftTurn. It has to await this first.
    //
    // MEASURED, an iPhone SE 2nd generation tapping the picture sample at
    // launch: the precook prefill was still running, the system was down to
    // 71 MB free, and the vision tower asked for its first pipeline. The Metal
    // shader compiler is a separate XPC service, so it was jetsammed
    // mid-compile and the pipeline build trapped. Our own footprint claimed
    // 1.9 GB of headroom throughout.
    //
    // Unlike endPriming this does NOT stop the prime, it waits for it: the
    // caller wants the engine next, not instead of.
    public func awaitPriming() async {
        await priming?.value
    }

    // Restore a precooked prefix and complete the system block with the
    // current dynamic tail. false -> no usable cache (absent, a changed
    // prompt/tools/template, a backend without byte state, or a token-
    // boundary mismatch); the caller cooks or takes the plain fresh turn.
    public func prime(from url: URL) async -> Bool {
        let t0 = Date()
        let r = systemRenders()
        var ok = false
        if !r.prefix.isEmpty {
            ok = (try? await loadContext(
                from: url, stamp: ChatSession.stamp(r.prefix))) == true
        }
        if ok {
            let fullIds = backend.encode(r.full)
            let n = committed.count
            // The restored state must hold exactly the committed prefix and
            // that prefix must open today's render -- a backend whose
            // serialization is a no-op, or a BPE seam shift, fails here.
            ok = await backend.position == n && n <= fullIds.count
                && Array(fullIds.prefix(n)) == committed
            if ok, n < fullIds.count {
                ok = (try? await backend.extend(
                    Array(fullIds[n...]))) != nil
            }
            if ok {
                committed = fullIds
                ok = (try? await backend.mark()) != nil
            }
            history = [AgentMessage(role: "system",
                                    content: systemStable + systemTail)]
            if !ok {
                // A torn restore must not leave half a prefix in the KV.
                await backend.reset()
                committed = []
            }
        }
        if ok {
            trace(.prefill, from: t0, ctx: await backend.position,
                  tokens: committed.count,
                  summary: String(format: "primed from cache (%.2fs)",
                                  Date().timeIntervalSince(t0)))
        }
        return ok
    }

    // Stream one turn's reply. The returned stream yields visible text as
    // whole UTF-8 scalars complete; cancelling the consuming task ends the run
    // (the Stop button). lastMetrics carries the KV size and think/content
    // split once the stream finishes.
    public nonisolated func reply(
        _ user: String,
        onReasoning: (@Sendable (String) -> Void)? = nil,
        onTool: (@Sendable (String) -> Void)? = nil,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)? = nil
    ) -> AsyncStream<String> {
        AsyncStream { cont in
            let task = Task {
                await self.runTurn(user, onReasoning: onReasoning,
                                   onTool: onTool,
                                   onToolRound: onToolRound) { piece in
                    cont.yield(piece)
                }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    // Stream one image-first vision turn's reply. `tiles` are the tower patch
    // tensors (thumbnail + any detail tiles); gridH/gridW/tokensPerImage come
    // from the loaded tower. The image is prefilled on THIS turn and carried,
    // so later plain reply() turns still see it.
    public nonisolated func replyVision(
        _ user: String, tiles: [MLMultiArray], gridH: Int, gridW: Int,
        tokensPerImage: Int, numberImages: Bool = false,
        thumbnail: Data? = nil,
        onReasoning: (@Sendable (String) -> Void)? = nil,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)? = nil
    ) -> AsyncStream<String> {
        let boxed = VisionTiles(tiles)
        return AsyncStream { cont in
            let task = Task {
                await self.runVisionTurn(
                    user, tiles: boxed, gridH: gridH, gridW: gridW,
                    tokensPerImage: tokensPerImage, numberImages: numberImages,
                    thumbnail: thumbnail,
                    onReasoning: onReasoning, onToolRound: onToolRound
                ) { piece in cont.yield(piece) }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    // Stream one turn carrying attachments already encoded into SoftSpans
    // (image / clip / video, in any combination). `parts` is the COMPLETE
    // content list, text included, so a caller can interleave positionally --
    // [.text("is "), .image, .text(" inside "), .image, .text("?")] renders
    // in that order on both shipped templates. `user` is the plain string
    // history and the trace carry.
    //
    // The attachment is prefilled on THIS turn and carried, so later plain
    // reply() turns still see it -- and a LATER attachment appends its own
    // delta rather than re-encoding the earlier ones.
    public nonisolated func replySoft(
        _ user: String, parts: [ContentPart], spans: [SoftSpan],
        labelled: Bool = true,
        onReasoning: (@Sendable (String) -> Void)? = nil,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)? = nil
    ) -> AsyncStream<String> {
        AsyncStream { cont in
            let task = Task {
                await self.runSoftTurn(
                    user, parts: parts, spans: spans, labelled: labelled,
                    onReasoning: onReasoning, onToolRound: onToolRound
                ) { piece in cont.yield(piece) }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    // A short title for the CURRENT conversation, generated on THIS model with
    // no second model and no disturbance to the live chat. The conversation is
    // already in the KV, so this appends a brief title instruction, decodes a
    // few words, then rolls the whole turn back: checkpoint carries the state
    // AND the turn mark, so restore() drops the appended tokens and leaves the
    // next real turn's rewind pointing exactly where it did. Thinking + tools
    // are forced off and the trace is muted for the throwaway turn. Empty on an
    // empty conversation or any failure, so the caller keeps its instant title.
    // The caller must hold the engine (no concurrent send) for the duration.

    // What a title turn appends to the generation prompt so the decode
    // starts PAST the reasoning channel. Asked of the wire rather than
    // assumed: the three shipped templates hand over different states --
    // gemma-4 emits nothing, Qwen3.5 opens the marker and leaves it open,
    // the dense Qwen3 already bakes the closed block -- wanting both
    // markers, the closing one, and nothing, respectively.

    static func titleSeed(_ gen: String, _ wire: ChatWire) -> String {
        var out = ""
        if !wire.closesReasoning(gen), wire.opensReasoning(gen) {
            out = wire.reasoningClose
        }
        return out
    }

    private func oneShot(_ instruction: String, stopAfter: Int,
                         _ label: String) async -> String {
        await priming?.value
        var raw = ""
        if history.count > 1, let save = try? await backend.checkpoint() {
            let began = Date()
            let savedThinking = enableThinking
            let savedRunner = runner
            let savedHistory = history
            let savedCommitted = committed
            let savedMetrics = lastMetrics
            let savedOutcome = turnOutcome
            let savedSink = traceSink
            let savedSuppress = suppressReasoning
            enableThinking = false
            suppressReasoning = true
            runner = nil
            traceSink = nil
            metaTurn = true
            for await piece in reply(instruction) {
                raw += piece
                if raw.count > stopAfter { backend.requestStop() }
            }
            metaTurn = false
            let spent = lastMetrics
            enableThinking = savedThinking
            suppressReasoning = savedSuppress
            runner = savedRunner
            traceSink = savedSink
            history = savedHistory
            committed = savedCommitted
            lastMetrics = savedMetrics
            turnOutcome = savedOutcome
            try? await backend.rollback(save)
            Diag.shared.report(String(
                format: "%@ raw=%@ think=%d content=%d in %.1fs", label,
                raw.debugDescription, spent.thinkTokens, spent.contentTokens,
                Date().timeIntervalSince(began)))
        }
        return raw
    }

    public func makeTitle() async -> String {
        let raw = await oneShot(ChatSession.titleInstruction, stopAfter: 160,
                                "makeTitle")
        return ChatSession.cleanTitle(raw)
    }

    public func makeFollowup() async -> String {
        let raw = await oneShot(ChatSession.followUpInstruction,
                                stopAfter: 280, "makeFollowup")
        let hint = ChatSession.cleanFollowup(raw)
        return alreadyAsked(hint) ? "" : hint
    }

    private func alreadyAsked(_ hint: String) -> Bool {
        let candidate = ChatSession.bagOfWords(hint)
        var repeated = candidate.isEmpty
        for m in history where m.role == "user" && !repeated {
            let prior = ChatSession.bagOfWords(m.content)
            let union = candidate.union(prior).count
            repeated = union > 0
                && Double(candidate.intersection(prior).count)
                    / Double(union) >= ChatSession.echoOverlap
        }
        return repeated
    }

    static let echoOverlap = 0.6

    static func bagOfWords(_ s: String) -> Set<String> {
        Set(s.lowercased()
            .split(whereSeparator: { c in !c.isLetter && !c.isNumber })
            .map(String.init))
    }

    // Name the SUBJECT. Left to itself a model will happily title a turn by
    // its shape rather than its content -- "Audio Clip Reply", "Picture 1
    // Description" -- which reads the same for every spoken or attached
    // conversation and so tells the user nothing when they come back to a
    // list of them.
    static let titleInstruction =
        "In a few words, give a short title for this conversation, naming "
        + "what it is ABOUT. Never mention audio, pictures, recordings, "
        + "attachments or their labels, and never say that anything was "
        + "spoken or attached. At most four words, no punctuation, no "
        + "quotes. Reply with the title only."

    // The first line, trimmed of quotes / markup / a "Title:" prefix, capped to
    // a handful of words.
    static func withoutMarkup(_ raw: String) -> String {
        var out = ""
        var rest = Substring(raw)
        while let lt = rest.firstIndex(of: "<") {
            let tail = rest[lt...]
            if let gt = tail.firstIndex(of: ">") {
                out += rest[..<lt]
                rest = tail[tail.index(after: gt)...]
            } else {
                out += rest
                rest = rest[rest.endIndex...]
            }
        }
        return out + rest
    }

    static func cleanTitle(_ raw: String) -> String {
        let line = ChatSession.withoutMarkup(raw)
            .split(whereSeparator: \.isNewline)
            .first.map(String.init) ?? ""
        let trimmed = line.trimmingCharacters(
            in: CharacterSet(charactersIn: " \t\"'`.*_#"))
        var body = trimmed
        if let r = body.range(of: "title:",
                              options: [.caseInsensitive, .anchored]) {
            body = String(body[r.upperBound...])
                .trimmingCharacters(in: .whitespaces)
        }
        var words: [String] = []
        var chars = 0
        for word in body.split(separator: " ").prefix(6)
        where chars + word.count <= 40 {
            words.append(String(word))
            chars += word.count + 1
        }
        let prose = words.contains { word in
            word.contains(where: { c in c.isLetter })
        }
        let deliberating = ChatSession.deliberation.contains { phrase in
            body.lowercased().contains(phrase)
        }
        let usable = prose && words.count > 1 && !deliberating
        return usable ? words.joined(separator: " ") : ""
    }

    static let deliberation = [
        "the user wants", "the user is asking", "the user would",
        "the conversation is about", "short title for", "thinking process",
    ]

    static let followUpInstruction =
        ProcessInfo.processInfo.environment["GADEON_FOLLOWUP"]
            ?? defaultFollowUpInstruction

    static let defaultFollowUpInstruction =
        "Pick one specific detail from your last answer that the user would "
        + "most want to know more about, and write the question they would "
        + "type to ask about it. One sentence ending in a question mark. "
        + "Never repeat a question already asked. Output only the question."

    static let followupMin = 12
    static let followupMax = 140

    static func cleanFollowup(_ raw: String) -> String {
        let lines = ChatSession.withoutMarkup(raw)
            .split(whereSeparator: \.isNewline)
            .map { line in line.trimmingCharacters(in: .whitespaces) }
            .filter { line in !line.isEmpty }
        let asked = lines.filter { line in line.contains("?") }
        var out = ""
        if asked.count == 1, let line = asked.first {
            out = ChatSession.oneQuestion(line)
        }
        return out
    }

    private static let assistantVoice = [
        "would you like", "do you want", "shall i", "can i help",
        "is there anything", "let me know", "should i", "like me to",
        "your last answer",
    ]

    private static func oneQuestion(_ line: String) -> String {
        var body = line
        if let mark = body.firstIndex(of: "?") {
            body = String(body[...mark])
        }
        if let colon = body.range(of: ": "),
           !body[..<colon.lowerBound].contains("?"),
           body.distance(from: body.startIndex, to: colon.lowerBound) <= 40 {
            body = String(body[colon.upperBound...])
        }
        body = body.trimmingCharacters(
            in: CharacterSet(charactersIn: " \t\"'`*_-#>0123456789."))
        let lower = body.lowercased()
        let markup = body.contains(where: { c in "{}<>\"".contains(c) })
        let aboutTheUser = lower.hasPrefix("the user")
            || lower.hasPrefix("user ")
        let assistant = ChatSession.assistantVoice.contains { phrase in
            lower.contains(phrase)
        }
        let usable = !aboutTheUser && !assistant && !markup
            && body.hasSuffix("?")
            && body.count >= ChatSession.followupMin
            && body.count <= ChatSession.followupMax
        return usable ? body : ""
    }

    // A text turn: append the user message, seed the KV by rendering ONLY this
    // turn's delta and appending it (no whole-history re-render, no prefix
    // diff), then decode. Engine errors end the turn quietly; a prefill Stop
    // rolls it fully back.
    private func runTurn(_ user: String,
                         onReasoning: (@Sendable (String) -> Void)?,
                         onTool: (@Sendable (String) -> Void)?,
                         onToolRound: (@Sendable (ToolRoundEvent) -> Void)?,
                         _ yield: @Sendable (String) -> Void) async {
        await priming?.value
        let saved = await enterTurn()
        trace(.user, ctx: await backend.position,
              summary: String(user.prefix(80)), text: user)
        history.append(AgentMessage(role: "user", content: user))
        await installTurnSampler(vision: visionContext)
        await runSeed(vision: nil, soft: [], numberImages: false, saved: saved,
                      onReasoning: onReasoning, onTool: onTool,
                      onToolRound: onToolRound, yield)
    }

    // A turn carrying tower features. It seeds like every other turn -- the
    // delta render emits the template's own placeholders and extendSoft
    // splices the rows over them -- so nothing about the continuation model
    // changes; only where the embeddings come from at those positions.
    private func runSoftTurn(
        _ user: String, parts: [ContentPart], spans: [SoftSpan],
        labelled: Bool,
        onReasoning: (@Sendable (String) -> Void)?,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)?,
        _ yield: @Sendable (String) -> Void) async {
        await priming?.value
        let saved = await enterTurn()
        let rows = spans.reduce(0) { sum, span in sum + span.rows }
        trace(.user, ctx: await backend.position,
              summary: String(user.prefix(80))
                  + " [\(spans.count) attachment(s), \(rows) soft]",
              text: user)
        history.append(AgentMessage(role: "user", content: user,
                                    contentParts: numbered(parts,
                                                            labelled)))
        await installTurnSampler(vision: true)
        await runSeed(vision: nil, soft: spans, numberImages: false,
                      saved: saved, onReasoning: onReasoning, onTool: nil,
                      onToolRound: onToolRound, yield)
    }

    // A vision turn: append the user message as list-content (a placeholder per
    // tile, then the text), then seed the same way -- the delta render carries
    // the image, and extendVision splices its feats. Turn one seeds fresh; a
    // LATER image rewinds to the prior mark and the delta is just [previous
    // stripped answer, this image turn], so earlier turns (and earlier images)
    // stay in the KV and each image encodes exactly once.
    private func runVisionTurn(
        _ user: String, tiles: VisionTiles, gridH: Int, gridW: Int,
        tokensPerImage: Int, numberImages: Bool, thumbnail: Data?,
        onReasoning: (@Sendable (String) -> Void)?,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)?,
        _ yield: @Sendable (String) -> Void) async {
        await priming?.value
        let saved = await enterTurn()
        trace(.user, ctx: await backend.position,
              summary: String(user.prefix(80))
                  + " [\(tiles.tiles.count) tile(s)]", text: user,
              image: thumbnail)
        var parts: [ContentPart] = []
        for _ in 0 ..< tiles.tiles.count { parts.append(.image) }
        parts.append(.text(user))
        history.append(AgentMessage(role: "user", content: user,
                                    contentParts: parts))
        await installTurnSampler(vision: true)
        let v = (tiles: tiles, gridH: gridH, gridW: gridW,
                 tokensPerImage: tokensPerImage)
        await runSeed(vision: v, soft: [], numberImages: numberImages,
                      saved: saved, onReasoning: onReasoning, onTool: nil,
                      onToolRound: onToolRound, yield)
    }

    // Shared turn body: seed the KV via the delta render, decode, and -- when
    // tools are active and the model emits a tool call -- run it, append its
    // result, re-seed, and decode again, up to maxToolRounds. A prefill Stop
    // rolls the turn back. `vision` non-nil splices this turn's image (tool
    // rounds are always text).
    private func runSeed(
        vision: (tiles: VisionTiles, gridH: Int, gridW: Int,
                 tokensPerImage: Int)?,
        soft: [SoftSpan],
        numberImages: Bool, saved: SavedTurn,
        onReasoning: (@Sendable (String) -> Void)?,
        onTool: (@Sendable (String) -> Void)?,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)?,
        _ yield: @Sendable (String) -> Void) async {
        let first = await seedOnce(fresh: committed.isEmpty,
                                   numberImages: numberImages, vision: vision,
                                   soft: soft)
        if first.stopped {
            await rollbackTurn(saved)
        } else {
            let pp = first.pp
            var round = 0
            var pagedURL: String? = nil
            var paged = 0
            var spent: [String: String] = [:]
            var step = await decodeStep(seed: first.seed, pp: pp,
                                        onReasoning: onReasoning, yield)
            while let call = step.pending, round < ChatSession.maxToolRounds {
                let resolved = ChatSession.resolveTool(
                    call.functionName, toolSpecs.map { spec in spec.name })
                // Sequential offset-pages of ONE url are a single logical
                // lookup -- each extends the previous fetch rather than
                // starting a new one, so the chain costs one round. The
                // per-chain cap (8 pages ~ 128KB at the default limit)
                // keeps runaway paging from spinning past the budget.
                let url = resolved == "fetch_url"
                    ? call.params.first { p in
                        ["url", "link", "u"].contains(p.name)
                    }?.value
                    : nil
                if url != nil && url == pagedURL && paged < 8 {
                    paged += 1
                } else {
                    round += 1
                    paged = 0
                }
                pagedURL = url
                onTool?(call.functionName)
                onToolRound?(ToolRoundEvent(
                    round: round, name: call.functionName,
                    resolved: resolved, params: call.params, result: nil))
                trace(.toolCall, summary: call.functionName
                          + (resolved == call.functionName || resolved == nil
                             ? "" : " -> \(resolved!)"),
                      text: wire.toolCallOpen + call.rawBlock
                          + wire.toolCallClose)
                toolLog("tool call \(round): \(call.functionName) "
                    + call.params.map { "\($0.name)=\($0.value)" }
                        .joined(separator: " ")
                    + " | raw="
                    + call.rawBlock.replacingOccurrences(of: "\n", with: "\\n"))
                let toolT0 = Date()
                let signature = ChatSession.callSignature(
                    resolved ?? call.functionName,
                    sanitizedArgs(call, resolved))
                var result = spent[signature].map { prior in
                    ChatSession.repeatedCall(prior)
                } ?? ""
                if result.isEmpty {
                    result = await runTool(call, resolved: resolved)
                    spent[signature] = result
                } else {
                    toolLog("tool call \(round): REFUSED as an exact repeat")
                }
                onToolRound?(ToolRoundEvent(
                    round: round, name: call.functionName,
                    resolved: resolved, params: call.params, result: result))
                // from: toolT0 makes the event's duration the tool's own
                // wall time (network fetch, calculator, ...), distinct from
                // the prefill that follows.
                trace(.toolResult, from: toolT0,
                      summary: "\(resolved ?? call.functionName) -> "
                          + "\(result.count) chars",
                      text: result)
                toolLog("tool result \(round): \(result.count) chars: "
                    + result.prefix(160))
                let seed = await toolContinuation(
                    call: call, resolved: resolved,
                    preamble: step.preamble, result: result)
                // Keep publishing the turn's real prefill rate: a tool round's
                // decodeStep would otherwise land pp 0 in lastMetrics, and the
                // LAST step of the turn wins the status bar.
                step = await decodeStep(seed: seed, pp: pp,
                                        onReasoning: onReasoning, yield)
            }
            // Tool budget spent with a call still pending: lay that call with a
            // stop-nudge as its response and generate ONE final answer from what
            // the model has, rather than committing nothing. decodeStep commits
            // the answer when it lands; if the model STILL only emits another
            // call, keep whatever it said before it (usually empty) -- there is
            // nothing more to give it.
            if let call = step.pending {
                let resolved = ChatSession.resolveTool(
                    call.functionName, toolSpecs.map { spec in spec.name })
                onToolRound?(ToolRoundEvent(
                    round: round + 1, name: call.functionName,
                    resolved: resolved, params: call.params,
                    result: ChatSession.toolBudgetNudge))
                trace(.toolResult, summary: "tool budget spent -> nudge",
                      text: ChatSession.toolBudgetNudge)
                let seed = await toolContinuation(
                    call: call, resolved: resolved, preamble: step.preamble,
                    result: ChatSession.toolBudgetNudge)
                step = await decodeStep(seed: seed, pp: pp,
                                        onReasoning: onReasoning, yield)
                if step.pending != nil {
                    history.append(AgentMessage(role: "assistant",
                                                content: step.preamble))
                }
            }
            // A turn that committed NOTHING must not leave an EMPTY assistant
            // message behind. The next delta renders it as a bare
            // "<turn>model<turn|>", and a model asked to continue from a turn
            // where it said nothing answers with token salad -- which then
            // enters history itself, so every later turn inherits it and the
            // conversation never recovers (only New Chat clears it). MEASURED:
            // a Stop pressed after the prefill but before the first token, then
            // 1936 tokens of multilingual noise on the following turn.
            //
            // Rolling back is what a turn that produced nothing IS, and it
            // reuses the path a prefill Stop already takes -- including the
            // flag the app reads to drop the two transcript bubbles. Decode
            // Stops that DID produce text are untouched: they keep their
            // partial answer.
            let empty = history.last.map { last in
                last.role == "assistant" && last.content.isEmpty
            } ?? false
            if empty {
                await rollbackTurn(saved, why: ChatSession.userStopped(
                    lastMetrics.endReason) ? .stopped : .answerless)
            }
        }
    }

    // One delta seed (this turn's, or a tool round's continuation): render +
    // prefill the delta. Returns the first token to decode, the pp rate over
    // just that delta, and whether a prefill Stop aborted it.
    private func seedOnce(
        fresh: Bool, numberImages: Bool,
        vision: (tiles: VisionTiles, gridH: Int, gridW: Int,
                 tokensPerImage: Int)?,
        soft: [SoftSpan]
    ) async -> (seed: Int32, pp: Double, stopped: Bool) {
        let t0 = Date()
        var seed = backend.eos
        var added = 0
        var stopped = false
        do {
            let r = try await seedDelta(fresh: fresh,
                                        numberImages: numberImages,
                                        vision: vision, soft: soft)
            seed = r.seed
            added = r.added
        } catch EngineError.stopped {
            stopped = true
        } catch {
            seed = backend.eos
        }
        // The tower encode is the leading phase of a vision seed; surface it
        // as its own trace event and exclude it from the prefill span, so pp
        // is the LM's own rate and the debug view shows where the wall went.
        var tower = 0.0
        if let vision, !stopped {
            tower = await backend.visionEncodeSeconds()
            if tower > 0 {
                trace(.vision, from: t0,
                      until: t0.addingTimeInterval(tower),
                      tokens: vision.tiles.tiles.count * vision.tokensPerImage,
                      summary: String(format: "%d tile(s), tower %.1fs",
                                      vision.tiles.tiles.count, tower))
            }
        }
        let sec = Date().timeIntervalSince(t0) - tower
        let pp = sec > 0 && added > 0 ? Double(added) / sec : 0
        trace(.prefill, from: t0.addingTimeInterval(tower),
              ctx: await backend.position, tokens: added,
              summary: stopped ? "stopped mid-prefill"
                               : String(format: "%.0f t/s", pp))
        return (seed, pp, stopped)
    }

    // Seed the KV for the current turn by rendering only the DELTA and appending
    // it: fresh -> the whole (small) [system, user] with a reset; otherwise just
    // [previous stripped assistant, new user] with a rewind to the prior mark.
    // For this template that delta is byte-identical to the tail of a whole
    // render (system emitted only when messages[0] is system; a past assistant
    // renders stripped), so no history is re-rendered and no prefix is diffed
    // BOTH the closed (mark-safe) part and
    // the generation prompt are the template's own output: the delta renders
    // with and without add_generation_prompt and splits at the closed
    // render's length -- nothing assumes what an opener or a think prefix
    // looks like. The mark lands at that boundary, so the next rewind drops
    // this turn's raw <think> and re-appends the stripped answer as a whole
    // block. Returns the first token to decode and the delta token count.
    private func seedDelta(
        fresh: Bool, numberImages: Bool,
        vision: (tiles: VisionTiles, gridH: Int, gridW: Int,
                 tokensPerImage: Int)?,
        soft: [SoftSpan]
    ) async throws -> (seed: Int32, added: Int) {
        // Non-fresh with only [system, user] in history = the first turn
        // over a precooked/primed prefix: the system block is already in the
        // KV, so the delta is the user turn alone.
        let deltaMsgs = fresh
            ? history
            : (history.count == 2
                ? [history[history.count - 1]]
                : [history[history.count - 2], history[history.count - 1]])
        // Tool specs render into the system message, so only on the fresh turn
        // (which includes system); a continuation delta starts with an
        // assistant/user message and the template's tool-injection breaks on a
        // non-system-first conversation. The specs are already in the KV.
        let tools = fresh ? toolSpecs : []
        var closedText = (try? renderPrompt(
            template: template, messages: deltaMsgs, tools: tools,
            addGenerationPrompt: false, enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            addVisionId: numberImages,
            bosToken: backend.bosToken)) ?? ""
        var fullText = (try? renderPrompt(
            template: template, messages: deltaMsgs, tools: tools,
            addGenerationPrompt: true, enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            addVisionId: numberImages,
            bosToken: backend.bosToken)) ?? ""
        // The delta model holds only while rendering [prev assistant, new
        // user] alone equals the tail of rendering the whole history. Gemma-4
        // breaks it: `enable_thinking` ALONE opens its system turn, so every
        // delta carries an empty "<|turn>system\n<|think|>\n<turn|>\n" the
        // fresh turn already laid, and laying it again splices one spurious
        // system turn into the KV per turn. Templates with no such block
        // render nothing here and the subtraction is a no-op.
        let lead = fresh ? "" : leadingBlock(thinking: enableThinking)
        if !lead.isEmpty, closedText.hasPrefix(lead), fullText.hasPrefix(lead) {
            closedText = String(closedText.dropFirst(lead.count))
            fullText = String(fullText.dropFirst(lead.count))
        }
        if fullText.isEmpty {
            chatLog.error("template rendered nothing for these variables")
        }
        if !fullText.hasPrefix(closedText) {
            // A template whose generation prompt is not a pure render suffix
            // breaks the delta-continuation model; mark after everything
            // rather than guess at its markup.
            chatLog.warning("generation prompt is not a render suffix")
        }
        var genText = fullText.hasPrefix(closedText)
            ? String(fullText.dropFirst(closedText.count)) : ""
        if metaTurn { genText += ChatSession.titleSeed(genText, wire) }
        genStartsThink = wire.startsInReasoning(genPrompt: genText,
                                                enabled: true)
        let expanded = Continuation.expandPads(
            backend.encode(closedText), pad: imagePadId,
            count: vision?.tokensPerImage ?? 0)
        // The two expansions never both apply: a soft turn's model spells no
        // <|image_pad|>, so imagePadId is -1 there and expandPads passes the
        // ids through untouched.
        let closed = soft.isEmpty ? expanded.ids
            : Continuation.expandSpans(expanded.ids, soft)
        let gen = backend.encode(genText)
        if fresh {
            // Conversation start: the engine is normally already empty (New
            // Chat reset it, or it was freshly built), so this reset is a
            // silent no-op. A NON-empty engine here means the append-only
            // continuation broke -- reset LOUDLY: edge AI owns the session
            // and never re-prefills from scratch like a stateless server.
            let stale = await backend.position
            if stale > 0 {
                let msgs = history.count
                chatLog.warning("RESET reprefill (history=\(msgs) msgs)")
                trace(.reset, ctx: 0,
                      summary: "fresh turn over stale state (\(stale) tok)")
            }
            await backend.reset()
            committed = []
        } else {
            try await backend.rewind()
            trace(.rewind, ctx: await backend.position,
                  summary: "rewind to turn mark")
        }
        trace(.render, tokens: closed.count + gen.count,
              summary: fresh ? "fresh (system + tools + user)"
                             : "delta (prev answer + user)",
              text: fullText)
        let afterHead: Int32
        if let vision {
            afterHead = try await backend.extendVision(
                closed, tiles: vision.tiles, imageStarts: expanded.starts,
                gridH: vision.gridH, gridW: vision.gridW)
        } else if !soft.isEmpty {
            afterHead = try await backend.extendSoft(closed, spans: soft)
        } else {
            afterHead = try await backend.extend(closed)
        }
        committed += closed
        try await backend.mark()
        let seed = gen.isEmpty ? afterHead : try await backend.extend(gen)
        return (seed, closed.count + gen.count)
    }

    // A conversation-global count per modality, so an attachment gets a name
    // the user (and the model) can refer to for the rest of the turn -- "is
    // Picture 1 inside Picture 2". Conversation-global rather than per-turn
    // because an attachment already laid into the KV can never be re-labelled:
    // the delta render only ever sees the CURRENT turn, so a name skipped now
    // is unreachable forever.
    private var attachmentCounts: [String: Int] = [:]

    // Qwen's own nouns, so a model whose template does the numbering itself
    // and one numbered here read the same to a user.
    private static func noun(_ part: ContentPart) -> String? {
        switch part {
        case .image: "Picture"
        case .audio: "Audio"
        case .video: "Video"
        case .text: nil
        }
    }

    // `parts` with a label ahead of each attachment, unless the TEMPLATE
    // numbers them itself. Gemma trims each text part, so the label
    // necessarily abuts its marker ("Picture 1:<|image|>") -- there is no way
    // to carry a separator across a part boundary.
    //
    // DICTATION passes labelled: false. A name earns its place on something
    // handed over to be referred to later; speech is the user talking, and
    // labelling it turns the turn into an exhibit -- shown "Audio 1:" the
    // model reasons about "two audio snippets ... presented as text
    // contextually" and answers ABOUT the recording instead of to the person.
    private func numbered(_ parts: [ContentPart],
                          _ labelled: Bool = true) -> [ContentPart] {
        var out: [ContentPart] = []
        for part in parts {
            if let noun = ChatSession.noun(part), !templateNumbers, labelled {
                let n = (attachmentCounts[noun] ?? 0) + 1
                attachmentCounts[noun] = n
                out.append(.text("\(noun) \(n): "))
            }
            out.append(part)
        }
        return out
    }

    // Whether the template numbers attachments on its own, DERIVED rather
    // than assumed: render the same two-image turn with add_vision_id on and
    // off and see whether it changed anything. Qwen renders "Picture 1: " and
    // must not be numbered twice; gemma ignores the flag entirely.
    private static func numbersAttachments(_ template: String) -> Bool {
        let msg = AgentMessage(role: "user", content: "x",
                               contentParts: [.image, .image, .text("x")])
        func render(_ vid: Bool) -> String {
            (try? renderPrompt(template: template, messages: [msg], tools: [],
                               addGenerationPrompt: false,
                               enableThinking: false,
                               addVisionId: vid)) ?? ""
        }
        let on = render(true)
        return !on.isEmpty && on != render(false)
    }

    // Pre-turn snapshot for a clean prefill-cancel rollback: the engine
    // checkpoint (state + rewind mark) plus the conversation bookkeeping. Cheap
    // -- the checkpoint is COW (shares completed KV pages). Clears the roll-back
    // flag for this turn.
    private struct SavedTurn {
        let checkpoint: (any BackendState)?
        let history: [AgentMessage]
        let committed: [Int32]
    }

    static func userStopped(_ endReason: String) -> Bool {
        endReason == "cancelled" || endReason == "stop"
    }

    private func enterTurn() async -> SavedTurn {
        turnOutcome = .answered
        runner?.beginTurn()
        let checkpoint = try? await backend.checkpoint()
        return SavedTurn(checkpoint: checkpoint, history: history,
                         committed: committed)
    }

    // Undo a turn: restore the engine (state + mark) and the conversation
    // bookkeeping, and record WHY, which is what the app keys its transcript
    // on. Decode-phase cancels do NOT come here -- they keep the partial
    // answer.
    private func rollbackTurn(_ saved: SavedTurn,
                              why: TurnOutcome = .stopped) async {
        if let checkpoint = saved.checkpoint {
            try? await backend.rollback(checkpoint)
        }
        history = saved.history
        committed = saved.committed
        turnOutcome = why
        trace(.rewind, ctx: await backend.position,
              summary: "turn rollback (\(why.rawValue))")
    }

    // Per-turn sampler: the model card's preset for this turn's mode (thinking
    // on/off x text/vision), plus the overthink bias on branch-openers only
    // while thinking.
    private static let greedyConfig: SamplerConfig = {
        var c = SamplerConfig.default
        c.temperature = 0
        return c
    }()

    private func installTurnSampler(vision: Bool) async {
        let reasons = enableThinking && !suppressReasoning
        var turnConfig = metaTurn
            ? ChatSession.greedyConfig
            : presets.select(thinking: reasons, vision: vision)
        // DRY is the degeneration guard, not a taste knob: the card presets
        // leave some cells penalty-free (thinking+vision runs presence 0,
        // repeat 1.0), and a small model there re-emits whole sentences
        // verbatim -- a cycle far longer than per-token penalties suppress.
        // DRY penalizes EXTENDING a verbatim repeat, exponentially in its
        // length, so it only ever bites on degeneration. A model config that
        // sets it explicitly still wins.
        if !turnConfig.setMask.contains(.dryMultiplier) {
            turnConfig.dryMultiplier = 0.8
        }
        if samplerSeed != 0 { turnConfig.seed = samplerSeed }
        var sampler = Sampler(vocabSize: vocabSize, config: turnConfig)
        sampler.penaltyExempt = wireTokens
        if overthinkLambda != 0 && reasons && !overthinkTokens.isEmpty {
            sampler.overthinkTokens = overthinkTokens
            sampler.overthinkLambda = overthinkLambda
        }
        // Base sampler carries NO grammar mask, so free generation (reasoning,
        // the answer) keeps the engine's fast path. The tool-call grammar is
        // swapped in only while a <tool_call> is open (installMask, on arm /
        // disarm) -- a logitMask forces full-vocab CPU logits every token, an
        // ~7x decode cost that must not span the whole turn.
        turnSampler = sampler
        if let gate = ensureGate() { gate.disarm() }
        await backend.useSampler(sampler)
    }

    // Swap the engine sampler in/out of grammar-masked mode. Called only when
    // the grammar arms or disarms (twice per tool call), so the extra await is
    // rare.
    private func installMask(_ on: Bool) async {
        if var sampler = turnSampler {
            if on, let gate {
                sampler.logitMask = { logits in gate.mask(&logits) }
            }
            await backend.useSampler(sampler)
        }
    }

    // The tool-call grammar gate for this session, built on first use from the
    // backend's per-token bytes (once -- the vocab is large). nil when the
    // grammar is disabled for debugging.
    private func ensureGate() -> GrammarGate? {
        var result: GrammarGate? = nil
        if ChatSession.grammarMode != .off && toolDialectXML {
            if grammarVocab == nil {
                var toks: [[UInt8]] = []
                toks.reserveCapacity(vocabSize)
                for id in 0 ..< vocabSize {
                    toks.append(backend.tokenBytes(Int32(id)))
                }
                grammarVocab = GrammarVocab(toks)
            }
            if let vocab = grammarVocab {
                let g = gate ?? GrammarGate(
                    vocab: vocab,
                    structuralOnly: ChatSession.grammarMode == .structural)
                gate = g
                result = g
            }
        }
        return result
    }

    // Advance the tool-call grammar by the just-decoded token. Arms lazily the
    // first time an unclosed <tool_call> is in the stream (replaying from that
    // tag recovers its state), then steps it per committed token; disarms when
    // the grammar dies (</tool_call> reached, or the model drove it invalid) so
    // free text resumes.
    private func advanceGrammar(_ gate: GrammarGate, _ token: Int32,
                                _ bytes: [UInt8]) {
        if gate.armed {
            gate.grammar?.advance(backend.tokenBytes(token))
            if gate.grammar?.dead ?? true { gate.grammar = nil }
        } else if let off = stillOpen(bytes) {
            let g = Grammar.toolCall()
            g.advance(Array(bytes[off...]))
            gate.grammar = g
        }
    }

    // The markers the decode loop matches on, out of the model's OWN template
    // (ChatWire). Qwen says <think> / <tool_call>, gemma-4 says
    // <|channel>thought / <|tool_call> and closes them asymmetrically; a
    // template that renders neither leaves the Qwen wire standing.
    private let openTag: [UInt8]
    private let closeTag: [UInt8]
    private let thinkOpen: [UInt8]
    private let thinkClose: [UInt8]
    // No trailing '>': the rotting 4B emitted "</function >", and the
    // spaced close is still a complete body. Stays literal -- it is a repair
    // for the Qwen XML call dialect specifically, not part of any wire.
    private static let closeFn = Array("</function".utf8)

    // Whether the decoded stream's LEADING bytes (past any whitespace) are a
    // re-opened reasoning marker: settled once they equal it (.isThink) or
    // diverge from it (.notThink); .pending while a partial prefix could still
    // complete it. For gemma this is the NORMAL path, not a stray: its
    // generation prompt never opens the channel, so the model always opens
    // <|channel>thought itself as its first output.
    // `.isThink` carries the byte index just PAST the opener, because a model
    // that emits its own opener has put markup in the stream and the caller
    // has to step over it -- the closer's own handling already does exactly
    // that (`emitted = at + thinkClose.count`).
    enum LeadingThink { case pending, isThink(Int), notThink }

    static func leadingThink(_ b: [UInt8], _ open: [UInt8]) -> LeadingThink {
        var i = 0
        while i < b.count && (b[i] == 0x20 || b[i] == 0x09
                              || b[i] == 0x0A || b[i] == 0x0D) { i += 1 }
        let rest = b.count - i
        var result = LeadingThink.notThink
        if rest == 0 {
            result = .pending
        } else {
            let n = min(rest, open.count)
            var j = 0
            while j < n && b[i + j] == open[j] { j += 1 }
            if j == open.count {
                result = .isThink(i + open.count)
            } else if j == n {
                result = .pending
            }
        }
        return result
    }

    // Length of the longest tail of b[..<n] that is a proper PREFIX of `pat`
    // -- bytes the stream must hold back because the next token may complete
    // the marker.
    private static func partialSuffix(_ b: [UInt8], _ pat: [UInt8],
                                      _ n: Int) -> Int {
        var hold = 0
        var len = min(pat.count - 1, n)
        while hold == 0 && len > 0 {
            var j = 0
            while j < len && b[n - len + j] == pat[j] { j += 1 }
            if j == len { hold = len }
            len -= 1
        }
        return hold
    }
    private func stillOpen(_ b: [UInt8]) -> Int? {
        var result: Int? = nil
        if let open = ChatSession.lastIndex(b, openTag),
           ChatSession.index(b, closeTag, open + openTag.count) == nil {
            result = open
        }
        return result
    }

    private static func index(_ b: [UInt8], _ pat: [UInt8],
                              _ from: Int) -> Int? {
        var result: Int? = nil
        var i = max(from, 0)
        let last = b.count - pat.count
        while result == nil && i <= last {
            var j = 0
            while j < pat.count && b[i + j] == pat[j] { j += 1 }
            if j == pat.count { result = i }
            i += 1
        }
        return result
    }

    private static func lastIndex(_ b: [UInt8], _ pat: [UInt8]) -> Int? {
        var result: Int? = nil
        var i = b.count - pat.count
        while result == nil && i >= 0 {
            var j = 0
            while j < pat.count && b[i + j] == pat[j] { j += 1 }
            if j == pat.count { result = i }
            i -= 1
        }
        return result
    }

    // Decode `seed` onward: stream visible pieces to `yield`, reasoning to
    // onReasoning, tally the <think>/content split, then commit the answer.
    // Shared by the text and vision turns; the delta is already in the KV
    // (seedDelta), `pp` is the prompt-prefill rate to publish.
    private func decodeStep(
        seed: Int32, pp: Double,
        onReasoning: (@Sendable (String) -> Void)?,
        _ yield: @Sendable (String) -> Void)
        async -> (pending: ToolCall?, preamble: String) {
        forceEndThink = false  // a prior turn's Quick Answer must not fire
        let toolsActive = runner != nil && !toolSpecs.isEmpty
        // A completed call leaves the grammar parked in its absorbing match
        // state, so the gate would stay armed into the NEXT decode step --
        // masking nothing (every byte legal) but keeping a non-nil logitMask
        // installed, which silently disables spec decode for the rest of the
        // turn. Each step is a fresh byte stream: start disarmed and let an
        // open <tool_call> re-arm.
        if let gate, gate.armed {
            gate.disarm()
            await installMask(false)
        }
        var pending: ToolCall? = nil
        let startCtx = await backend.position   // KV size after seeding
        var cur = seed
        var ids: [Int32] = []
        var bytes: [UInt8] = []
        // Incremental split state, all byte offsets into `bytes`, so each
        // token costs O(new bytes), not a rescan from position 0: `emitted`
        // is the stream watermark, closeAt the reasoning-close marker,
        // toolAt the answer-region tool-call opener (both out of the
        // template, see ChatWire); the *Search cursors make each
        // rolling marker search touch every byte once.
        var emitted = 0
        var closeAt: Int? = nil
        // Whether decoding begins INSIDE <think> is the template's call, not
        // the flag's: the gen prompt either opens <think> (reasoning on) or
        // bakes a closed empty <think></think> (reasoning none -- AND older
        // Qwen3, the dense 1.7B, whose template closes it even with thinking
        // on). A gen prompt that already closed the think decodes into content
        // regardless of enableThinking, else the model's whole answer streams
        // to the reasoning channel. A small model can still RE-OPEN <think> as
        // its FIRST output (seen after tool-heavy turns); leadingThink routes
        // that stray block to reasoning too, so only real content reaches
        // `yield`. thinkDecided holds the split until the leading bytes settle
        // to <think> or diverge; a <think> later in the answer stays content.
        let startsInThink = enableThinking && genStartsThink
        var inThinkRegion = startsInThink
        var thinkDecided = startsInThink
        var wsDone = !startsInThink
        var thinkSearch = 0
        var toolAt: Int? = nil
        // Outlives toolAt being cleared by the quoted-pair recovery below.
        var sawToolOpen = false
        var toolSearch = 0
        var toolCloseSearch = 0
        var toolReopenSearch = 0
        var think = 0
        var content = 0
        var steps = 0
        var stop = false
        var thinkRescues = 0
        let g0 = Date()
        // Publish pp immediately so the status bar shows it the moment the
        // reply starts; tg fills in as tokens land.
        lastMetrics = TurnMetrics(ctx: startCtx, thinkTokens: 0,
                                  contentTokens: 0, pp: pp, tg: 0)
        while !backend.eosIds.contains(cur) && !stop && !Task.isCancelled
              && !backend.shouldStop() {
            ids.append(cur)
            bytes.append(contentsOf: backend.tokenBytes(cur))
            // Step the grammar by this token BEFORE decoding the next, so the
            // sampler's mask sees the state through the current byte stream.
            // Install / remove the mask only when the armed state flips (a
            // <tool_call> opened or closed), keeping free generation on the
            // fast path.
            if toolsActive, let gate {
                let wasArmed = gate.armed
                advanceGrammar(gate, cur, bytes)
                if gate.armed != wasArmed { await installMask(gate.armed) }
            }
            // Reasoning-on starts in the think region (the <think> opener is
            // in the PROMPT). Reasoning-none starts in content, but a leading
            // model-emitted <think> re-opens it -- decided once, from the
            // settling leading bytes.
            if !thinkDecided {
                switch ChatSession.leadingThink(bytes, thinkOpen) {
                case .isThink(let past):
                    inThinkRegion = true
                    // The model emitted the opener itself (gemma opens
                    // <|channel>thought where Qwen's opener sits in the
                    // PROMPT and never reaches this stream). It is markup:
                    // step over it, or it streams into the reasoning
                    // disclosure as its first words.
                    emitted = past
                    wsDone = false          // skip the ws after its </think>
                    thinkDecided = true
                case .notThink:
                    thinkDecided = true
                case .pending:
                    break
                }
            }
            if inThinkRegion && closeAt == nil {
                closeAt = ChatSession.index(bytes, thinkClose,
                                            thinkSearch)
                if closeAt == nil {
                    thinkSearch = max(thinkSearch,
                        bytes.count - thinkClose.count + 1)
                }
            }
            let inThink = inThinkRegion && closeAt == nil
            if inThink { think += 1 } else { content += 1 }
            // Emit only newly completed UTF-8 scalars, so a multi-token emoji
            // streams whole. Reasoning streams up to the </think> marker (a
            // tail that could be the marker's prefix is held back until the
            // next token settles it); the marker and its leading whitespace
            // stream to NEITHER side; NEITHER channel streams past a
            // tool-call opener (a short natural-language preamble before it,
            // which the template allows, does stream).
            let n = Tokenizer.completeUTF8Count(bytes)
            // The opener is searched over the WHOLE stream, reasoning
            // included: a small model emits real calls inside <think>
            // without ever closing it, and ignoring those streamed raw
            // markup into the disclosure and killed the turn answerless. A
            // call merely QUOTED in prose still passes through -- the parse
            // below requires a <function=> body, and a failed parse resumes
            // the stream (see the pending == nil recovery).
            if toolsActive && toolAt == nil {
                toolAt = ChatSession.index(
                    bytes, openTag, toolSearch)
                sawToolOpen = sawToolOpen || toolAt != nil
                if toolAt == nil {
                    toolSearch = max(toolSearch,
                        bytes.count - openTag.count + 1)
                }
            }
            let openHold = toolsActive && toolAt == nil
                ? ChatSession.partialSuffix(bytes, openTag, n)
                : 0
            if inThink {
                let hold = max(openHold, ChatSession.partialSuffix(
                    bytes, thinkClose, n))
                let end = min(toolAt ?? Int.max, n - hold)
                if end > emitted {
                    onReasoning?(String(decoding: bytes[emitted ..< end],
                                        as: UTF8.self))
                    emitted = end
                }
            } else if thinkDecided {
                if let at = closeAt, emitted <= at {
                    let flush = min(at, toolAt ?? Int.max)
                    if flush > emitted {
                        onReasoning?(String(decoding: bytes[emitted ..< flush],
                                            as: UTF8.self))
                    }
                    emitted = at + thinkClose.count
                }
                while !wsDone && emitted < n {
                    let b = bytes[emitted]
                    if b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D {
                        emitted += 1
                    } else {
                        wsDone = true
                    }
                }
                if wsDone {
                    var hold = openHold
                    var markAt = Int.max
                    var markLen = 0
                    // A close with no opener ANYWHERE this round closes
                    // nothing; a quoted pair keeps its close, one preceded it.
                    var marks = [thinkClose, thinkOpen]
                    if toolsActive {
                        if !sawToolOpen { marks.append(closeTag) }
                    } else {
                        marks.append(openTag)
                        marks.append(closeTag)
                    }
                    for pat in marks where !pat.isEmpty {
                        hold = max(hold,
                                   ChatSession.partialSuffix(bytes, pat, n))
                        if let at = ChatSession.index(bytes, pat, emitted),
                           at < markAt {
                            markAt = at
                            markLen = pat.count
                        }
                    }
                    let end = min(toolAt ?? Int.max, n - hold)
                    let stop = min(end, markAt)
                    if stop > emitted {
                        yield(String(decoding: bytes[emitted ..< stop],
                                     as: UTF8.self))
                        emitted = stop
                    }
                    if markAt == emitted, markAt + markLen <= n {
                        if markLen == closeTag.count, !sawToolOpen {
                            chatLog.error("suppressed a stray close tag")
                        }
                        emitted = markAt + markLen
                    }
                }
            }
            steps += 1
            // Refresh the running tg every few tokens (not every one) so the
            // status bar shows a live rate without per-token churn.
            if steps % 8 == 0 {
                let tg = Double(steps)
                    / max(Date().timeIntervalSince(g0), 1e-6)
                lastMetrics = TurnMetrics(
                    ctx: startCtx + steps, thinkTokens: think,
                    contentTokens: content, pp: pp, tg: tg)
            }
            // End reasoning on: Quick Answer, the SOFT cap passed at a
            // paragraph break (clean cut), or the HARD cap (immediate). All
            // gated by inThink, so a late trigger is a safe no-op.
            let softOver = softReasoningCap > 0 && think >= softReasoningCap &&
                bytes.last == 0x0A
            let looping = Continuation.isLooping(ids,
                tokenBytes: { [backend] id in backend.tokenBytes(id) })
            // A loop INSIDE <think> is unrecoverable reasoning, not an
            // unrecoverable turn (the digit-exempt sampler lets a
            // "0.00000..." run go where penalties once dampened it): close
            // the think ONCE like the caps and the EOS rescue do -- the
            // answer region can still call tools and answer. A loop that
            // survives the rescue ends the turn for real.
            let loopRescue = looping && inThinkRegion && closeAt == nil
                && toolAt == nil && thinkRescues == 0
            // toolAt non-nil suppresses injection: a call is mid-flight and
            // </think> must not land inside its markup.
            let overflow = inThink && toolAt == nil && (forceEndThink ||
                suppressReasoning || softOver || loopRescue
                || (maxReasoning > 0 && think >= maxReasoning))
            // A completed tool call ends the decode; the caller runs it and
            // re-seeds. The opener is searched over the WHOLE stream (above),
            // so a call the model opens inside <think> is still caught.
            if toolsActive && pending == nil, let open = toolAt {
                if toolCloseSearch < open + openTag.count {
                    toolCloseSearch = open + openTag.count
                }
                if toolReopenSearch < open + openTag.count {
                    toolReopenSearch = open + openTag.count
                }
                let at = ChatSession.index(bytes, closeTag,
                                           toolCloseSearch)
                let reopen = ChatSession.index(bytes, openTag,
                                               toolReopenSearch)
                if let reopen, at == nil || reopen < at! {
                    // A second opener before the close IMPLICITLY ends the
                    // block: the model reopened without closing (observed
                    // on-device), and parsing the merged span bleeds the
                    // next call's params into this one and swallows it.
                    pending = Tools.parse(Substring(String(
                        decoding: bytes[
                            (open + openTag.count) ..< reopen],
                        as: UTF8.self)))
                    // No <function=> in the span: junk markup. Release it
                    // to the stream and track the NEW opener instead.
                    if pending == nil {
                        toolAt = reopen
                        toolSearch = reopen
                        toolCloseSearch = reopen + openTag.count
                        toolReopenSearch = reopen + openTag.count
                    }
                } else if let at {
                    let end = at + closeTag.count
                    pending = completedCall(String(
                        decoding: bytes[open ..< end], as: UTF8.self))
                    // A malformed pair (no <function=> body -- markup merely
                    // quoted in prose) parses to nil: move the cursors past
                    // it and drop the opener anchor, so the held-back text
                    // streams after all and a later real call still detects.
                    if pending == nil {
                        toolCloseSearch = end
                        toolSearch = end
                        toolReopenSearch = end
                        toolAt = nil
                    }
                } else {
                    toolCloseSearch = max(toolCloseSearch,
                        bytes.count - closeTag.count + 1)
                    toolReopenSearch = max(toolReopenSearch,
                        bytes.count - openTag.count + 1)
                }
            }
            let openRunaway = toolAt.map { at in
                bytes.count - at > ChatSession.maxOpenCallBytes
            } == true
            stop = steps >= maxTokens
                || (looping && !loopRescue)
                || pending != nil
                || openRunaway
            if overflow && !stop {
                if await backend.queuedCount() > 0 {
                    // Spec-committed tokens the transcript has not received
                    // yet: injecting </think> now would land it after
                    // invisible tokens (KV and transcript diverge for the
                    // rest of the turn). Consume the queue first -- bounded
                    // by the draft count -- and let the caps re-fire.
                    cur = (try? await backend.decode(cur)) ?? backend.eos
                } else {
                    forceEndThink = false
                    if loopRescue { thinkRescues = 1 }
                    // Force the model out of <think>: inject </think> so it
                    // must produce the answer next. The next iteration's
                    // marker search finds the injected close marker and the
                    // whitespace skip eats its "\n\n", so none of it streams.
                    let close = backend.encode(
                        wire.reasoningClose + "\n\n")
                    ids.append(contentsOf: close)
                    for id in close {
                        bytes.append(contentsOf: backend.tokenBytes(id))
                    }
                    trace(.inject,
                          summary: "</think> injected (think \(think))")
                    cur = (try? await backend.extend(close)) ?? backend.eos
                }
            } else if !stop {
                cur = (try? await backend.decode(cur)) ?? backend.eos
                // EOS while still inside <think> is never a finished
                // turn -- no answer exists by construction (the 0.8B
                // thought itself to EOS after a 16KB ingest and committed
                // nothing). Close the think exactly like the caps do and
                // decode on, ONCE; a model that EOSes again commits empty
                // as before.
                if backend.eosIds.contains(cur), inThinkRegion,
                   closeAt == nil,
                   toolAt == nil, thinkRescues == 0,
                   await backend.queuedCount() == 0 {
                    thinkRescues = 1
                    let close = backend.encode(
                        wire.reasoningClose + "\n\n")
                    ids.append(contentsOf: close)
                    for id in close {
                        bytes.append(contentsOf: backend.tokenBytes(id))
                    }
                    trace(.inject,
                          summary: "</think> injected (eos inside think)")
                    cur = (try? await backend.extend(close)) ?? backend.eos
                }
            }
        }
        let whole = Tokenizer.completeUTF8Count(bytes)
        if !toolsActive, emitted < whole, thinkDecided,
           closeAt != nil || !inThinkRegion {
            let tail = String(decoding: bytes[emitted ..< whole],
                              as: UTF8.self)
            if !tail.isEmpty {
                yield(tail)
                emitted = whole
            }
        }

        // An EOS-cut call whose BODY completed: the model emitted
        // </function> and died before </tool_call> (observed twice on the
        // 4B, both sessions ending answerless with a recoverable call on
        // the floor). The close tag carries nothing the parse needs, so
        // dispatch the round -- the re-lay writes the canonical close
        // anyway. A span with no </function> may be cut mid-value and
        // never dispatches; Stop / cancel / breaker ends never rescue.
        if pending == nil, backend.eosIds.contains(cur), !Task.isCancelled,
           !backend.shouldStop(), let open = toolAt,
           ChatSession.index(bytes, ChatSession.closeFn, open) != nil {
            pending = Tools.parse(Substring(String(
                decoding: bytes[(open + openTag.count)...],
                as: UTF8.self)))
            if pending != nil {
                toolLog("eos-cut call rescued (body complete)")
            }
        }
        // WHY the decode ended, surfaced to the log: "no final answer" turns
        // are undiagnosable without knowing eos vs loop-breaker vs a cap.
        let reason: String
        if pending != nil {
            reason = "tool-call"
        } else if Task.isCancelled {
            reason = "cancelled"
        } else if backend.shouldStop() {
            reason = "stop"
        } else if backend.eosIds.contains(cur) {
            reason = "eos"
        } else if steps >= maxTokens {
            reason = "max-tokens"
        } else if toolAt.map({ at in
            bytes.count - at > ChatSession.maxOpenCallBytes }) == true {
            reason = "tool-runaway"
        } else {
            reason = "loop-breaker"
        }
        toolLog("decode end: \(reason) after \(steps) tokens "
            + "(think \(think), content \(content))")
        // No pending call: commit the final answer. A pending call: return the
        // content decoded before it (after </think>, usually empty), which the
        // caller re-lays canonically before the tool response. The raw decode
        // is NOT fed forward -- the caller rewinds past all of it.
        var preamble = ""
        var committedAnswer: String? = nil
        if pending == nil {
            // An opener that never resolved (EOS / breaker mid-call) must
            // not enter history as prose -- cut the commit at it, like the
            // pending path cuts its preamble. A quoted pair the recovery
            // released has already cleared toolAt, so it stays committed.
            let raw = toolAt.map { at in
                String(decoding: bytes[..<at], as: UTF8.self)
            } ?? backend.text(ids)
            var answer = answerText(raw, sawThink: inThinkRegion)
            // A released complete-but-unparseable block CAN be the entire
            // answer (EOS right after it, nothing streamed): committing
            // it would teach the model its own malformed wire on the next
            // delta. Markup quoted INSIDE prose keeps the prose.
            if strippedOfToolBlocks(answer).isEmpty {
                answer = ""
            } else {
                for pat in [wire.toolCallOpen, wire.toolCallClose]
                where !pat.isEmpty && answer.contains(pat) {
                    chatLog.error("stripped stray tool markup from an answer")
                    answer = answer.replacingOccurrences(of: pat, with: "")
                }
                answer = answer.trimmingCharacters(
                    in: .whitespacesAndNewlines)
            }
            history.append(AgentMessage(role: "assistant", content: answer))
            committedAnswer = answer
        } else {
            let decoded = backend.text(ids)
            var body = inThinkRegion ? "" : decoded
            if inThinkRegion,
               let c = decoded.range(of: wire.reasoningClose) {
                body = String(decoded[c.upperBound...])
            }
            if let tc = body.range(of: wire.toolCallOpen) {
                body = String(body[..<tc.lowerBound])
            }
            preamble = body.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let ctx = await backend.position
        let tgSec = Date().timeIntervalSince(g0)
        let tg = tgSec > 0 ? Double(steps) / tgSec : 0
        lastMetrics = TurnMetrics(ctx: ctx, thinkTokens: think,
                                  contentTokens: content, pp: pp, tg: tg,
                                  endReason: reason)
        trace(.decode, from: g0, ctx: ctx, tokens: steps,
              summary: reason + String(format:
                  " (think %d, content %d, tg %.1f t/s)", think, content, tg),
              text: backend.text(ids))
        if let committedAnswer {
            trace(.answer, tokens: content,
                  summary: "\(committedAnswer.count) chars",
                  text: committedAnswer)
        }
        return (pending, preamble)
    }

    // Parse the first complete tool-call block in the decoded text, framed by
    // this template's own markers.
    private func completedCall(_ text: String) -> ToolCall? {
        var result: ToolCall? = nil
        if let body = Tools.findToolCall(in: text, from: 0,
                                         open: wire.toolCallOpen,
                                         close: wire.toolCallClose) {
            result = Tools.parse(text[body])
        }
        return result
    }

    // Run one sanitized tool call, returning its result. NOT committed to
    // history -- the tool exchange lives only in the KV for this turn and is
    // compressed away after (like past-turn reasoning), so a later turn's
    // append-delta never renders a user-less [tool, assistant] slice (which the
    // Qwen template rejects for having no user query).
    // stderr only: an os_log copy doubled every tool line in Console
    // captures (Engine.report made the same call), and the structured trace
    // already carries each round for the app's surfaces.
    private func toolLog(_ s: String) {
        try? FileHandle.standardError.write(contentsOf: Data("\(s)\n".utf8))
    }

    // The canonical re-lay is the model's one view of the CORRECT wire, so
    // it carries only the resolved tool's advertised parameters (emitted
    // aliases canonicalized) with non-empty values. A hallucinated argument
    // (arg_value, script, max_results, ...) or an empty one re-laid
    // verbatim reads as system-accepted schema and costs context on every
    // following turn. An unknown tool has no schema; its args just drop
    // empties.
    private func sanitizedArgs(_ call: ToolCall,
                               _ resolved: String?) -> [ToolArg] {
        let known = resolved
            .flatMap { name in toolSpecs.first { s in s.name == name } }
            .map { spec in Tools.parameterNames(spec.parametersJSON) }
        var out: [ToolArg] = []
        for arg in call.params where !arg.value.isEmpty {
            if let known {
                if let canon = Tools.canonicalArgName(arg.name, known),
                   !out.contains(where: { a in a.name == canon }) {
                    out.append(ToolArg(name: canon, value: arg.value))
                }
            } else {
                out.append(arg)
            }
        }
        return out
    }

    static func callSignature(_ name: String, _ args: [ToolArg]) -> String {
        name + "\u{1}" + args.map { a in a.name + "=" + a.value }
            .sorted().joined(separator: "\u{1}")
    }

    static func repeatedCall(_ prior: String) -> String {
        "You already made this exact call. It returned: \(prior)\n"
            + "Do NOT repeat it. Change the arguments, use a different tool, "
            + "or answer the user now from what you already have."
    }

    private func runTool(_ call: ToolCall,
                         resolved: String?) async -> String {
        var result = "error: no tool runner"
        if let runner {
            if let resolved {
                if let blocked = Tools.sanitize(resolved, call.params) {
                    result = blocked
                } else {
                    result = await runner.execute(resolved, call.params)
                }
            } else {
                // Unknown-tool grounding: the model asked for a tool that does
                // not exist and is not a near-typo of one. Say so plainly (never
                // scrape the call) with the real options, so it recovers in one
                // round instead of being steered onto a wrong tool.
                let names = toolSpecs.map { spec in spec.name }
                result = "error: no tool named '\(call.functionName)'. "
                    + "Available tools: \(names.joined(separator: ", ")). "
                    + "If none of them fit, answer the user directly from your "
                    + "own knowledge."
            }
        }
        return result
    }

    // Names the model reaches for that map to one of ours. `tavily_search` /
    // `tavily_fetch` are REAL (not hallucinations): QwenPaw auto-creates a Tavily
    // MCP client of those names for web search + page fetch (docs/mcp.en.md), so
    // the RL model emits them -- we serve them via our own web_search (Mwmbl, no
    // API key) and fetch_url. `google_web_search` is the base model's
    // generalization, observed emitted. Applied only when the target is
    // advertised (else the name stays unknown -> grounded). Add entries here only
    // when a name is confirmed in the repo or seen in a log -- edit distance
    // would never bridge these.
    private static let toolAliases: [String: String] = [
        "tavily_search": "web_search",
        "google_web_search": "web_search",
        "tavily_fetch": "fetch_url",
    ]

    // Resolve an emitted tool name to an advertised one: exact match, else a
    // known RL alias (tavily_search -> web_search), else the nearest by edit
    // distance within a tight threshold (a clear typo, e.g. "wabsearch" ->
    // "web_search"), else nil -- an unknown tool, grounded rather than steered
    // onto a wrong one. Case-insensitive.
    static func resolveTool(_ name: String, _ available: [String]) -> String? {
        var result: String? = nil
        let lower = name.lowercased()
        if available.contains(name) {
            result = name
        } else if let alias = toolAliases[lower], available.contains(alias) {
            result = alias
        } else {
            var best: (name: String, dist: Int)? = nil
            for cand in available {
                let d = editDistance(lower, cand.lowercased())
                if best == nil || d < best!.dist { best = (cand, d) }
            }
            if let best, best.dist <= max(1, name.count / 4) {
                result = best.name
            }
        }
        return result
    }

    // Levenshtein edit distance; inputs are short (tool names).
    static func editDistance(_ a: String, _ b: String) -> Int {
        let x = Array(a), y = Array(b)
        var result: Int
        if x.isEmpty {
            result = y.count
        } else if y.isEmpty {
            result = x.count
        } else {
            var prev = Array(0 ... y.count)
            var cur = [Int](repeating: 0, count: y.count + 1)
            for i in 1 ... x.count {
                cur[0] = i
                for j in 1 ... y.count {
                    let cost = x[i - 1] == y[j - 1] ? 0 : 1
                    cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                                 prev[j - 1] + cost)
                }
                swap(&prev, &cur)
            }
            result = prev[y.count]
        }
        return result
    }

    // Continue a tool round WITHOUT growing the transcript with transient
    // reasoning: rewind to the mark, re-lay the round THROUGH THE TEMPLATE,
    // and ADVANCE the mark past it. The round renders as the slice
    // [current user, assistant(preamble + tool_calls), tool(result)] -- the
    // Qwen template raises on any render without a user query -- and the
    // user's own render is subtracted as a prefix (those bytes are already
    // in the KV), so what is laid is exactly the template's assistant +
    // tool-response markup; add_generation_prompt then re-opens the
    // assistant with the template's own think prefix. The head ENDS at the
    // tool-response boundary -- a clean turn boundary, like seedDelta's
    // mark -- so the exchange persists across turns and is never
    // re-rendered.
    private func toolContinuation(call: ToolCall, resolved: String?,
                                  preamble: String,
                                  result: String) async -> Int32 {
        let user = history.last { m in m.role == "user" }
            ?? AgentMessage(role: "user", content: "")
        let round = [
            user,
            AgentMessage(role: "assistant", content: preamble,
                         toolCalls: [AgentToolCall(
                             name: call.functionName,
                             arguments: sanitizedArgs(call, resolved))]),
            AgentMessage(role: "tool", content: result,
                         name: resolved ?? call.functionName),
        ]
        let prefix = (try? renderPrompt(
            template: template, messages: [user], tools: [],
            addGenerationPrompt: false,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
        let closedText = (try? renderPrompt(
            template: template, messages: round, tools: [],
            addGenerationPrompt: false,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
        let fullText = (try? renderPrompt(
            template: template, messages: round, tools: [],
            addGenerationPrompt: true,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
        let head = closedText.hasPrefix(prefix)
            ? String(closedText.dropFirst(prefix.count)) : closedText
        var genText = fullText.hasPrefix(closedText)
            ? String(fullText.dropFirst(closedText.count)) : ""
        if metaTurn { genText += ChatSession.titleSeed(genText, wire) }
        genStartsThink = wire.startsInReasoning(genPrompt: genText,
                                                enabled: true)
        var seed = backend.eos
        do {
            let t0 = Date()
            try await backend.rewind()
            trace(.rewind, ctx: await backend.position,
                  summary: "rewind to turn mark")
            let laid = backend.encode(head)
            let gen = backend.encode(genText)
            trace(.render, tokens: laid.count + gen.count,
                  summary: "tool continuation (call + response)",
                  text: head + genText)
            let afterHead = try await backend.extend(laid)
            committed += laid
            try await backend.mark()
            seed = gen.isEmpty ? afterHead : try await backend.extend(gen)
            let sec = Date().timeIntervalSince(t0)
            let total = laid.count + gen.count
            trace(.prefill, from: t0, ctx: await backend.position,
                  tokens: total,
                  summary: sec > 0 ? String(format: "%.0f t/s",
                                            Double(total) / sec) : "")
        } catch {
            seed = backend.eos
        }
        return seed
    }

    // The answer minus every complete tool-call span: empty means the
    // "answer" was nothing but markup and must not commit.
    private func strippedOfToolBlocks(_ s: String) -> String {
        var out = s
        while let open = out.range(of: wire.toolCallOpen),
              let close = out.range(
                  of: wire.toolCallClose,
                  range: open.upperBound ..< out.endIndex) {
            out.removeSubrange(open.lowerBound ..< close.upperBound)
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // The answer to keep in history: content AFTER </think>, or EMPTY when
    // Stop interrupted thinking before </think> closed (else the whole chain
    // of thought would land as the answer and re-prefill huge next turn).
    //
    // `sawThink` is what the decode loop OBSERVED, not the enableThinking
    // flag, and the two disagree in both directions. Reasoning on with a model
    // that answered without ever opening a channel found no close marker and
    // committed the whole real answer as EMPTY -- which is the empty-assistant
    // turn that poisons the next delta. Reasoning off with a model that opens
    // one anyway (gemma-4 always does) took the else branch and committed the
    // raw block, markup and all, into the next render. The decode loop already
    // tracks the truth; this asks it rather than the flag.
    private func answerText(_ decoded: String, sawThink: Bool) -> String {
        var body = ""
        if sawThink {
            if let close = decoded.range(of: wire.reasoningClose) {
                body = String(decoded[close.upperBound...])
            }
        } else {
            body = decoded
        }
        return body.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// ---------------------------------------------------------------------
// WHY THE TOOL GRAMMAR DEFAULTS TO STRUCTURAL (Leo, 2026-07-23)
//
// It shipped disarmed while the parse tolerance + template re-lay
// absorbed every observed malformation. Then the 4B rotted its markup
// ROUND OVER ROUND inside one agentic session -- round 1 clean, then
// "< function=", <arg_key>/<arg_value> soup, "<Function=call>",
// "</function >", junk between calls, EOS mid-call by round 5 -- the
// signature of the repetition penalties (presence 1.5 + DRY) punishing
// the wire tokens a call must re-emit verbatim. Two levers landed
// together:
//
// - Sampler.penaltyExempt removes the pressure at its source (the wire
//   specials no longer accrue repeat/presence/frequency/DRY penalties),
//   which also covers what no cheap mask can: EOS at a free-run value
//   position.
// - STRUCTURAL masking (default here) makes every tag-position
//   malformation impossible: inside an open <tool_call> the only legal
//   bytes at tag positions are the canonical wire, so spaced tags,
//   arg-soup wrappers, and EOS-before-close cannot be emitted at all.
//   Measured == grammar-off decode speed (tmp/260718-1823-grammar-ab);
//   the armed span disables MTP spec decode (bounded: call bodies only,
//   matters only on the 9B); the one-time GrammarVocab build happens on
//   the first armed call.
//
// Known trades, accepted: values and names still free-run (junk args
// remain possible; the tool's own grounding recovers), and markup merely
// QUOTED in prose arms the gate and gets completed into a real call --
// rare, and a dispatched call is recoverable where a dead session is
// not. LLM_TOOL_GRAMMAR=off restores the old behavior; "full" (whole-
// vocab walk at value bytes, ~7x decode) stays a debugging tool.
// ---------------------------------------------------------------------

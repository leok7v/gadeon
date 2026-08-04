import XCTest
@testable import LLM

// The gemma-4 turn loop, end to end over the REAL shipped chat template
// (LLM/fixtures/gemma4-chat-template.jinja, extracted from the repacked
// GGUF). Everything here failed before Phase 3b: the turn loop matched Qwen
// markers that gemma never emits, the tool round could not render at all, and
// a continuation delta spliced a spurious system turn into the KV.
//
// The backend models the KV as the literal byte stream so the wire laid into
// it is assertable offline, the same way TapeBackend does for the Qwen tests.
private final class GemmaTape: AgentBackend, @unchecked Sendable {
    let eos: Int32 = -1
    private let scripts: [[Int32]]
    private let bytesOf: [Int32: [UInt8]]
    private var tape: [Int32] = []
    private var marked = 0
    private var round = -1
    private var cursor = 0

    init(scripts: [[Int32]], vocab: [Int32: [UInt8]]) {
        self.scripts = scripts
        self.bytesOf = vocab
    }

    // Every image marker encodes ATOMICALLY, as a real tokenizer does. A
    // byte spelling would make its first byte "<" the pad id, and expandPads
    // would then strip every "<" from the prompt -- which is every tag in
    // gemma's wire.
    private static let imagePad: Int32 = 999

    func encode(_ text: String) -> [Int32] {
        text == "<|image_pad|>" || text == "<|image|>"
            ? [GemmaTape.imagePad]
            : text.utf8.map { byte in Int32(byte) }
    }
    func tokenBytes(_ id: Int32) -> [UInt8] {
        id >= 0 && id < 256 ? [UInt8(id)] : (bytesOf[id] ?? [])
    }
    func text(_ ids: [Int32]) -> String {
        var b: [UInt8] = []
        for id in ids { b.append(contentsOf: tokenBytes(id)) }
        return String(decoding: b, as: UTF8.self)
    }
    var kvText: String { text(tape) }
    var position: Int { get async { tape.count } }
    func reset() async {
        tape = []
        round += 1
        cursor = 1
    }
    func useSampler(_ s: Sampler?) async {}
    private func script() -> [Int32] {
        round >= 0 && round < scripts.count ? scripts[round] : []
    }
    func extend(_ ids: [Int32]) async throws -> Int32 {
        tape.append(contentsOf: ids)
        let s = script()
        return s.isEmpty ? eos : s[0]
    }
    func mark() async throws { marked = tape.count }
    func rewind() async throws {
        tape = Array(tape.prefix(marked))
        round += 1
        cursor = 1
    }
    func decode(_ token: Int32) async throws -> Int32 {
        tape.append(token)
        let s = script()
        let out = cursor < s.count ? s[cursor] : eos
        cursor += 1
        return out
    }
    struct State: BackendState { let tape: [Int32]; let marked: Int }
    func saveState() async throws -> any BackendState {
        State(tape: tape, marked: marked)
    }
    func loadState(_ state: any BackendState) async throws {
        if let s = state as? State {
            tape = s.tape
            marked = s.marked
        }
    }
}

// A tape that lifts the modality markers out of a render ATOMICALLY wherever
// they occur, the way a real tokenizer does. The whole-string shortcut the
// tape above takes never yields a placeholder id inside a full render, so an
// expansion measured against it would silently do nothing and pass.
private final class SoftTape: AgentBackend, @unchecked Sendable {
    static let image: Int32 = 999
    static let audio: Int32 = 996
    static let boi: Int32 = 997
    static let eoi: Int32 = 998

    let eos: Int32 = -1
    private let scripts: [[Int32]]
    private let bytesOf: [Int32: [UInt8]]
    private var tape: [Int32] = []
    private var marked = 0
    private var round = -1
    private var cursor = 0
    private(set) var softCalls = 0
    private(set) var softRows = 0

    init(scripts: [[Int32]], vocab: [Int32: [UInt8]]) {
        self.scripts = scripts
        self.bytesOf = vocab
    }

    func encode(_ text: String) -> [Int32] {
        var out: [Int32] = []
        var rest = Substring(text)
        while let hit = SoftTape.firstMarker(rest) {
            out.append(contentsOf: rest[rest.startIndex ..< hit.0.lowerBound]
                .utf8.map { byte in Int32(byte) })
            out.append(hit.1)
            rest = rest[hit.0.upperBound...]
        }
        out.append(contentsOf: rest.utf8.map { byte in Int32(byte) })
        return out
    }

    private static func firstMarker(
        _ s: Substring
    ) -> (Range<Substring.Index>, Int32)? {
        var best: (Range<Substring.Index>, Int32)? = nil
        for (text, id) in [("<|image|>", image), ("<|audio|>", audio)] {
            if let r = s.range(of: text),
               best == nil || r.lowerBound < best!.0.lowerBound {
                best = (r, id)
            }
        }
        return best
    }

    func tokenBytes(_ id: Int32) -> [UInt8] {
        id >= 0 && id < 256 ? [UInt8(id)] : (bytesOf[id] ?? [])
    }
    func text(_ ids: [Int32]) -> String {
        var b: [UInt8] = []
        for id in ids { b.append(contentsOf: tokenBytes(id)) }
        return String(decoding: b, as: UTF8.self)
    }
    var kvText: String { text(tape) }
    var position: Int { get async { tape.count } }
    func reset() async {
        tape = []
        round += 1
        cursor = 1
    }
    func useSampler(_ s: Sampler?) async {}
    private func script() -> [Int32] {
        round >= 0 && round < scripts.count ? scripts[round] : []
    }
    func extend(_ ids: [Int32]) async throws -> Int32 {
        tape.append(contentsOf: ids)
        let s = script()
        return s.isEmpty ? eos : s[0]
    }
    func supportsSoftTokens() async -> Bool { true }
    func extendSoft(_ ids: [Int32], spans: [SoftSpan]) async throws -> Int32 {
        softCalls += 1
        let feed = SoftFeed(spans)
        for id in ids where feed.row(id) != nil { softRows += 1 }
        return try await extend(ids)
    }
    func mark() async throws { marked = tape.count }
    func rewind() async throws {
        tape = Array(tape.prefix(marked))
        round += 1
        cursor = 1
    }
    func decode(_ token: Int32) async throws -> Int32 {
        tape.append(token)
        let s = script()
        let out = cursor < s.count ? s[cursor] : eos
        cursor += 1
        return out
    }
    struct State: BackendState { let tape: [Int32]; let marked: Int }
    func saveState() async throws -> any BackendState {
        State(tape: tape, marked: marked)
    }
    func loadState(_ state: any BackendState) async throws {
        if let s = state as? State {
            tape = s.tape
            marked = s.marked
        }
    }
}

private final class GemmaRunner: ToolRunner, @unchecked Sendable {
    let tools: [ToolSpec]
    private let reply: String
    private(set) var calls: [String] = []

    init(reply: String) {
        self.tools = [ToolSpec(
            name: "calculator", description: "evaluate math.",
            parametersJSON: "{\"type\":\"object\",\"properties\":"
                + "{\"expression\":{\"type\":\"string\","
                + "\"description\":\"the expression\"}},"
                + "\"required\":[\"expression\"]}")]
        self.reply = reply
    }
    func execute(_ name: String, _ args: [ToolArg]) async -> String {
        calls.append(name)
        return reply
    }
}

final class GemmaWireTests: XCTestCase {

    private func template() throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
        return try String(
            contentsOf: root.appendingPathComponent(
                "fixtures/gemma4-chat-template.jinja"),
            encoding: .utf8)
    }

    private func vocab(_ pairs: [(Int32, String)]) -> [Int32: [UInt8]] {
        var out: [Int32: [UInt8]] = [:]
        for (id, s) in pairs { out[id] = Array(s.utf8) }
        return out
    }

    private func drain(_ stream: AsyncStream<String>) async -> String {
        var out = ""
        for await piece in stream { out += piece }
        return out
    }

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var acc = ""
        func add(_ s: String) { lock.lock(); acc += s; lock.unlock() }
        var text: String { lock.lock(); defer { lock.unlock() }; return acc }
    }

    // Gemma's generation prompt is a bare "<|turn>model\n" -- the model opens
    // its OWN <|channel>thought. Decoding therefore starts in CONTENT, the
    // leading-marker scan routes the channel to reasoning, and only what
    // follows <channel|> is the answer.
    func testReasoningChannelSplits() async throws {
        let backend = GemmaTape(
            scripts: [[1001, 1002, 1003, 1004]],
            vocab: vocab([(1001, "<|channel>thought\n"),
                          (1002, "let me work this out"),
                          (1003, "\n<channel|>"),
                          (1004, "The answer is 4.")]))
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256, enableThinking: true)
        let reasoning = Box()
        var content = ""
        let stream = session.reply("2+2?",
                                   onReasoning: { r in reasoning.add(r) })
        for await piece in stream { content += piece }
        XCTAssertEqual(content, "The answer is 4.",
                       "gemma's answer did not reach the content stream")
        XCTAssertTrue(reasoning.text.contains("let me work this out"),
                      "reasoning lost: \(reasoning.text)")
        XCTAssertFalse(content.contains("<|channel>"),
                       "raw channel markup leaked into content: \(content)")
    }

    // The KV must carry gemma's own turn wire, and a SECOND turn must not
    // splice another system turn in: enable_thinking alone opens one on every
    // render, so the continuation delta has to subtract it.
    func testSecondTurnDoesNotDoubleTheSystemBlock() async throws {
        let backend = GemmaTape(
            scripts: [[1001], [1002]],
            vocab: vocab([(1001, "First answer."),
                          (1002, "Second answer.")]))
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256, enableThinking: true)
        let a1 = await drain(session.reply("first"))
        XCTAssertEqual(a1, "First answer.")
        let a2 = await drain(session.reply("second"))
        XCTAssertEqual(a2, "Second answer.")
        let kv = backend.kvText
        let systems = kv.components(separatedBy: "<|turn>system").count - 1
        XCTAssertEqual(systems, 1,
            "the continuation delta re-laid the system turn:\n\(kv)")
        XCTAssertTrue(kv.contains("<|turn>user\nsecond<turn|>"),
                      "second user turn missing from the KV:\n\(kv)")
        XCTAssertTrue(kv.hasSuffix("<|turn>model\nSecond answer."),
                      "turn two did not open the answer correctly:\n\(kv)")
    }

    // A full tool round in gemma's dialect: the model emits
    // <|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>, the loop detects it
    // with the DERIVED markers, Tools.parse reads the third dialect, and the
    // round re-lays through the template as <|tool_response>response:NAME{..}.
    func testToolRoundDispatchesAndRelays() async throws {
        let call = "<|tool_call>call:calculator{expression:"
            + "<|\"|>2 + 2<|\"|>}<tool_call|>"
        let backend = GemmaTape(
            scripts: [[1001], [1002]],
            vocab: vocab([(1001, call), (1002, "It is 4.")]))
        let runner = GemmaRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("what is 2 + 2?"))
        XCTAssertEqual(runner.calls, ["calculator"],
                       "gemma's tool dialect did not dispatch")
        XCTAssertEqual(answer, "It is 4.")
        let kv = backend.kvText
        XCTAssertTrue(kv.contains("<|tool_call>call:calculator{expression:"),
                      "the call did not re-lay in gemma's wire:\n\(kv)")
        XCTAssertTrue(kv.contains("<|tool_response>response:calculator{"),
                      "the response did not name the tool:\n\(kv)")
        XCTAssertFalse(kv.contains("response:unknown"),
                       "the tool response lost its name:\n\(kv)")
    }

    // Tool DECLARATIONS must carry their parameters. Gemma's template walks
    // the schema structurally rather than tojson-ing it, so an opaque
    // parameters node advertised every tool with no arguments at all.
    func testToolDeclarationCarriesParameters() async throws {
        let backend = GemmaTape(scripts: [[1001]],
                                vocab: vocab([(1001, "Hi.")]))
        let runner = GemmaRunner(reply: "unused")
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256, runner: runner)
        _ = await drain(session.reply("hello"))
        let kv = backend.kvText
        XCTAssertTrue(kv.contains("<|tool>declaration:calculator{"),
                      "no tool declaration in the KV:\n\(kv)")
        XCTAssertTrue(kv.contains("parameters:{"),
                      "the declaration lost its parameters block:\n\(kv)")
        XCTAssertTrue(kv.contains("expression:{"),
                      "the declaration lost its parameter name:\n\(kv)")
        XCTAssertTrue(kv.contains("required:["),
                      "the declaration lost its required list:\n\(kv)")
    }

    // ---- soft-token attachments -------------------------------------

    private func softVocab() -> [Int32: [UInt8]] {
        vocab([(SoftTape.image, "<img>"), (SoftTape.audio, "<aud>"),
               (SoftTape.boi, "<boi>"), (SoftTape.eoi, "<eoi>"),
               (1001, "A pharmacy."), (1002, "It was red."),
               (1003, "Two of them.")])
    }

    private func imageSpan(_ count: Int) -> SoftSpan {
        SoftSpan.bracketed(
            begin: SoftTape.boi, placeholder: SoftTape.image,
            end: SoftTape.eoi, count: count,
            features: [Float](repeating: 0.5, count: count * 2))
    }

    // The template writes ONE placeholder; the span expands it in place to
    // begin + N + end, and every one of those N positions must be fed a
    // tower row. A count that is not a constant is the whole point -- this
    // tower's output follows the aspect ratio.
    func testSoftTurnExpandsPlaceholderAndFeedsRows() async throws {
        let backend = SoftTape(scripts: [[1001]], vocab: softVocab())
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256)
        let answer = await drain(session.replySoft(
            "what is this?", parts: [.image], spans: [imageSpan(3)]))
        XCTAssertEqual(answer, "A pharmacy.")
        XCTAssertEqual(backend.softCalls, 1, "the soft path did not run")
        XCTAssertEqual(backend.softRows, 3,
                       "every placeholder position must be fed a row")
        XCTAssertTrue(backend.kvText.contains("<boi><img><img><img><eoi>"),
                      "the span did not expand in place:\n\(backend.kvText)")
        XCTAssertFalse(backend.kvText.contains("<|image|>"),
                       "an unexpanded marker survived:\n\(backend.kvText)")
    }

    // The multi-turn contract: a plain turn AFTER an attachment must not
    // re-lay it (the tower ran once and its rows are already in the KV), and
    // must not double the system turn the way an un-subtracted leading block
    // would.
    func testAttachmentSurvivesLaterPlainTurn() async throws {
        let backend = SoftTape(scripts: [[1001], [1002]], vocab: softVocab())
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256)
        _ = await drain(session.replySoft("what is this?", parts: [.image],
                                          spans: [imageSpan(3)]))
        let a2 = await drain(session.reply("what colour was it?"))
        XCTAssertEqual(a2, "It was red.")
        XCTAssertEqual(backend.softCalls, 1,
                       "the attachment was re-encoded on a later turn")
        let kv = backend.kvText
        XCTAssertEqual(kv.components(separatedBy: "<boi>").count - 1, 1,
                       "the attachment block was re-laid:\n\(kv)")
        XCTAssertEqual(kv.components(separatedBy: "<|turn>system").count - 1, 1,
                       "a later turn spliced a second system turn:\n\(kv)")
        XCTAssertTrue(kv.contains("A pharmacy."),
                      "the first answer left the KV:\n\(kv)")
    }

    // An attachment arriving on a LATER turn takes the rewind-and-append
    // path rather than the fresh one, and must leave the earlier turns
    // standing.
    func testAttachmentOnLaterTurnAppends() async throws {
        let backend = SoftTape(scripts: [[1001], [1002]], vocab: softVocab())
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256)
        _ = await drain(session.reply("hello"))
        _ = await drain(session.replySoft("and this?", parts: [.image],
                                          spans: [imageSpan(4)]))
        XCTAssertEqual(backend.softCalls, 1)
        XCTAssertEqual(backend.softRows, 4)
        let kv = backend.kvText
        XCTAssertTrue(kv.contains("<boi><img><img><img><img><eoi>"),
                      "the later attachment did not expand:\n\(kv)")
        XCTAssertTrue(kv.contains("hello"),
                      "the earlier turn was lost:\n\(kv)")
        XCTAssertEqual(kv.components(separatedBy: "<|turn>system").count - 1, 1,
                       "the later attachment doubled the system turn:\n\(kv)")
    }

    // Two modalities in ONE turn walk independent cursors: the image rows
    // and the audio rows are different placeholder ids, and feeding either
    // from the other's queue would put a clip where a picture belongs.
    func testTwoModalitiesInOneTurnKeepSeparateCursors() async throws {
        let backend = SoftTape(scripts: [[1003]], vocab: softVocab())
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256)
        let clip = SoftSpan.bracketed(
            begin: SoftTape.boi, placeholder: SoftTape.audio,
            end: SoftTape.eoi, count: 2,
            features: [Float](repeating: 0.25, count: 4))
        _ = await drain(session.replySoft(
            "what is here?", parts: [.image, .audio],
            spans: [imageSpan(3), clip]))
        XCTAssertEqual(backend.softRows, 5, "3 image + 2 audio rows")
        let kv = backend.kvText
        XCTAssertTrue(kv.contains("<boi><img><img><img><eoi>"),
                      "image span wrong:\n\(kv)")
        XCTAssertTrue(kv.contains("<boi><aud><aud><eoi>"),
                      "audio span wrong:\n\(kv)")
    }

    // Gemma's template numbers nothing, so ChatSession names each attachment
    // itself -- the handle a within-turn question needs ("is Picture 1 inside
    // Picture 2"). The nouns are Qwen's, so both lineages read alike.
    func testSoftTurnNumbersAttachments() async throws {
        let backend = SoftTape(scripts: [[1003]], vocab: softVocab())
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256)
        _ = await drain(session.replySoft(
            "which is inside which?", parts: [.image, .image,
                                              .text("which is inside which?")],
            spans: [imageSpan(2), imageSpan(2)]))
        let kv = backend.kvText
        XCTAssertTrue(kv.contains("Picture 1:<boi><img><img><eoi>"),
                      "first attachment unnamed:\n\(kv)")
        XCTAssertTrue(kv.contains("Picture 2:<boi><img><img><eoi>"),
                      "second attachment unnamed:\n\(kv)")
    }

    // The counter is conversation-global, NOT per turn: an attachment already
    // in the KV can never be re-labelled, so turn two must continue the
    // numbering rather than restart it and collide.
    func testAttachmentNumberingIsConversationGlobal() async throws {
        let backend = SoftTape(scripts: [[1001], [1002]], vocab: softVocab())
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256)
        _ = await drain(session.replySoft("first?", parts: [.image,
                                                            .text("first?")],
                                          spans: [imageSpan(2)]))
        _ = await drain(session.replySoft("second?", parts: [.image,
                                                             .text("second?")],
                                          spans: [imageSpan(2)]))
        let kv = backend.kvText
        XCTAssertTrue(kv.contains("Picture 1:"), "turn one unnamed:\n\(kv)")
        XCTAssertTrue(kv.contains("Picture 2:"),
                      "turn two restarted the numbering:\n\(kv)")
        XCTAssertEqual(kv.components(separatedBy: "Picture 1:").count - 1, 1,
                       "two attachments share a name:\n\(kv)")
    }

    // Each modality counts on its own, as Qwen's template does with its
    // separate image and video counters.
    func testModalitiesNumberIndependently() async throws {
        let backend = SoftTape(scripts: [[1003]], vocab: softVocab())
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256)
        let clip = SoftSpan.bracketed(
            begin: SoftTape.boi, placeholder: SoftTape.audio,
            end: SoftTape.eoi, count: 2,
            features: [Float](repeating: 0.25, count: 4))
        _ = await drain(session.replySoft(
            "what?", parts: [.image, .audio, .text("what?")],
            spans: [imageSpan(3), clip]))
        let kv = backend.kvText
        XCTAssertTrue(kv.contains("Picture 1:"), "image unnamed:\n\(kv)")
        XCTAssertTrue(kv.contains("Audio 1:"),
                      "audio took the image's counter:\n\(kv)")
    }

    // Text parts are laid where the CALLER put them, which is what makes an
    // A-inside-B question expressible at all.
    func testSoftTurnInterleavesTextPositionally() async throws {
        let backend = SoftTape(scripts: [[1003]], vocab: softVocab())
        let session = ChatSession(
            backend: backend, template: try template(),
            system: "You are a bot.", vocabSize: 256)
        _ = await drain(session.replySoft(
            "is a inside b?",
            parts: [.text("is"), .image, .text("inside"), .image,
                    .text("?")],
            spans: [imageSpan(1), imageSpan(1)]))
        let kv = backend.kvText
        XCTAssertTrue(kv.contains("isPicture 1:<boi><img><eoi>insidePicture 2:"
                          + "<boi><img><eoi>?"),
                      "parts did not render in caller order:\n\(kv)")
    }

    // Qwen numbers attachments in its OWN template (add_vision_id), so
    // ChatSession must not number them a second time.
    func testTemplateThatNumbersIsNotDoubleNumbered() async throws {
        let qwenish = "{%- for m in messages -%}"
            + "<|im_start|>{{ m.role }}\n"
            + "{%- for p in m.content -%}"
            + "{%- if p.type == 'image' -%}"
            + "{%- if add_vision_id %}{{- 'Picture 9: ' -}}{% endif -%}"
            + "<|image|>{%- else -%}{{- p.text -}}{%- endif -%}"
            + "{%- endfor -%}<|im_end|>\n{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|im_start|>assistant\n"
            + "{%- endif -%}"
        let backend = SoftTape(scripts: [[1001]], vocab: softVocab())
        let session = ChatSession(
            backend: backend, template: qwenish,
            system: "You are a bot.", vocabSize: 256)
        _ = await drain(session.replySoft("what?",
                                          parts: [.image, .text("what?")],
                                          spans: [imageSpan(2)]))
        XCTAssertFalse(backend.kvText.contains("Picture 1:"),
                       "a self-numbering template was numbered twice:\n"
                       + backend.kvText)
    }

    // A block's OWN placeholders are soft positions, not further
    // attachments: expanding what it appends would recurse a video's
    // per-frame blocks into nonsense.
    func testExpandSpansDoesNotReexpandItsOwnBlock() {
        let span = SoftSpan.bracketed(begin: 7, placeholder: 9, end: 8,
                                      count: 2, features: [0, 0])
        let out = Continuation.expandSpans([1, 9, 2], [span])
        XCTAssertEqual(out, [1, 7, 9, 9, 8, 2])
    }

    // Two spans sharing a placeholder are consumed in the order the
    // template emitted them, one per marker.
    func testExpandSpansConsumesOnePerMarker() {
        let a = SoftSpan(placeholder: 9, ids: [70, 9, 80], features: [0])
        let b = SoftSpan(placeholder: 9, ids: [71, 9, 9, 81], features: [0, 0])
        let out = Continuation.expandSpans([9, 5, 9], [a, b])
        XCTAssertEqual(out, [70, 9, 80, 5, 71, 9, 9, 81])
    }

    // The gemma call body parses standalone, including a value carrying the
    // ',' and ':' that otherwise delimit fields.
    func testParseGemmaDialect() {
        let body = "call:calculator{expression:<|\"|>2 + 2<|\"|>}"
        let call = Tools.parse(Substring(body))
        XCTAssertEqual(call?.functionName, "calculator")
        XCTAssertEqual(call?.params.first?.name, "expression")
        XCTAssertEqual(call?.params.first?.value, "2 + 2")

        let tricky = "call:web_search{query:<|\"|>a, b: c<|\"|>,limit:5}"
        let two = Tools.parse(Substring(tricky))
        XCTAssertEqual(two?.functionName, "web_search")
        XCTAssertEqual(two?.params.count, 2)
        XCTAssertEqual(two?.params.first?.value, "a, b: c",
                       "a quoted value must survive its own delimiters")
        XCTAssertEqual(two?.params.last?.name, "limit")
        XCTAssertEqual(two?.params.last?.value, "5")
    }
}

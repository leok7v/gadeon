import CoreML
import Foundation
import ImageIO
import OSLog
import LLM
import MD
import UniformTypeIdentifiers

@MainActor @Observable final class ChatModel {

    struct Doc: Identifiable {
        let id = UUID()
        let name: String
        let content: String
        // For the reader only. `content` is what the turn carries.
        let url: URL?
        // Recorded, never inferred: raising the budget later cannot un-cut
        // what an earlier turn already sent.
        let short: Bool
    }

    struct DocRef: Hashable {
        let url: URL
        let bytes: Int
        let short: Bool
    }

    struct ImageAttachment: Identifiable {
        let id = UUID()
        let name: String
        let data: Data
        // Decoded ONCE at attach, so the chip does not re-decode per render.
        let thumbnail: CGImage?
    }

    // A URL, never bytes: holding a video in memory for the life of a
    // conversation is how an app gets jetsammed, and only the decode wants it.
    struct ClipAttachment: Identifiable {
        let id = UUID()
        let name: String
        let url: URL
        let isVideo: Bool
        // nil for sound, which has no picture.
        let thumbnail: CGImage?

        // A video carries BOTH its frames and its own soundtrack, which the
        // library keeps apart so a caller can order them.
        func spans(_ media: Gemma4Media,
                   onFrame: (@Sendable (VideoPeek) -> Void)? = nil)
            async throws -> [(ContentPart, SoftSpan)] {
            var out: [(ContentPart, SoftSpan)] = []
            if isVideo {
                // Scaled HERE, on the decoding thread: the frame handed over
                // is a source-resolution bitmap released the moment this
                // returns.
                var seen = 0
                out.append((.video, try await media.video(url: url) {
                    img, _ in
                    if let onFrame,
                       let peek = VideoPeek(index: seen, full: img) {
                        onFrame(peek)
                    }
                    seen += 1
                }))
                // A soundtrack past the model's ceiling costs the SOUND, not
                // the whole attachment.
                for heard in (try? await media.audio(url: url)) ?? [] {
                    out.append((.audio, heard))
                }
            } else {
                for heard in try await media.audio(url: url) {
                    out.append((.audio, heard))
                }
            }
            return out
        }
    }

    struct ToolRound: Identifiable {
        let id: Int                  // round number within the turn
        let emitted: String          // the name the model asked for
        let label: String            // display name (resolved tool, or emitted)
        let symbol: String           // SF Symbol for the row glyph
        let args: String             // key: "value" summary of the call params
        // The exact string the model received back; nil while it runs.
        var result: String?
    }

    struct Message: Identifiable {
        let id = UUID()
        let fromUser: Bool
        var text: String
        var images: [CGImage] = []
        // The PATH, never the bytes: a conversation is kilobytes and a phone
        // clip is tens of megabytes. A path can go stale, so the view tests
        // the file and names it rather than offering it.
        var clips: [URL] = []
        // Same path-not-bytes rule as clips. The size recorded is of the
        // TEXT taken from the file, which is the whole of what the model got.
        var docs: [DocRef] = []
        var toolRounds: [ToolRound] = []
        var reasoning = ""
        // The text is a STAND-IN rather than words: a dictated turn shows
        // "Spoken, 1.9s". Nothing that names the conversation may come from
        // one.
        var placeholder = false
        var loopStopped = false
        // The streams are reference types whose sealed blocks never
        // re-parse; only the open block does.
        var answerDoc = Markdown.Document.empty
        var reasoningDoc = Markdown.Document.empty
        let answerStream = MarkdownStream()
        let reasoningStream = MarkdownStream()
    }

    // Host-side policy, independent of the tower's baked grid.
    enum VisionMode: String, CaseIterable, Identifiable {
        case tile, fit
        var id: String { rawValue }
        var label: String { self == .tile ? "Tile" : "Fit" }
        var detail: String {
            var result = "Scale the whole image to fit a single tile. Fast "
                + "and light; less detail on dense pictures."
            if self == .tile {
                result = "Split a large image into overlapping detail tiles "
                    + "plus a whole-image overview. Best detail on dense "
                    + "pictures; slower and heavier on memory."
            }
            return result
        }
    }

    static let defaultSystemPrompt = "You are a helpful assistant."

    var messages: [Message] = []
    // A reopened conversation touches no session or KV: the transcript is
    // shown and the composer hidden.
    var readOnly = false
    var currentConversationId: UUID?
    var generatedTitle: String?
    var theme: AppTheme = {
        AppTheme(rawValue:
            UserDefaults.standard.string(forKey: "theme") ?? "") ?? .system
    }() {
        didSet { UserDefaults.standard.set(theme.rawValue, forKey: "theme") }
    }
    // NOTCHES either side of what the device already gives, so zero is its
    // own size and an absent default reads the same. A multiplier of ours
    // rather than a dynamicTypeSize shift, which does nothing on macOS:
    // AppKit has no content size category for Font.body or @ScaledMetric to
    // read.
    var textZoom: Int = ChatModel.clampZoom(
        UserDefaults.standard.integer(forKey: "textZoom")) {
        didSet { UserDefaults.standard.set(textZoom, forKey: "textZoom") }
    }
    static let zoomLimit = 2

    static func clampZoom(_ notches: Int) -> Int {
        min(max(notches, -zoomLimit), zoomLimit)
    }

    // A notch is a tenth: 0.8 to 1.2 across the five stops.
    var textScale: CGFloat { ChatModel.zoomScale(textZoom) }

    static func zoomScale(_ notch: Int) -> CGFloat {
        1 + CGFloat(notch) / 10
    }

    // Bounded before it is converted: a two-finger fling reports a ratio of
    // any size, where the whole ladder spans 1.5.
    static func zoomNotch(nearest scale: CGFloat) -> Int {
        clampZoom(Int(((min(max(scale, 0.5), 2) - 1) * 10).rounded()))
    }

    func flashZoom(_ notch: Int) {
        flashHUD("Zoom \(Int(ChatModel.zoomScale(notch) * 100))%")
    }

    var canZoomIn: Bool { textZoom < ChatModel.zoomLimit }
    var canZoomOut: Bool { textZoom > -ChatModel.zoomLimit }
    var atDefaultZoom: Bool { textZoom == 0 }

    func zoomIn() { if canZoomIn { textZoom += 1 } }
    func zoomOut() { if canZoomOut { textZoom -= 1 } }
    func resetZoom() { textZoom = 0 }

    var input = ""
    // UTF-16 offset into `input`, synced from the editor so a dropped file's
    // reference lands at the caret.
    var caret = 0
    var attachedImages: [ImageAttachment] = []
    var attachedDocs: [Doc] = []
    // Files being read into Markdown right now. A drop is accepted the moment
    // it starts, and only the text arrives late.
    private(set) var converting = 0
    var attachedClips: [ClipAttachment] = []
    var status = "loading model..."
    // A built session is NOT enough: build makes it before the warmup and
    // carry compiles finish.
    var ready: Bool { session != nil && !compiling }
    var busy: Bool { genTask != nil }

    // ONLY FilmStrip reads this, so a new frame re-evaluates that one view
    // and leaves the transcript alone.
    var lookingAt: VideoPeek?
    // Raised by the encoder, which knows when it is done; "nothing arrived
    // recently" would need a timer racing the next arrival.
    var watching = false
    // Wider than `busy`, which ends with the last token while the voice
    // reads on.
    var inTurn: Bool { busy || listening || speech.engaged }

    // Offered, never opened: a microphone that reopens itself gets left on,
    // and the room it would open into is the one the reply just played into.
    var voiceReady: Bool {
        lastTurnSpoken && canAttachAudio && !busy && !listening
            && !speech.engaged && input.isEmpty
    }

    private(set) var lastTurnSpoken = false
    var statsLabel = ""             // the status line's one line of numbers

    // Read from the model's own files at build; nil until one is loaded.
    private(set) var modelShape: ModelShape?

    // Parts rather than one string: each weight is drawn behind its own SF
    // Symbol.
    struct Tower: Identifiable {
        let id: Int
        let symbol: String
        let weight: String
    }

    var modelTowers: [Tower] {
        var out: [Tower] = []
        if let shape = modelShape {
            for (i, t) in shape.towers.enumerated() {
                out.append(Tower(id: i, symbol: ChatModel.towerSymbol(t.name),
                                 weight: ChatModel.weight(t.bytes)))
            }
        }
        return out
    }

    private static func towerSymbol(_ name: String) -> String {
        var out = "text.alignleft"
        if name == "vision" {
            out = "eye"
        } else if name == "audio" {
            out = "waveform.path"
        }
        return out
    }

    var modelFacts: [String] {
        var out: [String] = []
        if let shape = modelShape {
            if shape.trainedContext > 0 {
                out.append(ChatModel.tokens(shape.trainedContext) + " ctx")
            }
            if shape.embedding > 0 {
                out.append(shape.embedding.formatted(.number) + " dim")
            }
        }
        return out
    }

    // The short form is EXACT, not an approximation: anything that is not a
    // round binary number keeps its digits.
    private static func tokens(_ n: Int) -> String {
        n >= 1024 && n % 1024 == 0 ? "\(n / 1024)K" : n.formatted(.number)
    }

    // A unified checkpoint's projections weigh single-digit megabytes, which
    // in gigabytes would round to "0.0 GB".
    private static func weight(_ bytes: Int) -> String {
        let gb = Double(bytes) / 1_073_741_824
        return gb >= 1 ? String(format: "%.1f GB", gb)
                       : String(format: "%.0f MB",
                                Double(bytes) / 1_048_576)
    }
    let speech = VoiceSession()
    // The microphone is open; the turn starts only once it closes.
    var listening = false
    // Speech CAPTURED so far this session, which is not the time the
    // microphone has been open: the pauses are already gone.
    var heardSeconds = 0.0
    // A speech run is OPEN, and how far over the gate's bar the voice sits.
    var hearingSpeech = false
    var speechLevel = 0.0
    // Before the first decoded token.
    var prefilling = false
    // A tool is running or its result is being ingested. Set by the session's
    // onTool, cleared when the answer begins to stream.
    var consulting = false
    // One per surface that can show a phrase at once: the transcript's
    // working line and the reasoning disclosure header.
    var thinkStatus = "Thinking"
    var thinkLabel = "Thinking"
    var eulaAccepted = UserDefaults.standard.bool(forKey: "eulaAccepted")
    var accepted = UserDefaults.standard.bool(forKey: "disclaimerAccepted")
    // Sanitized to a model this platform ships: a choice saved on a Mac must
    // not stick when the defaults land on a phone.
    private static func startModel() -> String {
        let saved = UserDefaults.standard.string(forKey: "modelName")
            ?? Models.start
        return Models.all.contains(saved) ? saved : Models.start
    }
    var modelName: String = ChatModel.startModel()
    // Non-nil surfaces the consent panel; cleared once the set is verified.
    var downloadName: String? = nil
    var downloading = false
    // BYTES, not a file count, so the bar and ETA track real throughput.
    var downloadDone: Int64 = 0
    var downloadTotal: Int64 = 0

    var downloadFraction: Double {
        downloadTotal > 0 ? Double(downloadDone) / Double(downloadTotal) : 0
    }

    var downloadCounter: String {
        String(format: "%.1f of %.1f GB",
               Double(downloadDone) / 1_000_000_000,
               Double(downloadTotal) / 1_000_000_000)
    }

    // Trimmed, not just !input.isEmpty: a field holding only spaces must
    // leave Send disabled and a stray Return doing nothing.

    var canSend: Bool {
        let text = input.trimmingCharacters(
            in: .whitespacesAndNewlines)
        let has = !text.isEmpty || !attachedImages.isEmpty
            || !attachedDocs.isEmpty || !attachedClips.isEmpty
        return ready && !busy && has
    }

    var typing: Bool { !input.isEmpty }

    // A FLOOR: it counts trunk prefill at roughly 4 chars a token, not the
    // per-tile tower encode. nil below a noticeable wait.
    var attachmentWarning: String? {
        let docTokens = attachedDocs.reduce(0) { $0 + $1.content.utf8.count / 4 }
        let total = attachedImages.count * perImageTokens + docTokens
        var result: String? = nil
        if total >= Self.warnTokens {
            let secs = lastPP > 0 ? Double(total) / lastPP : 0
            let time = secs >= 2 ? " (~\(Int(secs.rounded()))s)" : ""
            result = "Large attachment\(time); this may take a moment."
        }
        return result
    }

    // The CoreML tile path. Re-read from the backend after each build.
    private(set) var modelSupportsVision = false

    // Tower features taken directly (gemma-4's soft tokens). A DIFFERENT
    // capability from modelSupportsVision: no backend offers both, and the
    // two feed different send paths.
    private(set) var modelSupportsSoftTokens = false
    // Held for the session so the vision and audio towers are built once.
    @ObservationIgnored var media: Gemma4Media?

    var canAttachImages: Bool {
        modelSupportsVision || modelSupportsSoftTokens
    }
    var canAttachAudio: Bool { modelSupportsSoftTokens }
    // One video per turn: the picker stops offering them rather than
    // accepting a file it would silently drop.
    var canAttachVideo: Bool {
        modelSupportsSoftTokens
            && !attachedClips.contains { c in c.isVideo }
    }

    // Docs ride every model, and so do the formats that become one: a PDF or
    // an office file is read into Markdown before it is attached.
    var attachableTypes: [UTType] {
        var out: [UTType] = [.plainText, .pdf]
        out += Docs2md.readable.compactMap { ext in
            UTType(filenameExtension: ext)
        }
        if canAttachImages { out.append(.image) }
        if canAttachAudio { out.append(.audio) }
        if canAttachVideo { out.append(.movie) }
        return out
    }

    var hasAttachments: Bool {
        !attachedImages.isEmpty || !attachedClips.isEmpty
            || !attachedDocs.isEmpty
    }

    var attachGlyph: String {
        var out = "plus"
        if !attachedClips.isEmpty {
            out = attachedClips.contains(where: { c in c.isVideo })
                ? "film.fill" : "waveform"
        } else if !attachedImages.isEmpty {
            out = "photo.fill"
        } else if !attachedDocs.isEmpty {
            out = "doc.text.fill"
        }
        return out
    }

    var attachHelp: String {
        var out = "Attach a document"
        if canAttachAudio {
            out = "Attach an image, sound, video or document"
        } else if canAttachImages {
            out = "Attach an image or document"
        }
        return out
    }

    // Asked of the model's own chat template at session build: one whose
    // generation prompt bakes a closed empty <think> never thinks.
    private(set) var modelSupportsThinking = true

    // Everything that drives a turn reads THIS, so a non-reasoning model
    // cannot be put into a state the engine will ignore.
    var thinkingActive: Bool { thinking && modelSupportsThinking }

    var allowsTiling: Bool {
        !isOS && modelName != Models.fallback && modelSupportsVision
    }

    var downloadSizeText: String {
        let bytes = downloadName.flatMap { name in
            ModelCatalog.source(name)?.bytes
        } ?? 0
        return ByteCountFormatter.string(fromByteCount: bytes,
                                         countStyle: .file)
    }

    // Measured in each program's model.mil LoC as it finishes its ANE
    // compile, against a total computed up front.

    var compiling = false
    var compileDoneLoC = 0
    var compileTotalLoC = 0
    var compileFraction: Double {
        compileTotalLoC > 0
            ? min(1.0, Double(compileDoneLoC) / Double(compileTotalLoC)) : 0
    }

    var loadError: String? = nil

    // Defaults true, so the first frame errs toward the reassuring copy.
    var firstCompile = true

    // Two tiers, split by WHAT LEAVES THE DEVICE. `wikipedia` gates the
    // Wikimedia-only tools, whose lookup is matched on-device so nothing
    // typed is sent. `webAccess` gates search, fetch and weather, where the
    // model's search terms and an IP-derived location do go out.
    var wikipedia = true
    var webAccess = true

    enum Access { case offline, wikipedia, full }
    var accessState: Access {
        webAccess ? .full : (wikipedia ? .wikipedia : .offline)
    }

    func cycleAccess() {
        switch accessState {
        case .offline: setAccess(wikipedia: true, web: false)
        case .wikipedia: setAccess(wikipedia: true, web: true)
        case .full: setAccess(wikipedia: false, web: false)
        }
        switch accessState {
        case .offline: flashHUD("Airplane Mode")
        case .wikipedia: flashHUD("Wikipedia Only")
        case .full: flashHUD("Web Access")
        }
    }

    struct Flash: Equatable {
        let text: String
        // Larger, for a flash the user is WAITING on rather than one
        // confirming something they just did.
        let prominent: Bool
    }

    private(set) var hud: Flash?
    @ObservationIgnored private var hudTask: Task<Void, Never>?

    private func flashHUD(_ text: String, prominent: Bool = false,
                          seconds: Double = 3) {
        hudTask?.cancel()
        hud = Flash(text: text, prominent: prominent)
        hudTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(seconds))
            if !Task.isCancelled { hud = nil }
        }
    }

    func setAccess(wikipedia wiki: Bool, web: Bool) {
        wikipedia = wiki
        webAccess = web
        let s = session
        let r = toolRunner
        Task { await s?.setTools(r) }
    }
    var thinking = true
    // Applied only when the session is rebuilt, never mid-turn, so it cannot
    // mismatch a live KV prefix.
    var systemPrompt: String = UserDefaults.standard
        .string(forKey: "systemPrompt") ?? ChatModel.defaultSystemPrompt {
        didSet {
            UserDefaults.standard.set(systemPrompt, forKey: "systemPrompt")
            // Every precooked prefix bakes the prompt, so an edit makes them
            // all stale.
            ChatModel.wipePrecook()
        }
    }
    // Persisted PER MODEL: a tiled image on the 9B or 27B costs minutes of
    // tower encode and prefill, so those default to Fit.
    private static func visionKey(_ name: String) -> String {
        "visionMode.\(name)"
    }

    private static func visionMode(for name: String) -> VisionMode {
        let def: VisionMode = ["QwenPaw-Flash-9B", "Ternary-Bonsai-27B"]
            .contains(name) ? .fit : .tile
        let raw = UserDefaults.standard.string(forKey: visionKey(name)) ?? ""
        return VisionMode(rawValue: raw) ?? def
    }

    var visionMode: VisionMode =
        ChatModel.visionMode(for: ChatModel.startModel()) {
        didSet {
            UserDefaults.standard.set(visionMode.rawValue,
                                      forKey: Self.visionKey(modelName))
        }
    }
    // The docs are maintained regardless of the toggle, so flipping it
    // re-renders past turns too.
    var renderMarkdown: Bool = UserDefaults.standard
        .object(forKey: "renderMarkdown") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(renderMarkdown,
                                      forKey: "renderMarkdown")
        }
    }
    // Gates only the macOS trash: the iOS swipe is its own confirmation.
    var confirmDeleteConversation: Bool = UserDefaults.standard
        .object(forKey: "confirmDeleteConversation") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(confirmDeleteConversation,
                                      forKey: "confirmDeleteConversation")
        }
    }
    var statusLine: Bool = UserDefaults.standard.bool(forKey: "statusLine") {
        didSet { UserDefaults.standard.set(statusLine, forKey: "statusLine") }
    }
    var showSettings = false
    // The trace behind it is ALWAYS recording: drill-in is retroactive.
    var showDebug = false
    var traceEvents: [TraceEvent] = []
    // NOT built in a Release build: this file is the session's own content,
    // which is to say the user's conversation. Constructed rather than
    // silenced, because Diag.startRun creates the folder and archives the
    // previous run before a single line is written.
    private let traceFile: TraceFile? = debugBuild ? TraceFile() : nil
    private let instrument = Instrument.install()
    var tracePath: String { traceFile?.path ?? "" }
    var diagPath: String { Diag.shared.path }
    // A ring cap, so a marathon session cannot grow the array unbounded. The
    // on-disk transcript keeps everything.
    private static let traceCap = 2000
    // A static logit bias on reasoning branch-openers while thinking. Robust
    // across 0.5-4.0. TODO: surface as a setting.
    private static let overthinkLambda: Float = 1.0

    // Anchored in TIME, not tokens: the cap is derived per model and device
    // from the measured decode rate, so "M is about 20 seconds" holds on an
    // M3 Max and an iPad alike and stays true as the engine gets faster.
    enum ThinkBudget: String, CaseIterable, Identifiable {
        case xs = "XS", s = "S", m = "M", l = "L", xl = "XL"
        var id: String { rawValue }
        var seconds: Double {
            switch self {
            case .xs: return 5
            case .s: return 10
            case .m: return 20
            case .l: return 40
            case .xl: return 90
            }
        }
    }

    var thinkBudget: ThinkBudget = {
        let raw = UserDefaults.standard.string(forKey: "thinkBudget") ?? ""
        return ThinkBudget(rawValue: raw) ?? .m
    }() {
        didSet {
            UserDefaults.standard.set(thinkBudget.rawValue,
                                      forKey: "thinkBudget")
        }
    }

    private static func tgKey(_ name: String) -> String { "tg.\(name)" }

    // A conservative 20 t/s until the first turn records one.
    var measuredTG: Double {
        let v = UserDefaults.standard.double(forKey: Self.tgKey(modelName))
        return v > 0 ? v : 20
    }

    // EMA, so one outlier turn cannot yank the budget around.
    private func recordTG(_ tg: Double) {
        if tg > 0 {
            let old = UserDefaults.standard.double(
                forKey: Self.tgKey(modelName))
            let ema = old > 0 ? 0.7 * old + 0.3 * tg : tg
            UserDefaults.standard.set(ema, forKey: Self.tgKey(modelName))
        }
    }

    // Rounded to 100, so the figure shown is not falsely precise. The hard
    // runaway backstop is twice this.
    var thinkTokenCap: Int {
        let raw = thinkBudget.seconds * measuredTG
        return max(200, Int((raw / 100).rounded()) * 100)
    }

    private let log = Logger(subsystem: "io.github.leok7v.gadeon",
                              category: "ui")
    private var chat: AneChat?
    // Set INSTEAD of `chat` for a GGUF model. Vision and the ANE heavy/carry
    // sets do not apply there, which `chat` being nil is what guards.
    private var ggufBackend: (any AgentBackend)?
    private var ggufTemplate = ""
    private var ggufVocabCount = 0
    // One per conversation: recreated on model switch, reset on New Chat.
    private var session: ChatSession?
    // Held so Stop can cancel it. Generation is otherwise unbounded.
    private var genTask: Task<Void, Never>?
    // Compiles each downloaded .mlmodelc in-process WHILE the rest of the set
    // still streams, so the one-time compile overlaps the download.
    private let primer = Primer()
    // Learned from the last turn; 0 until one has measured it.
    private var lastPP = 0.0
    // A baseline until the first image turn learns the tower's real
    // merged-token count.
    private var perImageTokens = 256
    // Captured at build, so makeSession can rebuild without re-reading the
    // model directory.
    private var activePresets: SamplingPresets = .qwen35
    @ObservationIgnored private var phaseStart = Date()
    // EMA-smoothed across the per-program estimates so it tracks BOTH
    // directions: a warm early phase must not pin it under a cold later one.
    @ObservationIgnored private var optimizeFinish: Date?

    // The fetch does NOT start here. The App Store requires explicit consent,
    // so it begins only when Download is tapped on the consent panel.
    func acceptEULA() {
        eulaAccepted = true
        UserDefaults.standard.set(true, forKey: "eulaAccepted")
        load(name: modelName)
    }

    func accept() {
        accepted = true
        UserDefaults.standard.set(true, forKey: "disclaimerAccepted")
    }

    // Separate from the EULA and from the size-consent panel: it binds the
    // user to a THIRD party's terms, which is a different decision from
    // agreeing to spend the bytes.
    var gemmaTermsAccepted = GemmaTerms.accepted

    func acceptGemmaTerms() {
        GemmaTerms.accept()
        gemmaTermsAccepted = true
    }

    func load(name: String) {
        loadError = nil
        ChatModel.pruneUnavailable()
        let setDir = ModelCatalog.localSet(name, in: Bundle.modelStore())
        if let setDir, ModelCatalog.isComplete(setDir) {
            if ModelCatalog.isGguf(name),
               let path = ModelCatalog.ggufPath(name, in: Bundle.modelStore()) {
                buildGguf(name: name, path: path)
            } else {
                build(setDir: setDir)
            }
        } else if ModelCatalog.source(name) != nil {
            downloadName = name
            status = "download required"
        } else {
            loadError = "Model \(name) is not available."
        }
    }

    // Only the Sendable backend, template, vocab and sampler recs cross back
    // to the main actor; the non-Sendable engine stays inside the backend.
    private func buildGguf(name: String, path: String) {
        compiling = true
        compileDoneLoC = 0
        compileTotalLoC = 0
        phaseStart = Date()
        loadError = nil
        // No ANE compile here: the wait is an mmap and a GPU upload.
        firstCompile = false
        // DROP THE OUTGOING MODEL BEFORE MAPPING THE NEW ONE, or a switch
        // needs room for the sum of both. `session` retains the backend, so
        // clearing it first is what actually lets the backend go.
        session = nil
        chat = nil
        media = nil
        ggufBackend = nil
        Task.detached { [weak self] in
            var built: (backend: any AgentBackend, template: String,
                        vocab: Int, presets: SamplingPresets,
                        shape: ModelShape)?
            var media: Gemma4Media? = nil
            Footprint.report("before \(name)")
            do {
                // The FILE says which engine it needs, by its own
                // general.architecture. Never the catalog name, which is a
                // label we chose.
                if Gemma4Model.isGemma4(path: path) {
                    let c = try GemmaChat(ggufPath: path)
                    let gpu = try c.metalBackend()
                    built = (gpu, c.chatTemplate,
                             c.vocabCount, c.samplingPresets, c.shape)
                    // Over the text engine's own mapping, never a second one.
                    media = c.media(ctx: gpu.ctx)
                } else {
                    let c = try MetalChat(ggufPath: path)
                    built = (c.backend(), c.chatTemplate,
                             c.tokenizer.vocabCount, c.samplingPresets,
                             c.shape)
                }
            } catch {
                // The reason cannot be inferred from a footprint log: the
                // whole GGUF is one bytesNoCopy MTLBuffer, and Metal's
                // maxBufferLength is a hard per-device ceiling a large set
                // can exceed.
                Diag.shared.report("model prep FAILED \(name): \(error)")
                built = nil
            }
            Footprint.report("loaded \(name)")
            let vision = await built?.backend.supportsVision() ?? false
            let soft = await built?.backend.supportsSoftTokens() ?? false
            await MainActor.run {
                if let b = built {
                    self?.ggufBackend = b.backend
                    self?.ggufTemplate = b.template
                    self?.ggufVocabCount = b.vocab
                    self?.activePresets = b.presets
                    self?.modelSupportsVision = vision
                    self?.modelSupportsSoftTokens = soft
                    self?.modelShape = b.shape
                    self?.media = media
                    Instrument.timed("makeSession") { self?.makeSession() }
                    Instrument.timed("show chat") {
                        self?.compiling = false
                        self?.status = ""
                    }
                    self?.primeSession()
                } else {
                    self?.compiling = false
                    self?.loadError = Self.prepFailed
                }
            }
        }
    }

    // Wipes the whole compile cache first, so a program corrupted by an OS
    // reboot mid-compile is thrown away rather than reloaded.

    func retry() {
        loadError = nil
        clearCompileCache()
        load(name: modelName)
    }

    func confirmDownload() {
        if let name = downloadName, let src = ModelCatalog.source(name) {
            commitSwitch(name)
            downloadName = nil
            downloading = true
            downloadDone = 0
            downloadTotal = 0
            phaseStart = Date()
            status = "downloading \(name)…"
            let dest = Bundle.modelStore().appendingPathComponent(name)
            // In place, so files land at their final cache-keyed paths and
            // the primer can compile each finished program while later files
            // still stream.
            let setDir = dest.appendingPathComponent(src.revision)
            Task.detached { [weak self] in
                // Snapshot BEFORE the primer's first compile, so the streamed
                // set's programs land in its ownership claim.
                AneCache.shared.downloadBegan()
                var set: URL? = nil
                var failure = "download failed, check your connection"
                do {
                    set = try await HubFetch.fetch(
                        repo: src.repo, into: dest, revision: src.revision,
                        files: src.files, excludeFromBackup: true) { s in
                        Task { @MainActor in
                            self?.downloadDone = s.done
                            self?.downloadTotal = s.total
                            self?.primer.observe(s.file, set: setDir)
                        }
                    }
                } catch HubError.digest {
                    // Corruption, not connectivity.
                    failure = "download failed verification, try again"
                } catch {
                }
                await MainActor.run {
                    // The build compiles the remainder itself from here, so
                    // the helper must not contend for the serial ANE
                    // compiler.
                    self?.primer.cancel()
                    self?.downloading = false
                    if let self, let set {
                        Self.saveSeconds("download", name,
                            Date().timeIntervalSince(self.phaseStart))
                        if ModelCatalog.isGguf(name), let path =
                            ModelCatalog.ggufPath(name, in: Bundle.modelStore()) {
                            self.buildGguf(name: name, path: path)
                        } else {
                            self.build(setDir: set)
                        }
                    } else {
                        self?.status = failure
                    }
                }
            }
        }
    }

    // Waits for the WHOLE set to finish compiling before enabling chat: no
    // half-compiled partial is ever served.

    private func build(setDir: URL, attempt: Int = 0) {
        compiling = true
        compileDoneLoC = 0
        compileTotalLoC = 0
        phaseStart = Date()
        optimizeFinish = nil
        activePresets = Self.presets(setDir)
        loadError = nil
        let key = Self.compiledKey(setDir)
        // A container migration forces a cold recompile whatever the
        // compiled-once flag says: the e5 cache keys bind to container
        // identity.
        firstCompile = !UserDefaults.standard.bool(forKey: key)
            || AneCache.shared.containerMigrated()
        Task.detached { [weak self] in
            // Snapshot BEFORE the first MLModel touch, so everything this
            // build compiles lands in the set's ownership claim.
            let cacheBefore = AneCache.shared.buildBegan()
            // A restore lands in a dir the e5 daemon does not trust unless it
            // created it, so compile the smallest program first and relink
            // the shadow's inodes only after that.
            Engine.primeCacheDir(setDir)
            AneCache.shared.restoreFromShadow()
            let total = Engine.compileTotalLoC(setDir)
            await MainActor.run { self?.compileTotalLoC = total }
            let report: @Sendable (Int) -> Void = { loc in
                Task { @MainActor in
                    self?.compileDoneLoC += loc
                    self?.tightenOptimizeETA()
                }
            }
            let built = try? AneChat(modelsDir: setDir, onCompiledLoC: report)
            let vision = await built?.engine.supportsVision() ?? false
            await MainActor.run {
                self?.chat = built
                if let built {
                    self?.modelSupportsVision = vision
                    self?.modelSupportsSoftTokens = false
                    self?.modelShape = built.shape
                    self?.media = nil
                    self?.makeSession()
                } else {
                    self?.buildFailed(setDir, attempt)
                }
            }
            if let built {
                do {
                    try await built.engine.warmup()
                    try await built.loadHeavy()
                    try await built.loadCarry()
                    // A silent no-op unless the set ships the drafter.
                    await built.loadMTP()
                    // The tower compiled during essential load; drop its
                    // residency so text-only serving does not hold it.
                    await built.engine.offloadVision()
                    AneCache.shared.buildEnded(setDir: setDir,
                                               before: cacheBefore)
                    await MainActor.run {
                        // Only a COLD compile times the real optimize; a warm
                        // relaunch is instant and must not overwrite it.
                        if let self, self.firstCompile {
                            Self.saveSeconds("optimize", self.modelName,
                                Date().timeIntervalSince(self.phaseStart))
                        }
                        self?.compiling = false
                        self?.status = ""
                        UserDefaults.standard.set(true, forKey: key)
                        self?.primeSession()
                    }
                } catch {
                    await MainActor.run {
                        self?.chat = nil
                        self?.session = nil
                        self?.buildFailed(setDir, attempt)
                    }
                }
            }
        }
    }

    // A relinked bundle may be corrupt or stale, so the cache and shadow are
    // purged and rebuilt ONCE. `attempt` caps that, so a genuine
    // unsupported-device failure still surfaces.
    private func buildFailed(_ setDir: URL, _ attempt: Int) {
        if AneCache.cacheMode == .hardlink, attempt == 0 {
            AneCache.shared.purgeSet(setDir)
            build(setDir: setDir, attempt: 1)
        } else {
            chat = nil
            session = nil
            compiling = false
            loadError = Self.prepFailed
        }
    }

    // Three things are folded in and each is load-bearing: the {sha} leaf
    // identifies the set, the mf0 graph's .mil size makes a re-emitted graph
    // read as a fresh compile, and the OS version because the e5 cache is
    // keyed by OS build, so an update recompiles everything.

    private static func compiledKey(_ setDir: URL) -> String {
        let n = Engine.programCount(setDir)
        let mil = setDir.appendingPathComponent("mf0of\(n).mlmodelc")
            .appendingPathComponent("model.mil")
        let attrs = try? FileManager.default.attributesOfItem(atPath: mil.path)
        let size = (attrs?[.size] as? Int) ?? 0
        let os = ProcessInfo.processInfo.operatingSystemVersion
        return "compiled.\(setDir.lastPathComponent).\(size)"
            + ".\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"
    }

    // An un-downloaded set surfaces the consent panel WITHOUT tearing down
    // the current model, so Cancel returns to it.

    func switchModel(_ name: String) {
        if name != modelName, !busy, Models.all.contains(name) {
            if isOnDisk(name) {
                commitSwitch(name)
                status = "loading model…"
                load(name: name)
            } else {
                downloadName = name
            }
        }
    }

    // At STARTUP the persisted choice may be un-downloaded with nothing
    // loaded, so a set already on disk is what keeps Cancel from stranding
    // the user on a mandatory download.

    func cancelDownload() {
        let fallback = ready ? nil : downloadedFallback()
        downloadName = nil
        if let fallback {
            commitSwitch(fallback)
            status = "loading model…"
            load(name: fallback)
        }
    }

    var canCancelDownload: Bool { ready || downloadedFallback() != nil }

    private func downloadedFallback() -> String? {
        isOnDisk(Models.start)
            ? Models.start : Models.all.first { isOnDisk($0) }
    }

    // Tears down the current engine and conversation, so it is taken only
    // when a switch is actually chosen, never while the consent panel is
    // still cancellable.

    private func commitSwitch(_ name: String) {
        genTask?.cancel()
        // A cook still running on the outgoing engine would hold that whole
        // mapping alive for work whose result is about to be discarded.
        session?.requestStop()
        primer.cancel()
        chat = nil
        ggufBackend = nil
        session = nil
        // The new model may have no eyes at all; both re-probe after build.
        modelSupportsVision = false
        modelSupportsSoftTokens = false
        modelShape = nil
        media = nil
        for img in attachedImages {
            input = AttachmentRefs.scrub(img.name, from: input)
        }
        attachedImages = []
        attachedClips = []
        clampCaret()
        downloading = false
        compiling = false
        compileDoneLoC = 0
        loadError = nil
        messages = []
        generatedTitle = nil
        statsLabel = ""
        modelName = name
        UserDefaults.standard.set(name, forKey: "modelName")
        visionMode = ChatModel.visionMode(for: name)
    }

    private func isOnDisk(_ name: String) -> Bool {
        var result = false
        if let local = ModelCatalog.localSet(name, in: Bundle.modelStore()) {
            result = ModelCatalog.isComplete(local)
        }
        return result
    }

    // Ticks on a delete, because isOnDisk reads no observable state and the
    // pane would otherwise not re-read the disk.
    private(set) var diskRevision = 0

    func isDownloaded(_ name: String) -> Bool { isOnDisk(name) }

    // The ACTIVE model is not deletable: the running engine mmaps it.
    func deleteModel(_ name: String) {
        if name != modelName, !busy, !downloading {
            ChatModel.erase(name)
            diskRevision += 1
        }
    }

    // Everything one model owns: the set, its precooked prefix, and every
    // size / OS variant of its compiled-once flag, so a re-download presents
    // as a fresh install rather than a warm load.
    private static func erase(_ name: String) {
        let fm = FileManager.default
        try? fm.removeItem(
            at: Bundle.modelStore().appendingPathComponent(name))
        try? fm.removeItem(at: precookURL(name))
        if let sha = ModelCatalog.source(name)?.revision {
            let d = UserDefaults.standard
            for k in d.dictionaryRepresentation().keys
            where k.hasPrefix("compiled.\(sha).") {
                d.removeObject(forKey: k)
            }
        }
    }

    // A set this device is not offered is unreachable, so its gigabytes have
    // no route off the device short of a reinstall. Anything erased returns
    // on demand; an offered model is never touched.
    private static func pruneUnavailable() {
        let offered = Set(Models.all)
        // Empty means the device runs nothing at all, not that every set is
        // dead: pruning on that would wipe the store.
        if !offered.isEmpty {
            let names = (try? FileManager.default.contentsOfDirectory(
                atPath: Bundle.modelStore().path)) ?? []
            for name in names where !offered.contains(name) {
                erase(name)
            }
        }
    }

    func requestDownload(_ name: String) {
        if !busy, !downloading, !isOnDisk(name),
           ModelCatalog.source(name) != nil {
            downloadName = name
        }
    }

    // Best-effort: absent means no wikipedia_query tool, and the app still
    // works without grounding.
    private static let minilmPath: String? = WikiSlugs.bundledModel?.path

    // get_current_time and calculator are local, so they are offered even in
    // airplane mode.
    private var toolRunner: (any ToolRunner)? {
        SafeToolRunner(slugsPath: wikipedia ? ChatModel.minilmPath : nil,
                       wikipedia: wikipedia, network: webAccess)
    }

    // Regions wholly or predominantly south of the equator, so the stated
    // season flips there. Resolved from the locale rather than asked.
    private static let southernRegions: Set<String> = [
        "AU", "NZ", "AR", "CL", "UY", "PY", "BO", "PE", "BR", "ZA", "NA",
        "BW", "ZW", "ZM", "MZ", "MG", "LS", "SZ", "AO", "MW", "PG", "FJ",
        "NC", "WS", "TO", "VU", "SB",
    ]

    // The prompt SPLITS for the precooked-prefix cache. This half is
    // everything whose bytes hold across sessions, and is what gets
    // precooked; the date/time tail changes every minute and would poison the
    // cache stamp, so it is laid separately after a restore.
    private var systemStable: String {
        let loc = Locale.current
        let region = loc.region?.identifier ?? "unknown"
        let units = loc.measurementSystem == .metric
            ? "metric (Celsius, kilometers)"
            : "US customary (Fahrenheit, miles)"
        var s = systemPrompt
        s += "\nUser locale: region \(region), timezone "
            + "\(TimeZone.current.identifier), \(units) units. Give "
            + "temperatures and distances in these units unless asked otherwise."
        if !webAccess && !wikipedia {
            s += "\nYou are offline: no web search or Wikipedia lookup is "
                + "available. Answer from your own knowledge; do not call any "
                + "search or lookup tool."
        } else if !webAccess {
            s += "\nYou have no web search or page fetching. You may consult "
                + "Wikipedia (wikipedia_query) and today's headlines "
                + "(get_news); for anything else answer from your own "
                + "knowledge."
        }
        // Speech reaches the model as its own tower's output and it does not
        // know that, so it argues itself into "I cannot process audio" and
        // then into a refusal that poisons every later turn. Asserting this
        // up front is the lever that works; correcting it afterwards does
        // not.
        if canAttachAudio {
            s += "\nThe user may speak to you. Their speech reaches you "
                + "already encoded by your own audio tower, so you hear it "
                + "directly: never say you cannot process audio, and never "
                + "treat it as a transcript someone pasted. Spoken words are "
                + "the user talking TO you -- answer them as you would the "
                + "same words typed, and write them out only when asked to."
        }
        // A text-tuned checkpoint with a grafted tower reflexively claims it
        // cannot see images even while describing one.
        if canAttachImages {
            s += "\nThe user may attach images. Each is encoded by your "
                + "vision tower and fully visible to you: describe what you "
                + "actually see, and never claim you cannot view images. "
                + "Images are letterboxed onto a square canvas; a flat "
                + "neutral-grey border or region is padding, not image "
                + "content -- do not describe it."
        }
        return s
    }

    private var systemDynamic: String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd hh:mm a zzz"
        // Stated outright: a small model does not reliably infer the season
        // from a date.
        let region = Locale.current.region?.identifier ?? "unknown"
        let south = ChatModel.southernRegions.contains(region)
        let month = Calendar.current.component(.month, from: Date()) - 1
        let season = ["winter", "winter", "spring", "spring", "spring",
                      "summer", "summer", "summer", "autumn", "autumn",
                      "autumn", "winter"][south ? (month + 6) % 12 : month]
        return "\nCurrent Date and Time: \(f.string(from: Date())) "
            + "(\(south ? "southern" : "northern")-hemisphere \(season))"
    }

    private func recordTrace(_ e: TraceEvent) {
        traceEvents.append(e)
        if traceEvents.count > Self.traceCap {
            traceEvents.removeFirst(traceEvents.count - Self.traceCap)
        }
        traceFile?.append(e)
    }

    private func hookTrace() {
        traceFile?.note("=== \(modelName) thinking=\(thinkingActive) "
            + "wiki=\(wikipedia) web=\(webAccess) \(Date())")
        let s = session
        Task {
            await s?.setTrace { [weak self] e in
                Task { @MainActor in self?.recordTrace(e) }
            }
        }
        Tools.setDiagSink { [weak self] msg in
            Task { @MainActor in
                self?.recordTrace(TraceEvent(
                    kind: .diag, t0: Date(), t1: Date(), ctx: -1,
                    tokens: 0, summary: String(msg.prefix(80)), text: msg))
            }
        }
    }

    private func makeSession() {
        if let chat {
            modelSupportsThinking =
                templateSupportsThinking(chat.chatTemplate)
            session = ChatSession(
                backend: EngineBackend(chat),
                template: chat.chatTemplate,
                system: systemStable,
                systemTail: systemDynamic,
                vocabSize: chat.tokenizer.vocabCount,
                presets: activePresets,
                enableThinking: thinkingActive,
                maxReasoning: thinkTokenCap * 2,
                softReasoningCap: thinkTokenCap,
                overthink: Self.overthinkLambda,
                runner: toolRunner)
            hookTrace()
        } else if let ggufBackend {
            modelSupportsThinking = templateSupportsThinking(ggufTemplate)
            session = ChatSession(
                backend: ggufBackend,
                template: ggufTemplate,
                system: systemStable,
                systemTail: systemDynamic,
                vocabSize: ggufVocabCount,
                presets: activePresets,
                enableThinking: thinkingActive,
                maxReasoning: thinkTokenCap * 2,
                softReasoningCap: thinkTokenCap,
                overthink: Self.overthinkLambda,
                runner: toolRunner)
            hookTrace()
        }
    }

    // One state file per model: the rendered system + tools prefix, prefilled
    // once and restored at every session build. The stamp inside the file
    // self-invalidates when the prompt, tools tier, locale or template
    // change.
    private static var precookDir: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? "app")
            .appendingPathComponent("precook", isDirectory: true)
    }

    private static func precookURL(_ name: String) -> URL {
        precookDir.appendingPathComponent(name + ".ctx")
    }

    static func wipePrecook() {
        try? FileManager.default.removeItem(at: precookDir)
    }

    // primeOrCook registers as the session's priming gate, so an early send
    // WAITS it out instead of interleaving with it.
    private func primeSession(resetFirst: Bool = false) {
        if let s = session {
            let url = ChatModel.precookURL(modelName)
            try? FileManager.default.createDirectory(
                at: ChatModel.precookDir, withIntermediateDirectories: true)
            Task { await s.primeOrCook(at: url, resetFirst: resetFirst) }
        }
    }

    func newChat() {
        lastTurnSpoken = false
        speech.stopSpeaking()
        commitCurrent()
        readOnly = false
        currentConversationId = nil
        generatedTitle = nil
        messages = []
        traceEvents = []
        statsLabel = ""
        // The outgoing session may still be cooking its prefix or running a
        // turn, and the fresh one resets the engine they SHARE, so both have
        // to be off it first. Held in genTask, so `busy` covers the swap.
        //
        // Cancelling alone does not land: a Metal forward is synchronous and
        // passes no cancellation point, so the stop FLAG is what reaches it
        // and the await is what makes "off the engine" true rather than
        // merely requested.
        let outgoing = session
        let running = genTask
        genTask = Task { @MainActor in
            if running != nil {
                outgoing?.requestStop()
                running?.cancel()
                _ = await running?.value
            }
            await outgoing?.endPriming()
            makeSession()
            primeSession(resetFirst: true)
            genTask = nil
        }
    }

    // Routed through genTask so the composer is disabled while it runs:
    // makeTitle holds the engine, and a concurrent send would interleave two
    // turns on one KV.
    private func maybeGenerateTitle() {
        let chars = messages.reduce(0) { sum, m in sum + m.text.count }
        let worth = !readOnly && generatedTitle == nil && genTask == nil
            && messages.count >= 2 && chars > 200
        if worth, let session {
            genTask = Task { @MainActor in
                let t = await session.makeTitle()
                if !t.isEmpty {
                    generatedTitle = t
                    commitCurrent()
                }
                genTask = nil
            }
        }
    }

    // Opening one dismisses the other: the routing shows debug first, so a
    // stale flag would shadow the requested view.

    func toggleSettings() {
        showSettings.toggle()
        if showSettings { showDebug = false }
    }

    // OPENS where the gear toggles: a second Command-comma must not close
    // what the first one asked for.

    func openSettings() {
        showDebug = false
        showSettings = true
    }

    func toggleDebug() {
        showDebug.toggle()
        if showDebug { showSettings = false }
    }

    // Takes effect next turn.

    func toggleThinking() {
        if modelSupportsThinking {
            thinking.toggle()
            let s = session
            let on = thinking
            Task { await s?.setThinking(on) }
            flashHUD(thinking ? "Thinking: On" : "Thinking: Off")
        }
    }

    func quickAnswer() {
        let s = session
        Task { await s?.requestQuickAnswer() }
    }

    // `at` is the UTF-16 offset the inline reference is inserted at.
    func attachImage(_ data: Data, name: String, at offset: Int) {
        let dup = attachedImages.contains { $0.data == data }
        if canAttachImages, attachedImages.count < Self.maxImages, !dup {
            let unique = uniqueName(name)
            // 96px covers a ~24pt chip at 3x, decoded straight to size so
            // even a huge source never materializes its full bitmap.
            let thumb = VisionPreprocess.thumbnail(data, maxPx: 96)
            attachedImages.append(
                ImageAttachment(name: unique, data: data, thumbnail: thumb))
            insertRef(unique, at: offset)
        }
    }

    // Only the URL is kept: the decode happens at send, so a long video never
    // sits in memory waiting to be asked about.
    func attachClip(_ url: URL, isVideo: Bool, at offset: Int) {
        let allowed = isVideo ? canAttachVideo : canAttachAudio
        let dup = attachedClips.contains { c in c.url == url }
        // ONE video per turn, where the general clip cap is two: a video is
        // 32 frames through the tower, and the composer can only show one of
        // them being looked at anyway.
        let seenVideo = isVideo
            && attachedClips.contains { c in c.isVideo }
        if allowed, attachedClips.count < Self.maxClips, !dup, !seenVideo {
            let unique = uniqueName(url.lastPathComponent)
            attachedClips.append(ClipAttachment(
                name: unique, url: url, isVideo: isVideo,
                thumbnail: isVideo ? nil : nil))
            insertRef(unique, at: offset)
        }
    }

    func attachDoc(_ name: String, _ content: String, at offset: Int,
                   from url: URL? = nil) {
        let capped = Self.capDoc(content, docBudget.bytes)
        if !attachedDocs.contains(where: { $0.content == capped }) {
            let unique = uniqueName(name)
            attachedDocs.append(
                Doc(name: unique, content: capped, url: url,
                    short: content.utf8.count > docBudget.bytes))
            insertRef(unique, at: offset)
        }
    }

    // Markdown BEFORE it is attached, so nothing downstream has to know where
    // the text came from. The file is COPIED first and the copy is what the
    // transcript keeps: neither the security scope a drop holds nor the
    // original path outlives this turn, and a sandboxed app cannot reopen a
    // dropped URL later.
    private func convertDoc(_ from: URL, _ name: String) {
        if let kept = ChatModel.keep(from, name) {
            converting += 1
            flashHUD("Reading \(name)")
            let t0 = Date()
            Task { @MainActor in
                let text = await ChatModel.markdown(of: kept)
                converting -= 1
                ChatModel.read(name, text, t0,
                               docBudget.bytes)
                if let text {
                    attachDoc(name, text, at: caret, from: kept)
                } else {
                    try? FileManager.default.removeItem(at: kept)
                    flashHUD("Cannot read \(name)")
                }
            }
        }
    }

    // Truncation is called out because the model then saw a PREFIX of the
    // document, and nothing downstream says so.
    private static func read(_ name: String, _ text: String?,
                             _ since: Date, _ limit: Int) {
        let bytes = text?.utf8.count ?? 0
        let cut = bytes > limit ? ", TRUNCATED to \(limit)" : ""
        Diag.shared.report(String(
            format: "attach document %@ -> %d bytes%@ (%.1fs)", name, bytes,
            cut, Date().timeIntervalSince(since)))
    }

    // Application Support, not Caches: it is preserved across upgrades where
    // Caches is purgeable.
    static let attachments: URL = {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask, appropriateFor: nil,
                                   create: true)) ?? fm.temporaryDirectory
        let dir = support.appendingPathComponent("attachments",
                                                 isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    // A directory of its own rather than a unique PREFIX, so the file keeps
    // the name it arrived with: a prefix would show in Quick Look's own
    // title.
    private static func keep(_ from: URL, _ name: String) -> URL? {
        let fm = FileManager.default
        let dir = attachments.appendingPathComponent(UUID().uuidString,
                                                     isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let to = dir.appendingPathComponent(name)
        return (try? fm.copyItem(at: from, to: to)) != nil ? to : nil
    }

    // Off the main actor: a scanned PDF puts every page through recognition,
    // and the office readers walk a whole archive.
    private static func markdown(of url: URL) async -> String? {
        await Task.detached(priority: .userInitiated) {
            url.pathExtension.lowercased() == "pdf"
                ? try? await Pdf2md.markdown(of: url)
                : try? Docs2md.markdown(of: url)
        }.value
    }

    // A doc whose file could not be kept is dropped rather than shown as a
    // chip that opens nothing.
    private static func refs(_ docs: [Doc]) -> [DocRef] {
        docs.compactMap { doc in
            doc.url.map { url in
                DocRef(url: url, bytes: doc.content.utf8.count,
                       short: doc.short)
            }
        }
    }

    // Cut on BYTES, backed off to the last whole character. Counting
    // Characters would let multi-byte text through at several times the
    // bound, and a byte prefix alone can land inside a scalar, which String
    // refuses to make -- hence the step back of at most three bytes.
    private static func capDoc(_ content: String, _ limit: Int) -> String {
        var out = content
        if content.utf8.count > limit {
            var take = limit
            var head: String? = nil
            while head == nil && take > 0 {
                head = String(content.utf8.prefix(take))
                take -= 1
            }
            out = (head ?? "")
                + "\n[... truncated at \(limit >> 10) KB]"
        }
        return out
    }

    func clearImage(_ id: UUID) {
        if let img = attachedImages.first(where: { $0.id == id }) {
            input = AttachmentRefs.scrub(img.name, from: input)
        }
        attachedImages.removeAll { $0.id == id }
        clampCaret()
    }

    func clearClip(_ id: UUID) {
        if let clip = attachedClips.first(where: { c in c.id == id }) {
            input = AttachmentRefs.scrub(clip.name, from: input)
        }
        attachedClips.removeAll { c in c.id == id }
        clampCaret()
    }

    func clearDoc(_ id: UUID) {
        if let doc = attachedDocs.first(where: { $0.id == id }) {
            input = AttachmentRefs.scrub(doc.name, from: input)
        }
        attachedDocs.removeAll { $0.id == id }
        clampCaret()
    }

    // Called on every input change. The scrub is idempotent, so the
    // re-entrant run it triggers settles.
    func reconcileAttachments() {
        let live = Set(AttachmentRefs.names(in: input))
        for doc in attachedDocs where !live.contains(doc.name) {
            input = AttachmentRefs.scrub(doc.name, from: input)
        }
        attachedDocs.removeAll { !live.contains($0.name) }
        for img in attachedImages where !live.contains(img.name) {
            input = AttachmentRefs.scrub(img.name, from: input)
        }
        attachedImages.removeAll { !live.contains($0.name) }
        for clip in attachedClips where !live.contains(clip.name) {
            input = AttachmentRefs.scrub(clip.name, from: input)
        }
        attachedClips.removeAll { c in !live.contains(c.name) }
        clampCaret()
    }

    private func insertRef(_ name: String, at offset: Int) {
        let r = AttachmentRefs.insert(name, into: input, at: offset)
        input = r.text
        caret = r.caret
    }

    private func clampCaret() {
        caret = max(0, min(caret, input.utf16.count))
    }

    // Unique across docs, images AND clips: two attachments sharing a
    // filename must stay distinct inline references.
    private func uniqueName(_ name: String) -> String {
        var candidate = name
        var n = 2
        while attachedDocs.contains(where: { d in d.name == candidate })
            || attachedImages.contains(where: { i in i.name == candidate })
            || attachedClips.contains(where: { c in c.name == candidate }) {
            candidate = "\(name) (\(n))"
            n += 1
        }
        return candidate
    }

    static let imageExts: Set<String> =
        ["png", "jpg", "jpeg", "heic", "heif", "gif", "webp", "bmp", "tiff"]
    static let docExts: Set<String> = ["txt", "md", "markdown", "text"]
    // Read into Markdown rather than read as text.
    static let convertExts: Set<String> = Set(["pdf"] + Docs2md.readable)
    // The SYSTEM's own type conformance, never a list here: the file panel
    // filters on UTType.audio / .movie, so an extension list could only
    // disagree with what it offers. The DECODE is the arbiter.
    static func clipKind(_ url: URL) -> Bool? {
        let type = UTType(filenameExtension:
                            url.pathExtension.lowercased())
        var out: Bool? = nil
        if type?.conforms(to: .movie) == true {
            out = true
        } else if type?.conforms(to: .audio) == true {
            out = false
        }
        return out
    }
    static let maxDocs = 6
    static let maxImages = 4
    // One clip is already hundreds to thousands of soft tokens.
    static let maxClips = 2
    // Roughly 4 bytes to a token, so M is around 8,000 of them.
    enum DocBudget: String, CaseIterable, Identifiable {
        case xs = "XS", s = "S", m = "M", l = "L", xl = "XL"
        var id: String { rawValue }
        var bytes: Int {
            switch self {
            case .xs: return 8 << 10
            case .s: return 16 << 10
            case .m: return 32 << 10
            case .l: return 64 << 10
            case .xl: return 128 << 10
            }
        }
        var label: String { "\(bytes >> 10) KB" }
    }

    var docBudget: DocBudget = {
        let raw = UserDefaults.standard.string(forKey: "docBudget") ?? ""
        return DocBudget(rawValue: raw) ?? .m
    }() {
        didSet {
            UserDefaults.standard.set(docBudget.rawValue, forKey: "docBudget")
        }
    }
    static let warnTokens = 4000

    func handleDrop(_ urls: [URL], at offset: Int) {
        caret = max(0, min(offset, input.utf16.count))
        var refused: [String] = []
        for url in urls where ready {
            let ext = url.pathExtension.lowercased()
            let scoped = url.startAccessingSecurityScopedResource()
            let before = attachedImages.count + attachedDocs.count
                + attachedClips.count + converting
            if Self.imageExts.contains(ext),
               attachedImages.count < Self.maxImages,
               let data = try? Data(contentsOf: url) {
                attachImage(data, name: url.lastPathComponent, at: caret)
            } else if Self.docExts.contains(ext),
                      attachedDocs.count < Self.maxDocs,
                      let text = try? String(contentsOf: url, encoding: .utf8) {
                attachDoc(url.lastPathComponent, text, at: caret,
                          from: Self.keep(url, url.lastPathComponent))
            } else if Self.convertExts.contains(ext),
                      attachedDocs.count < Self.maxDocs {
                convertDoc(url, url.lastPathComponent)
            } else if let isVideo = Self.clipKind(url) {
                attachClip(url, isVideo: isVideo, at: caret)
            }
            // A file still being READ counts as accepted, or every
            // convertible drop reports itself refused a second before it
            // arrives.
            let after = attachedImages.count + attachedDocs.count
                + attachedClips.count + converting
            if after == before { refused.append(ext.isEmpty ? "file" : ext) }
            if scoped { url.stopAccessingSecurityScopedResource() }
        }
        if !refused.isEmpty { flashHUD(refusedText(refused)) }
    }

    private func refusedText(_ kinds: [String]) -> String {
        let what = Set(kinds).sorted()
            .map { k in "." + k }.joined(separator: " ")
        var out = "Cannot read \(what)"
        if kinds.contains(where: { k in Self.imageExts.contains(k) }) {
            out = canAttachImages
                ? "Already at \(Self.maxImages) images"
                : "\(Models.display(modelName)) cannot see images"
        } else if kinds.contains(where: { k in Self.clipKind(
            URL(fileURLWithPath: "x." + k)) != nil }) {
            out = canAttachAudio
                ? "Already at \(Self.maxClips) clips"
                : "\(Models.display(modelName)) cannot hear or watch"
        }
        return out
    }

    // Each doc reference is replaced in place by that file's FENCED content,
    // so it cannot bleed into the chat markup; an image or clip reference is
    // dropped, since those ride their own path.
    private func promptFor(_ raw: String) -> String {
        AttachmentRefs.substitute(raw) { name in
            var out = "@\(name)"
            if let doc = self.attachedDocs.first(where: { $0.name == name }) {
                let md = doc.name.hasSuffix(".md")
                    || doc.name.hasSuffix(".markdown")
                out = "\n\n\(doc.name):\n```\(md ? "markdown" : "")\n"
                    + "\(doc.content)\n```\n\n"
            } else if self.attachedImages.contains(
                            where: { img in img.name == name })
                || self.attachedClips.contains(
                            where: { c in c.name == name }) {
                out = ""
            }
            return out
        }
    }

    // Only the SPEECH becomes an attachment: room tone costs 25 soft tokens a
    // second. Each utterance is one span and the whole session rides one
    // turn.

    @ObservationIgnored private var mic: Microphone?
    @ObservationIgnored private var gate: SpeechGate?
    @ObservationIgnored private let heard = HeardSpeech()
    @ObservationIgnored private var rateInUse: Double = 1
    @ObservationIgnored private var endOfTurn: Task<Void, Never>?
    @ObservationIgnored private var micPhrases: Task<Void, Never>?
    // "I am done talking to you", as against the 700 ms inside SpeechGate
    // that only means "that was a sentence". Long enough to think in: with
    // the gate's own hangover ahead of it, sending costs about 2.2 s of real
    // silence.
    private static let endOfTurnSilence = 1.5

    func voice() {
        if listening { endListening() } else { beginListening() }
    }

    private func beginListening() {
        // BEFORE the input is claimed: on iOS the record category makes
        // output silent, and a player left running into that never finishes.
        speech.stopSpeaking()
        // One session drives one turn, so the mic opens only once generation
        // has ended.
        if let media, ready, !busy {
            let rate = media.audioSampleRate
            rateInUse = rate
            let gate = SpeechGate(rate: rate,
                                  maxSeconds: media.maxAudioSeconds)
            let mic = Microphone(rate: rate)
            heard.clear()
            heardSeconds = 0
            Task { @MainActor in
                if await Microphone.permission() {
                    AudioSession.beginRecording()
                    do {
                        try mic.start { [heard] block in
                            heard.captured(block.count)
                            heard.observe(block)
                            heard.add(gate.push(block))
                        }
                        self.mic = mic
                        self.gate = gate
                        listening = true
                        micPhrases?.cancel()
                        micPhrases = phraseCycler()
                        watchForEndOfTurn()
                        Diag.shared.report(
                            "[mic] listening at \(Int(rate)) Hz "
                            + AudioSession.describe())
                    } catch {
                        AudioSession.endRecording()
                        Diag.shared.report("[mic] FAILED to start: \(error)")
                        flashHUD("\(error)")
                    }
                } else {
                    flashHUD("Microphone access is off")
                }
            }
        }
    }

    // Silence must reach neither the tower nor the transcript: a span with no
    // speech in it comes back as INVENTED speech.
    private func watchForEndOfTurn() {
        endOfTurn?.cancel()
        var ticks = 0
        endOfTurn = Task { @MainActor in
            while listening && !Task.isCancelled {
                // Fast enough that the ring tracks a syllable. The gate's own
                // hop is 20 ms, so nothing here is the limiting factor.
                try? await Task.sleep(for: .milliseconds(60))
                heardSeconds = heard.seconds
                hearingSpeech = gate?.hearing ?? false
                speechLevel = Double(gate?.level ?? 0)
                // A gate that opens and never closes produces no utterance,
                // so the stop line that would explain it is never written
                // either.
                ticks += 1
                if ticks % 16 == 0, let gate {
                    Diag.shared.report(String(
                        format: "[mic] .. %4.1fs open, hearing %@, bar %.5f, "
                            + "loudestFrame %.5f (%.1fx bar), peak %.5f, "
                            + "kept %.1fs, %d utt",
                        Double(heard.samples) / max(rateInUse, 1),
                        gate.hearing ? "YES" : "no ", gate.speechThreshold,
                        gate.loudestFrameEnergy,
                        gate.speechThreshold > 0
                            ? gate.loudestFrameEnergy / gate.speechThreshold
                            : 0,
                        heard.peak, heard.seconds,
                        heard.utteranceCount))
                }
                let quiet = Date().timeIntervalSince(heard.lastAt)
                if listening && heard.hasSpeech
                    && quiet >= ChatModel.endOfTurnSilence {
                    endListening()
                }
            }
        }
    }

    private func endListening() {
        endOfTurn?.cancel()
        endOfTurn = nil
        micPhrases?.cancel()
        micPhrases = nil
        mic?.stop()
        AudioSession.endRecording()
        heard.add(gate?.finish() ?? [])
        let bar = gate?.speechThreshold ?? 0
        mic = nil
        gate = nil
        listening = false
        hearingSpeech = false
        speechLevel = 0
        let said = heard.take()
        let secs = Double(heard.samples) / max(rateInUse, 1)
        // Every way this goes wrong looks the same from outside, so the line
        // carries the sample count, the bar and the PEAK: a dead input and a
        // quiet room both come back as "nothing was said", and only an
        // amplitude tells them apart.
        Diag.shared.report(String(
            format: "[mic] stopped: %.1fs captured, %d utterance(s), "
                + "%.1fs of speech, bar %.5f, peak %.6f, rms %.6f, %@",
            secs, said.count,
            said.reduce(0.0) { sum, u in sum + u.seconds }, bar,
            heard.peak, heard.rms, AudioSession.describe()))
        if said.isEmpty {
            flashHUD(secs < 0.5 ? "No audio from the microphone"
                                : "Nothing was said")
        } else {
            // The one gap with nothing in it: the mic has closed, the towers
            // have not run, and the transcript is still the previous turn.
            flashHUD(Whimsical.current(.heard, hold: 0), prominent: true,
                     seconds: 2)
            sendSpoken(said)
        }
    }

    func stop() {
        speech.stopSpeaking()
        genTask?.cancel()
        // The reliable lever, and nonisolated, so it lands even while a
        // synchronous forward holds the engine. Task cancellation alone does
        // not reach the Metal forward.
        session?.requestStop()
    }

    func factoryReset() {
        // The alert promises ALL settings go, so the whole persistent domain
        // goes rather than a hand-picked list that drifts as settings are
        // added.
        let d = UserDefaults.standard
        if let id = Bundle.main.bundleIdentifier {
            d.removePersistentDomain(forName: id)
        }
        clearCompileCache()
        ChatModel.wipePrecook()
        try? FileManager.default.removeItem(at: Bundle.modelStore())
        // The keeper's claims go with the cache they describe.
        if let support = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first {
            try? FileManager.default.removeItem(
                at: support.appendingPathComponent("anecache"))
        }
        quitApp()
    }

    static let prepFailed = "This model could not be prepared. Try Again "
        + "clears the compile cache and recompiles; if it keeps failing, this "
        + "device's Neural Engine is likely unsupported."

    // The compiled-once FLAGS go with the cache, or the next build reads as
    // warm and shows Loading over a cold recompile.
    private func clearCompileCache() {
        let d = UserDefaults.standard
        for k in d.dictionaryRepresentation().keys
            where k.hasPrefix("compiled.") {
            d.removeObject(forKey: k)
        }
        // iOS quitApp is exit(0), a hard exit that never flushes.
        d.synchronize()
        let fm = FileManager.default
        if let caches = fm.urls(for: .cachesDirectory,
                                in: .userDomainMask).first {
            var paths = [
                caches.appendingPathComponent("com.apple.e5rt.e5bundlecache")]
            if let id = Bundle.main.bundleIdentifier {
                paths.append(caches.appendingPathComponent(id)
                    .appendingPathComponent("com.apple.e5rt.e5bundlecache"))
            }
            for p in paths { try? fm.removeItem(at: p) }
        }
    }

    func send() {
        let attached = !attachedImages.isEmpty || !attachedClips.isEmpty
        if attached, ready, !busy {
            // A SIBLING of sendVision rather than a flag on it: that path
            // speaks MLMultiArray tiles on a square grid at one token rate,
            // and a native-resolution tower has none of those.
            if modelSupportsSoftTokens {
                sendSoft()
            } else {
                sendVision()
            }
        } else {
            sendText()
        }
    }

    // What to ask when a turn carries attachments and no typed question.
    //
    // Ask about PERCEPTION, never about the medium. "Write out what you hear"
    // is answered; "transcribe any speech in the audio" is refused, because
    // naming the audio sends the model hunting for a file it was never given.
    // It has the sound either way; only the question differs.
    private static func softDefaultPrompt(_ parts: [ContentPart]) -> String {
        var images = 0, videos = 0, sounds = 0
        for part in parts {
            switch part {
            case .image: images += 1
            case .video: videos += 1
            case .audio: sounds += 1
            case .text: break
            }
        }
        var asks: [String] = []
        if images > 0 {
            asks.append(images == 1 ? "Describe what you see."
                                    : "Describe what you see in each picture.")
        }
        if videos > 0 { asks.append("Describe what you see happening.") }
        if sounds > 0 { asks.append("Write out what you hear.") }
        return asks.joined(separator: " ")
    }

    private func sendSoft() {
        let raw = input
        let typed = promptFor(raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var scrubbed = raw
        for item in attachedImages {
            scrubbed = AttachmentRefs.scrub(item.name, from: scrubbed)
        }
        for item in attachedClips {
            scrubbed = AttachmentRefs.scrub(item.name, from: scrubbed)
        }
        let display = AttachmentRefs.stripped(scrubbed)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let media {
            let images = attachedImages
            let clips = attachedClips
            let docs = attachedDocs
            let previews = images.compactMap { img in
                VisionPreprocess.thumbnail(img.data, maxPx: 640)
            }
            input = ""
            caret = 0
            attachedImages = []
            attachedClips = []
            attachedDocs = []
            spokenTurn = false
            lastTurnSpoken = false
            let cue: SpokenCue.Kind
            if clips.contains(where: { clip in clip.isVideo }) {
                cue = .watching
            } else if !images.isEmpty {
                cue = .looking
            } else {
                cue = .reading
            }
            softTurn(display: display, typed: typed, previews: previews,
                     films: clips.filter { c in c.isVideo }.map { c in c.url },
                     papers: ChatModel.refs(docs),
                     cue: cue) { [weak self] in
                let out = try await ChatModel.encode(
                    media, images: images, clips: clips) { peek in
                        Task { @MainActor in
                            self?.lookingAt = peek
                            self?.watching = true
                        }
                    }
                await MainActor.run { self?.watching = false }
                return out
            }
        }
    }

    // The gate already decided what is speech, so each utterance goes to the
    // tower as it stands and no clip is re-cut.
    private func sendSpoken(_ said: [SpeechGate.Utterance]) {
        if let media {
            let secs = said.reduce(0.0) { sum, u in sum + u.seconds }
            spokenTurn = true
            lastTurnSpoken = true
            softTurn(display: String(format: "Spoken, %.1fs", secs),
                     typed: ChatModel.spokenPrompt, previews: [],
                     labelled: false) {
                try await ChatModel.encode(media, speech: said)
            }
        }
    }

    // Distinct from an attached clip's "write out what you hear": someone
    // holding the microphone is talking, not filing a transcription job. It
    // names no medium, for the same reason softDefaultPrompt does not.
    private static let spokenPrompt = "Reply to what I just said."

    // One soft-token turn, however its spans were produced: the attachment
    // path encodes files, dictation encodes utterances, and everything after
    // that is the same.
    private func softTurn(
        display: String, typed: String, previews: [CGImage],
        films: [URL] = [], papers: [DocRef] = [], labelled: Bool = true,
        cue: SpokenCue.Kind = .thinking,
        _ encode: @escaping @Sendable () async throws
            -> (parts: [ContentPart], spans: [SoftSpan], perImage: Int)
    ) {
        if let session, ready, !busy {
            prefilling = true
            activePrefillStage = .vision
            var asked = Message(fromUser: true, text: display,
                                images: previews, clips: films)
            asked.docs = papers
            asked.placeholder = spokenTurn
            messages.append(asked)
            messages.append(Message(fromUser: false, text: ""))
            let idx = messages.count - 1
            resetLiveBuffers()
            let onReasoning: @Sendable (String) -> Void = { piece in
                Task { @MainActor in
                    if self.prefilling { self.prefilling = false }
                    self.speech.reasoningArrived(piece)
                    if self.messages.indices.contains(idx) {
                        self.liveReason += piece
                        self.messages[idx].reasoningStream.append(piece)
                        self.flushLive(idx)
                    }
                }
            }
            let onToolRound: @Sendable (ToolRoundEvent) -> Void = { event in
                Task { @MainActor in self.applyToolRound(event, at: idx) }
            }
            speech.beginTurn(cue: cue)
            genTask = Task {
                let phrases = phraseCycler()
                let ticker = statsTicker(session)
                await session.setReasoningCaps(soft: thinkTokenCap,
                                               hard: thinkTokenCap * 2)
                do {
                    // The towers run on the SAME GPU as the prefix prime, and
                    // `busy` cannot see the prime: it lives in its own task,
                    // not in genTask. The encode is the one GPU step that
                    // runs before the session's turn path, which already
                    // waits.
                    await session.awaitPriming()
                    let built = try await encode()
                    if built.perImage > 0 { perImageTokens = built.perImage }
                    let ask = typed.isEmpty
                        ? ChatModel.softDefaultPrompt(built.parts) : typed
                    let stream = session.replySoft(
                        ask, parts: built.parts + [.text(ask)],
                        spans: built.spans, labelled: labelled,
                        onReasoning: onReasoning,
                        onToolRound: onToolRound)
                    for await piece in stream {
                        if prefilling { prefilling = false }
                        speech.answerArrived(piece)
                        if messages.indices.contains(idx) {
                            liveAnswer += piece
                            messages[idx].answerStream.append(piece)
                            flushLive(idx)
                        }
                    }
                    speech.endTurn()
                    flushLive(idx, force: true)
                    finishDocs(idx)
                    if await session.turnRolledBack {
                        if messages.count >= 2 { messages.removeLast(2) }
                    } else {
                        await refreshStats(session)
                        recordTG(await session.lastMetrics.tg)
                        noteLoopStop(await session.lastMetrics, idx)
                        commitCurrent()
                        maybeGenerateTitle()
                    }
                } catch {
                    if messages.indices.contains(idx) {
                        messages[idx].text = ChatModel.attachmentFailed(error)
                    }
                }
                phrases.cancel()
                ticker.cancel()
                genTask = nil
                prefilling = false
                lookingAt = nil
            }
        }
    }

    // The order the user attached them IS the order they reach the template,
    // so a question can refer to them positionally.
    private static func encode(
        _ media: Gemma4Media, images: [ImageAttachment],
        clips: [ClipAttachment],
        onFrame: (@Sendable (VideoPeek) -> Void)? = nil
    ) async throws -> (parts: [ContentPart], spans: [SoftSpan],
                       perImage: Int) {
        try await Task.detached(priority: .userInitiated) {
            var parts: [ContentPart] = []
            var spans: [SoftSpan] = []
            var widest = 0
            for img in images {
                let t0 = Date()
                let span = try media.image(img.data)
                widest = max(widest, span.rows)
                parts.append(.image)
                spans.append(span)
                ChatModel.note("image", img.name, span.rows, t0)
            }
            for clip in clips {
                let t0 = Date()
                for (part, span) in try await clip.spans(media,
                                                         onFrame: onFrame) {
                    parts.append(part)
                    spans.append(span)
                    ChatModel.note(part.noun, clip.name, span.rows, t0)
                }
            }
            // The towers run HERE, so this is the peak an attachment turn
            // reaches, and one no load-time number predicts.
            if !images.isEmpty || !clips.isEmpty {
                Footprint.report("encoded \(images.count) img "
                    + "\(clips.count) clip")
            }
            return (parts, spans, widest)
        }.value
    }

    // The gate already bounded each utterance under the tower's ceiling, so
    // media.audio returns a single span per utterance.
    private static func encode(
        _ media: Gemma4Media, speech: [SpeechGate.Utterance]
    ) async throws -> (parts: [ContentPart], spans: [SoftSpan],
                       perImage: Int) {
        try await Task.detached(priority: .userInitiated) {
            let t0 = Date()
            var parts: [ContentPart] = []
            var spans: [SoftSpan] = []
            for u in speech {
                for span in try media.audio(u.samples) {
                    parts.append(.audio)
                    spans.append(span)
                }
            }
            ChatModel.note("speech", "\(speech.count) utterance(s)",
                           spans.reduce(0) { sum, s in sum + s.rows }, t0)
            return (parts, spans, 0)
        }.value
    }

    // Without this a turn that went wrong is indistinguishable from one that
    // never encoded anything.
    nonisolated private static func note(_ kind: String, _ name: String,
                                        _ rows: Int, _ since: Date) {
        Diag.shared.report(String(
            format: "attach %@ %@ -> %d soft tokens (%.1fs)",
            kind, name, rows, Date().timeIntervalSince(since)))
    }

    private static func attachmentFailed(_ error: any Error) -> String {
        // Gemma4MediaError prints its own sentence, which the user can act
        // on; anything else would leak a Swift case name into the transcript.
        error is Gemma4MediaError
            ? "\(error)" : "Could not process the attachment."
    }

    // Sent exactly as if typed, through the same input and send path.
    static let sampleResearch =
        "Using Simple English Wikipedia as your primary source, " +
        "research Dark Matter and Dark Energy. Provide a highly factual, " +
        "structured summary of the current state of the art in the field. " +
        "Breakdown the explanation into: " +
        "1. Core Definitions, 2. Key Differences, and " +
        "3. Current Scientific Evidence. " +
        "Keep the language simple, direct, and free " +
        "of introductory or concluding remarks."
    static let sampleStory = "Describe everything you see in this "
        + "picture, then write a story based on it."
    static let sampleClip = "Describe what happens in this video."
    static let sampleEuler = "Using Euler's formula e^(ix) = cos(x) + "
        + "i*sin(x) and your calculator, explore: e^i (one radian around "
        + "the unit circle), Euler's identity e^(i*pi) + 1 = 0, i^i, "
        + "sqrt(i), and ln(-1). Compute each and explain briefly what it "
        + "means geometrically."
    // "multiplies by 1.05" rather than "compounded annually": the latter
    // summons the textbook formula, which a small model garbles.
    static let sampleInterest = "A savings account starts with $1,000 and "
        + "earns 5% interest each year, so every year its balance "
        + "multiplies by 1.05. Use the calculator to find the balance "
        + "after 22 years."
    // The shape the smallest model gets right: every number in the prompt,
    // one operation between them, no unit conversion and no
    // thousands-commas. ONE ingredient, because with two its
    // percentages-sum-to-100 prior computes the second as
    // flour-minus-first.
    static let sampleCookies = "Bakers weigh every ingredient as a "
        + "percentage of the flour weight. In my cookie recipe, butter "
        + "is 65% of the flour. I have 350 grams of flour. Use the "
        + "calculator to find how many grams of butter I need."

    var calcSampleIsInterest: Bool { modelName != Models.fallback }

    // App/Resources/ on disk, but the copy phase FLATTENS into the bundle,
    // so these stay a plain forResource lookup with no subdirectory.
    static let samplePicture: Data? = Bundle.main
        .url(forResource: "dogs-beach", withExtension: "jpg")
        .flatMap { url in try? Data(contentsOf: url) }

    // A URL, not Data: the streaming encode exists so a clip is never held
    // whole.
    static let sampleVideo: URL? = Bundle.main
        .url(forResource: "dogs-beach", withExtension: "mp4")
    static let samplePictureThumb: CGImage? = samplePicture.flatMap { data in
        VisionPreprocess.thumbnail(data, maxPx: 128)
    }

    // The answer is in the report's TABLE and the prose points the other way,
    // so a run that merely reads the words gets it wrong, visibly.
    static let sampleReport =
        "Which block gives the most fruit per tree, and what makes that "
        + "surprising?"

    static let samplePdf: URL? = Bundle.main
        .url(forResource: "harvest-report", withExtension: "pdf")

    // A STILL, never a frame pulled from the clip: drawing the card must not
    // depend on decoding the video.
    static let sampleClipThumb: CGImage? = Bundle.main
        .url(forResource: "dogs-beach-poster", withExtension: "jpg")
        .flatMap { url in try? Data(contentsOf: url) }
        .flatMap { data in VisionPreprocess.thumbnail(data, maxPx: 128) }

    // Off, each sample shows until it has been used once.
    var alwaysShowSamples: Bool =
        UserDefaults.standard.bool(forKey: "alwaysShowSamples") {
        didSet {
            UserDefaults.standard.set(alwaysShowSamples,
                                      forKey: "alwaysShowSamples")
        }
    }
    private var usedSamples: Set<String> = Set(
        UserDefaults.standard.stringArray(forKey: "usedSamples") ?? [])

    // MIRRORS the pill view's own gating: the two must agree, or the empty
    // screen shows a heading over nothing.
    private var applicableSampleIds: [String] {
        var ids = ["calc"]
        if accessState != .offline { ids.append("research") }
        if canAttachImages, ChatModel.samplePictureThumb != nil {
            ids.append("picture")
        }
        if canOfferVideoSample { ids.append("video") }
        if canOfferDocumentSample { ids.append("document") }
        return ids
    }

    // No capability gate: a document becomes Markdown before the model sees
    // it, so this rides every model, text-only included.
    var canOfferDocumentSample: Bool { ChatModel.samplePdf != nil }

    // A video turn is ~32 tower passes and thousands of soft tokens WHATEVER
    // the clip's own size: the frame count and per-frame budget are the
    // model's numbers, not the file's.
    var canOfferVideoSample: Bool {
        installedGB >= 4 && canAttachVideo && ChatModel.sampleVideo != nil
    }

    var showSamples: Bool {
        ready && !busy && messages.isEmpty
            && applicableSampleIds.contains { id in showSample(id) }
    }

    func showSample(_ id: String) -> Bool {
        alwaysShowSamples || !usedSamples.contains(id)
    }

    private func markSampleUsed(_ id: String) {
        usedSamples.insert(id)
        UserDefaults.standard.set(Array(usedSamples), forKey: "usedSamples")
    }

    func runResearchSample() {
        markSampleUsed("research")
        input = ChatModel.sampleResearch
        caret = input.utf16.count
        send()
    }

    func runEulerSample() {
        markSampleUsed("calc")
        input = ChatModel.sampleEuler
        caret = input.utf16.count
        send()
    }

    func runInterestSample() {
        markSampleUsed("calc")
        input = ChatModel.sampleInterest
        caret = input.utf16.count
        send()
    }

    func runCookiesSample() {
        markSampleUsed("calc")
        input = ChatModel.sampleCookies
        caret = input.utf16.count
        send()
    }

    func runPictureSample() {
        if let data = ChatModel.samplePicture {
            markSampleUsed("picture")
            input = ""
            caret = 0
            attachImage(data, name: "dogs-beach.jpg", at: 0)
            input += ChatModel.sampleStory
            send()
        }
    }

    // A bundle URL needs none of the copying a dropped file does.
    func runDocumentSample() {
        if let url = ChatModel.samplePdf {
            markSampleUsed("document")
            input = ""
            caret = 0
            flashHUD("Reading the report")
            Task { @MainActor in
                if let text = await ChatModel.markdown(of: url) {
                    attachDoc("harvest-report.pdf", text, at: 0,
                              from: url)
                    input += ChatModel.sampleReport
                    send()
                } else {
                    flashHUD("Cannot read the report")
                }
            }
        }
    }

    func runVideoSample() {
        if let url = ChatModel.sampleVideo {
            markSampleUsed("video")
            input = ""
            caret = 0
            attachClip(url, isVideo: true, at: 0)
            input += ChatModel.sampleClip
            send()
        }
    }

    // An unknown emitted name keeps its raw text with a question-mark glyph,
    // which is the no-such-tool case the strip exists to surface.
    private static let toolGlyphs: [String: (label: String, symbol: String)] = [
        "get_current_time": ("Current Time", "clock"),
        "calculator": ("Calculator", "function"),
        "web_search": ("Web Search", "magnifyingglass"),
        "fetch_url": ("URL Fetch", "link"),
        "get_news": ("News", "newspaper"),
        "get_weather": ("Weather", "cloud.sun"),
        "wikipedia_query": ("Wikipedia", "books.vertical"),
    ]

    // The index is re-checked on the actor hop, and a completion never
    // un-fills a result: main-actor task order across two hops is not
    // guaranteed.
    private func applyToolRound(_ event: ToolRoundEvent, at idx: Int) {
        if messages.indices.contains(idx) {
            let known = event.resolved
                .flatMap { name in ChatModel.toolGlyphs[name] }
            let args = event.params
                .map { param in "\(param.name): \"\(param.value)\"" }
                .joined(separator: "  ")
            // The spoken cue is driven from HERE and not from onTool, because
            // every send path reports rounds this way while only the text one
            // carries onTool.
            if event.result == nil {
                speech.toolStarted(event.resolved ?? event.name)
            } else {
                speech.toolFinished()
            }
            let at = messages[idx].toolRounds.firstIndex { row in
                row.id == event.round
            }
            if let at {
                if event.result != nil {
                    messages[idx].toolRounds[at].result = event.result
                }
            } else {
                messages[idx].toolRounds.append(ToolRound(
                    id: event.round, emitted: event.name,
                    label: known?.label ?? event.name,
                    symbol: known?.symbol ?? "questionmark.circle",
                    args: args, result: event.result))
            }
        }
    }

    // The tower's grid drives the tiling and the merged-token count. The
    // images are carried, so a later text turn still sees them.
    private func sendVision() {
        let raw = input
        let body = promptFor(raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let ask = body.isEmpty ? VLPrompt.defaultPrompt : body
        // The strip SHOWS the images, so their references leave the displayed
        // text where a doc's stays.
        var scrubbed = raw
        for img in attachedImages {
            scrubbed = AttachmentRefs.scrub(img.name, from: scrubbed)
        }
        let display = AttachmentRefs.stripped(scrubbed)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let session, modelSupportsVision, ready, !busy {
            let images = attachedImages
            input = ""
            caret = 0
            attachedImages = []
            attachedDocs = []
            prefilling = true
            activePrefillStage = .vision
            // Bounded, so the conversation does not retain full bitmaps for
            // its whole life.
            let previews = images.compactMap { img in
                VisionPreprocess.thumbnail(img.data, maxPx: 640)
            }
            messages.append(Message(fromUser: true, text: display,
                                    images: previews))
            messages.append(Message(fromUser: false, text: ""))
            let idx = messages.count - 1
            resetLiveBuffers()
            // Several images are distinct photos, not tiles of one picture,
            // so each is fit; only a lone image keeps the toggle.
            let tiled = allowsTiling && visionMode == .tile && images.count == 1
            // The index is re-checked on every MainActor hop: New Chat or a
            // rollback can shrink `messages` between a hop's enqueue and its
            // run.
            let onReasoning: @Sendable (String) -> Void = { piece in
                Task { @MainActor in
                    if self.prefilling { self.prefilling = false }
                    self.speech.reasoningArrived(piece)
                    if self.messages.indices.contains(idx) {
                        self.liveReason += piece
                        self.messages[idx].reasoningStream.append(piece)
                        self.flushLive(idx)
                    }
                }
            }
            let onToolRound: @Sendable (ToolRoundEvent) -> Void = { event in
                Task { @MainActor in self.applyToolRound(event, at: idx) }
            }
            speech.beginTurn(cue: .looking)
            genTask = Task {
                let phrases = phraseCycler()
                let ticker = statsTicker(session)
                // The budget is time-anchored, so the token caps are
                // re-derived from the latest measured decode rate each turn.
                await session.setReasoningCaps(soft: thinkTokenCap,
                                               hard: thinkTokenCap * 2)
                do {
                    let grid = await session.visionGrid() ?? VisionGrid.canonical
                    perImageTokens = grid.mergedTokens  // refine the estimate
                    var allTiles: [MLMultiArray] = []
                    for img in images {
                        allTiles.append(contentsOf: try VisionPreprocess.imageSet(
                            img.data, tiled: tiled, grid: grid))
                    }
                    let stream = session.replyVision(
                        ask, tiles: allTiles, gridH: grid.gridH,
                        gridW: grid.gridW, tokensPerImage: grid.mergedTokens,
                        numberImages: images.count > 1,
                        thumbnail: ChatModel.traceThumbnail(images.first?.data),
                        onReasoning: onReasoning, onToolRound: onToolRound)
                    for await piece in stream {
                        if prefilling { prefilling = false }
                        speech.answerArrived(piece)
                        if messages.indices.contains(idx) {
                            liveAnswer += piece
                            messages[idx].answerStream.append(piece)
                            flushLive(idx)
                        }
                    }
                    speech.endTurn()
                    flushLive(idx, force: true)
                    finishDocs(idx)
                    // A prefill-phase Stop rolls the turn back in the
                    // session, so the two bubbles it left go with it.
                    if await session.turnRolledBack {
                        if messages.count >= 2 { messages.removeLast(2) }
                    } else {
                        await refreshStats(session)
                        recordTG(await session.lastMetrics.tg)
                        noteLoopStop(await session.lastMetrics, idx)
                        commitCurrent()
                        maybeGenerateTitle()
                    }
                } catch {
                    if messages.indices.contains(idx) {
                        messages[idx].text = "Could not process the image."
                    }
                }
                phrases.cancel()
                ticker.cancel()
                genTask = nil
                prefilling = false
            }
        }
    }

    // Decoded straight to size, never the full bitmap.
    private static func traceThumbnail(_ data: Data?) -> Data? {
        var out: Data? = nil
        if let data, let cg = VisionPreprocess.thumbnail(data, maxPx: 128) {
            let buf = NSMutableData()
            if let dst = CGImageDestinationCreateWithData(
                buf, UTType.jpeg.identifier as CFString, 1, nil) {
                CGImageDestinationAddImage(dst, cg, [
                    kCGImageDestinationLossyCompressionQuality: 0.7,
                ] as CFDictionary)
                if CGImageDestinationFinalize(dst) { out = buf as Data }
            }
        }
        return out
    }

    private func sendText() {
        let raw = input
        let prompt = promptFor(raw)
        let display = AttachmentRefs.stripped(raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           ready, !busy, let session {
            activePrefillStage = attachedDocs.isEmpty ? .prefill : .documents
            spokenTurn = false
            lastTurnSpoken = false
            input = ""
            caret = 0
            let papers = ChatModel.refs(attachedDocs)
            attachedDocs = []
            prefilling = true
            var asked = Message(fromUser: true, text: display)
            asked.docs = papers
            messages.append(asked)
            messages.append(Message(fromUser: false, text: ""))
            let idx = messages.count - 1
            resetLiveBuffers()
            // Reasoning streams on ChatSession's actor, so each piece hops to
            // the main one. Think precedes content, so the first reasoning
            // token ending prefill holds.
            let onReasoning: @Sendable (String) -> Void = { piece in
                Task { @MainActor in
                    if self.prefilling { self.prefilling = false }
                    if self.consulting { self.consulting = false }
                    self.speech.reasoningArrived(piece)
                    if self.messages.indices.contains(idx) {
                        self.liveReason += piece
                        self.messages[idx].reasoningStream.append(piece)
                        self.flushLive(idx)
                    }
                }
            }
            // A tool call re-enters prefill while its result is ingested.
            let onTool: @Sendable (String) -> Void = { _ in
                Task { @MainActor in
                    self.consulting = true
                    self.prefilling = true
                }
            }
            let onToolRound: @Sendable (ToolRoundEvent) -> Void = { event in
                Task { @MainActor in self.applyToolRound(event, at: idx) }
            }
            speech.beginTurn()
            genTask = Task {
                let phrases = phraseCycler()
                let ticker = statsTicker(session)
                // The budget is time-anchored, so the token caps are
                // re-derived from the latest measured decode rate each turn.
                await session.setReasoningCaps(soft: thinkTokenCap,
                                               hard: thinkTokenCap * 2)
                let stream = session.reply(prompt, onReasoning: onReasoning,
                                           onTool: onTool,
                                           onToolRound: onToolRound)
                for await piece in stream {
                    if prefilling { prefilling = false }  // first token
                    if consulting { consulting = false }
                    speech.answerArrived(piece)
                    if messages.indices.contains(idx) {
                        liveAnswer += piece
                        messages[idx].answerStream.append(piece)
                        flushLive(idx)
                    }
                }
                speech.endTurn()
                flushLive(idx, force: true)
                finishDocs(idx)
                phrases.cancel()
                ticker.cancel()
                genTask = nil
                prefilling = false
                consulting = false
                // A prefill-phase Stop rolls the turn back in the session, so
                // the two bubbles it left go with it.
                if await session.turnRolledBack {
                    if messages.count >= 2 { messages.removeLast(2) }
                } else {
                    await refreshStats(session)
                    recordTG(await session.lastMetrics.tg)
                    noteLoopStop(await session.lastMetrics, idx)
                    // EVERY completed turn: title generation commits only
                    // once, so without this later turns are lost until the
                    // conversation is left.
                    commitCurrent()
                    maybeGenerateTitle()
                }
            }
        }
    }

    // A breaker-cut turn WITH content keeps its partial answer; only one that
    // committed nothing gets the note.
    private func noteLoopStop(_ m: TurnMetrics, _ idx: Int) {
        if m.endReason == "loop-breaker", messages.indices.contains(idx),
           messages[idx].text.isEmpty {
            messages[idx].loopStopped = true
        }
    }

    // Never per token: the memory walk is charged only at this cadence.
    private func refreshStats(_ session: ChatSession) async {
        let t = await session.lastMetrics
        if t.pp > 0 { lastPP = t.pp }   // feeds the attachment time estimate
        // ctx 0 is a conversation that has not run a turn yet, where every
        // number would read 0. The gate lives HERE rather than in the view,
        // because an empty label IS "there is nothing to say yet".
        if t.ctx > 0 {
            let tokens = thinkingActive
                ? "🤔 \(t.thinkTokens) 💬 \(t.contentTokens)"
                : "💬 \(t.thinkTokens + t.contentTokens)"
            statsLabel = "⇄ \(t.ctx.formatted(.number))  \(tokens) "
                + String(format: "🐏 %.1fGB t/s: %.1f/%.1f",
                         Self.footprintGiB(), t.pp, t.tg)
        }
    }

    // The one cadence behind both the live rate and the Markdown snapshots,
    // neither of which may run per token.
    private func statsTicker(_ session: ChatSession) -> Task<Void, Never> {
        Task { @MainActor in
            while busy && !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(400))
                if busy {
                    // Runs through the prefill too: pp lands the moment
                    // decode starts, and the line is on screen to receive it.
                    if !prefilling { refreshDocs() }
                    await refreshStats(session)
                }
            }
        }
    }

    private func refreshDocs() {
        if let idx = messages.indices.last, !messages[idx].fromUser {
            Instrument.timed("refreshDocs") {
                messages[idx].answerDoc = messages[idx].answerStream.snapshot()
                messages[idx].reasoningDoc =
                    messages[idx].reasoningStream.snapshot()
            }
        }
    }

    // NON-OBSERVABLE, and flushed on a bounded cadence: writing the
    // @Observable model per token re-diffs the whole non-lazy transcript at
    // the token rate and stalls the main thread.
    @ObservationIgnored private var liveAnswer = ""
    @ObservationIgnored private var liveReason = ""
    @ObservationIgnored private var lastFlushNs: UInt64 = 0
    private static let flushIntervalNs: UInt64 = 100_000_000   // 100ms

    private func resetLiveBuffers() {
        liveAnswer = ""
        liveReason = ""
        lastFlushNs = 0
    }

    // Equality-guarded, so an idle gap writes nothing.
    private func flushLive(_ idx: Int, force: Bool = false) {
        let now = DispatchTime.now().uptimeNanoseconds
        let due = force || now - lastFlushNs >= Self.flushIntervalNs
        if due, messages.indices.contains(idx) {
            lastFlushNs = now
            if messages[idx].text != liveAnswer {
                messages[idx].text = liveAnswer
            }
            if messages[idx].reasoning != liveReason {
                messages[idx].reasoning = liveReason
            }
        }
    }

    // Seals the streams, so the settled docs are the full parse: finish
    // resolves what a snapshot's open block could not.
    private func finishDocs(_ idx: Int) {
        if messages.indices.contains(idx), !messages[idx].fromUser {
            messages[idx].answerDoc = messages[idx].answerStream.finish()
            messages[idx].reasoningDoc =
                messages[idx].reasoningStream.finish()
        }
    }

    // Set at send: the stage a reply is ingesting under, before tokens flow
    // and the reasoning verbs take over.
    private var activePrefillStage: Whimsical.Stage = .prefill
    // Cleared when the turn ends, so a typed follow-up is back to the
    // ordinary words.
    private var spokenTurn = false
    private var whimsicalStage: Whimsical.Stage {
        let out: Whimsical.Stage
        if listening {
            out = .listening
        } else if consulting {
            out = .consulting
        } else if prefilling {
            out = spokenTurn ? .listening : activePrefillStage
        } else {
            out = spokenTurn ? .mulling : .reasoning
        }
        return out
    }

    private func phraseCycler() -> Task<Void, Never> {
        Task { @MainActor in
            while (busy || listening) && !Task.isCancelled {
                let p = Whimsical.pair(whimsicalStage)
                thinkStatus = p.first
                thinkLabel = p.second
                // Faster while listening, where the phrase IS the feedback.
                try? await Task.sleep(for: .seconds(listening ? 2 : 5))
            }
        }
    }

    // phys_footprint, which EXCLUDES the mmapped weights.
    private static func footprintGiB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<natural_t>.stride)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        let gib = 1_073_741_824.0
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / gib : 0
    }

    // The card values are the fallback for a set that predates the matrix.

    private static func presets(_ setDir: URL) -> SamplingPresets {
        let url = setDir.appendingPathComponent("generation_config.json")
        return (try? SamplingPresets.from(generationConfig: url,
                                          fallback: .qwen35)) ?? .qwen35
    }

    var downloadETA: String? { Self.eta(phaseStart, downloadFraction) }

    // A monotonic countdown from the held finish time, so the remaining never
    // jumps back up as the per-program estimates jitter.
    var optimizeETA: String? {
        optimizeFinish.map { finish in
            Self.formatETA(max(0, finish.timeIntervalSinceNow))
        }
    }

    private func tightenOptimizeETA() {
        if compileFraction > 0.02 {
            let now = Date()
            let elapsed = now.timeIntervalSince(phaseStart)
            let candidate = elapsed * (1 - compileFraction) / compileFraction
            let current = optimizeFinish?.timeIntervalSince(now) ?? candidate
            // EASED toward each fresh estimate rather than locked to the most
            // optimistic one, so a warm early phase cannot strand the ETA
            // under a cold later phase.
            let blended = current + (candidate - current) * 0.35
            optimizeFinish = now.addingTimeInterval(max(0, blended))
        }
    }

    private static func eta(_ start: Date, _ fraction: Double) -> String? {
        var result: String? = nil
        if fraction > 0.02 {
            let elapsed = Date().timeIntervalSince(start)
            result = formatETA(elapsed * (1 - fraction) / fraction)
        }
        return result
    }

    static func formatETA(_ seconds: Double) -> String {
        let s = max(1, Int(seconds.rounded()))
        let m = Int((Double(s) / 60).rounded())
        let body = s < 60 ? "~\(s) seconds"
            : "~\(m) minute" + (m == 1 ? "" : "s")
        return "ETA: " + body
    }

    // The model's own measured times when it has them, else the base model's
    // scaled by byte ratio. nil until that base has been through both phases.

    func estimatedMinutes(_ name: String) -> (download: Int, optimize: Int)? {
        var result: (download: Int, optimize: Int)? = nil
        let base = Models.fallback
        let dl0 = Self.savedSeconds("download", base)
        let opt0 = Self.savedSeconds("optimize", base)
        if dl0 > 0, opt0 > 0, let bytes = ModelCatalog.source(name)?.bytes,
           let base0 = ModelCatalog.source(base)?.bytes, base0 > 0 {
            let ratio = Double(bytes) / Double(base0)
            let dl = Self.savedSeconds("download", name)
            let opt = Self.savedSeconds("optimize", name)
            // Download scales with bytes; compile time tracks op count and is
            // superlinear in model size, ~n^1.3.
            // A GGUF set runs on the GPU with no ANE compile at all, so it
            // has no optimize phase to estimate and must not be extrapolated
            // one from the base model.
            let gpuOnly = ModelCatalog.source(name)?.files != nil
            result = (Self.minutes(dl > 0 ? dl : dl0 * ratio),
                      gpuOnly ? 0
                              : Self.minutes(opt > 0 ? opt
                                                     : opt0 * pow(ratio, 1.3)))
        }
        return result
    }

    private static func minutes(_ seconds: Double) -> Int {
        max(1, Int((seconds / 60).rounded()))
    }

    private static func timeKey(_ phase: String, _ name: String) -> String {
        "seconds.\(phase).\(name)"
    }

    private static func saveSeconds(_ phase: String, _ name: String,
                                    _ seconds: Double) {
        UserDefaults.standard.set(seconds, forKey: timeKey(phase, name))
    }

    private static func savedSeconds(_ phase: String, _ name: String) -> Double {
        UserDefaults.standard.double(forKey: timeKey(phase, name))
    }

}

// A lock rather than an actor: utterances arrive on the audio thread, which
// cannot await, and are read on the main one when the mic closes.
final class HeardSpeech: @unchecked Sendable {
    private let lock = NSLock()
    private var said: [SpeechGate.Utterance] = []
    private var count = 0
    private var at = Date()

    // When the last utterance CLOSED, not when the last block arrived: blocks
    // keep coming through silence and would never let the turn end.
    var lastAt: Date {
        lock.lock()
        defer { lock.unlock() }
        return at
    }

    var hasSpeech: Bool {
        lock.lock()
        defer { lock.unlock() }
        return !said.isEmpty
    }

    var utteranceCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return said.count
    }

    // NOT the time the mic has been open: the pauses are already gone.
    var seconds: Double {
        lock.lock()
        defer { lock.unlock() }
        return said.reduce(0.0) { sum, u in sum + u.seconds }
    }

    // What the microphone actually delivered, which is what tells a silent
    // room from a silent input.
    var samples: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func captured(_ n: Int) {
        lock.lock()
        count += n
        lock.unlock()
    }

    // So a stopped mic can be told from a quiet one after the fact.
    private var loudest: Float = 0
    private var square: Double = 0

    func observe(_ block: [Float]) {
        var top: Float = 0
        var sum: Double = 0
        for s in block {
            let a = abs(s)
            if a > top { top = a }
            sum += Double(s) * Double(s)
        }
        lock.lock()
        if top > loudest { loudest = top }
        square += sum
        lock.unlock()
    }

    var peak: Float {
        lock.lock()
        defer { lock.unlock() }
        return loudest
    }

    var rms: Double {
        lock.lock()
        defer { lock.unlock() }
        return count > 0 ? (square / Double(count)).squareRoot() : 0
    }

    func add(_ more: [SpeechGate.Utterance]) {
        if !more.isEmpty {
            lock.lock()
            said.append(contentsOf: more)
            at = Date()
            lock.unlock()
        }
    }

    func clear() {
        lock.lock()
        said = []
        count = 0
        loudest = 0
        square = 0
        at = Date()
        lock.unlock()
    }

    func take() -> [SpeechGate.Utterance] {
        lock.lock()
        defer { lock.unlock() }
        let out = said
        said = []
        return out
    }
}

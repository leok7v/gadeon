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
    }

    struct ImageAttachment: Identifiable {
        let id = UUID()
        let name: String
        let data: Data
        // A small EXIF-upright preview decoded ONCE at attach (chip thumbnail),
        // so the chip does not re-decode per render. iOS photo-library picks all
        // arrive named "image", so the preview is the only way to tell several
        // attached photos apart.
        let thumbnail: CGImage?
    }

    // One tool round of an assistant turn, a row in the transcript's tool
    // strip. `result` is the exact string the model received back as the
    // tool response; nil while the tool is still running (the row shimmers).
    struct ToolRound: Identifiable {
        let id: Int                  // round number within the turn
        let emitted: String          // the name the model asked for
        let label: String            // display name (resolved tool, or emitted)
        let symbol: String           // SF Symbol for the row glyph
        let args: String             // key: "value" summary of the call params
        var result: String?
    }

    struct Message: Identifiable {
        let id = UUID()
        let fromUser: Bool
        var text: String
        // Display previews of the turn's attached images (EXIF-upright,
        // decoded once at send, bounded size), so the transcript SHOWS
        // what was sent instead of a bare "@name" reference.
        var images: [CGImage] = []
        // The turn's tool rounds, appended live as the session runs them, so
        // the transcript shows which tools ran with what arguments -- the
        // "did it even call the tool" debugging surface.
        var toolRounds: [ToolRound] = []
        // The assistant turn's <think> stream, shown in a collapsed grey
        // disclosure above the answer. Empty for user turns and when
        // reasoning-effort is off.
        var reasoning = ""
        // The runaway loop breaker ended this turn with NO committed answer
        // (it fired mid-think); the transcript explains itself instead of
        // showing silence.
        var loopStopped = false
        // Markdown snapshots the transcript renders when the Markdown toggle
        // is on, maintained beside the raw strings so plain-text rendering,
        // copy, and rollback stay untouched. The streams are reference types
        // whose sealed blocks never re-parse; only the open block does.
        var answerDoc = Markdown.Document.empty
        var reasoningDoc = Markdown.Document.empty
        let answerStream = MarkdownStream()
        let reasoningStream = MarkdownStream()
    }

    // How an attached image is fed to the model (Settings). Host-side policy,
    // independent of the tower's baked grid.
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

    // Status bar detail under the composer (Settings + macOS View menu).
    enum StatusBarMode: String, CaseIterable, Identifiable {
        case off, short, extended
        var id: String { rawValue }
        var label: String {
            switch self {
            case .off: return "None"
            case .short: return "Short"
            case .extended: return "Extended"
            }
        }
    }

    static let defaultSystemPrompt = "You are a helpful assistant."

    var messages: [Message] = []
    // A reopened past conversation is READ-ONLY: the transcript is shown, the
    // composer is hidden, and no session/KV is touched (continuation is a later
    // opt-in). currentConversationId names the live-or-open conversation for the
    // history store; nil until the first turn is committed. Logic lives in
    // Conversations.swift.
    var readOnly = false
    var currentConversationId: UUID?
    // The model-generated title for the live conversation once made; nil until
    // then, when conversationTitle falls back to the first user message.
    var generatedTitle: String?
    // Appearance (the sidebar theme control), persisted; applied at the root.
    var theme: AppTheme = {
        AppTheme(rawValue:
            UserDefaults.standard.string(forKey: "theme") ?? "") ?? .system
    }() {
        didSet { UserDefaults.standard.set(theme.rawValue, forKey: "theme") }
    }
    var input = ""
    // Insertion point (UTF-16 offset into input), synced from the editor so a
    // dropped or picked file's reference lands at the caret.
    var caret = 0
    // Pending images for the NEXT turn (VL), up to maxImages. Non-empty -> the
    // composer shows a chip per image and Send runs the vision path, not the
    // text session. Each is referenced inline by its unique "@name", like docs.
    var attachedImages: [ImageAttachment] = []
    // Attached .txt/.md docs for the next turn: shown as chips, referenced
    // inline as "@name", and substituted into the prompt in place at send.
    var attachedDocs: [Doc] = []
    var status = "loading model..."
    // Servable: a built session AND the whole compile finished (build makes
    // the session before the warmup/carry compiles complete, so the session
    // alone is not enough).
    var ready: Bool { session != nil && !compiling }
    // The in-flight generation IS the busy state; no parallel flag to keep
    // in sync with it.
    var busy: Bool { genTask != nil }
    var statsLabel = ""             // full pp/tg/ctx/mem (status bar, Option held)
    var statsCompact = ""           // reasoning + mem (status bar, default)
    // Prefill is running (before the first decoded token): the ingest stage,
    // which the transcript quip and status bar draw a phrase from.
    var prefilling = false
    // A tool (wikipedia_query) is running / its result is being ingested: the
    // "consulting" stage (library whimsy). Set by the session's onTool, cleared
    // when the answer begins to stream.
    var consulting = false
    // Three distinct phrases, refreshed together while a reply runs, one per
    // surface that can show a phrase at once: thinkStatus is the transcript's
    // working line, thinkLabel the reasoning disclosure header, barStatus the
    // status bar's prefill line.
    var thinkStatus = "Thinking"
    var thinkLabel = "Thinking"
    var barStatus = "Thinking"
    // First-run onboarding, two persisted gates around an explicit
    // download-consent panel: the legal EULA, then -- once Download is tapped
    // on the consent panel -- the AI-mistakes disclaimer, which the download
    // overlaps. The App Store requires that consent before any fetch begins.
    var eulaAccepted = UserDefaults.standard.bool(forKey: "eulaAccepted")
    var accepted = UserDefaults.standard.bool(forKey: "disclaimerAccepted")
    // Selected model (persisted), sanitized to one this platform ships (a 2B
    // choice saved on the Mac must not stick when the defaults land on a
    // phone).
    private static func startModel() -> String {
        let saved = UserDefaults.standard.string(forKey: "modelName")
            ?? Models.start
        return Models.all.contains(saved) ? saved : Models.start
    }
    var modelName: String = ChatModel.startModel()
    // Download-on-demand state (the bundle ships no weights). downloadName !=
    // nil surfaces the consent panel; downloading drives the progress view;
    // done/total come from HubFetch. Cleared once the set is verified on disk.
    var downloadName: String? = nil
    var downloading = false
    // Bytes downloaded / total bytes (HubFetch reports live byte progress), so
    // the bar + ETA track real throughput, not a file count.
    var downloadDone: Int64 = 0
    var downloadTotal: Int64 = 0

    var downloadFraction: Double {
        downloadTotal > 0 ? Double(downloadDone) / Double(downloadTotal) : 0
    }

    // "7.3 of 7.8 GB": prose "of", one unit -- the bar already shows the
    // percent, so absolute bytes are the information the numbers add.
    var downloadCounter: String {
        String(format: "%.1f of %.1f GB",
               Double(downloadDone) / 1_000_000_000,
               Double(downloadTotal) / 1_000_000_000)
    }

    // A trimmed-empty prompt is not sendable, which is why this is not just
    // !input.isEmpty: a field holding only spaces must leave Send disabled
    // and must let a stray Return do nothing.

    var canSend: Bool {
        let text = input.trimmingCharacters(
            in: .whitespacesAndNewlines)
        let has = !text.isEmpty || !attachedImages.isEmpty
            || !attachedDocs.isEmpty
        return ready && !busy && has
    }

    var typing: Bool { !input.isEmpty }

    // A heads-up before a heavy turn: rough prefill tokens for the pending
    // attachments (docs ~4 chars/token; each image the tower's per-image run),
    // with a duration once a turn has measured the prefill rate. nil until the
    // estimate is large enough to be a noticeable wait. The number is a FLOOR --
    // it counts trunk prefill, not the per-tile tower encode.
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

    // Whether the ACTIVE model can see images, read from the backend after
    // each build (a CoreML set with a tower; the GGUF/Metal 27B says no
    // until its vision splice lands). Gates the attach button, image drops,
    // and the vision send path, so an unsupported model never dead-ends.
    private(set) var modelSupportsVision = false

    // Whether the ACTIVE model reasons at all, asked of its own chat template
    // at session build (templateSupportsThinking). A model whose generation
    // prompt bakes a closed empty <think> never thinks, so the lightbulb, the
    // Settings switch and the thinking budget are all inert and say so.
    private(set) var modelSupportsThinking = true

    // The user's Thinking preference AND the model's ability to act on it.
    // Everything that actually drives a turn reads THIS, so a non-reasoning
    // model cannot be put into a state the engine will ignore.
    var thinkingActive: Bool { thinking && modelSupportsThinking }

    // The 0.8B regresses when tiled (it drowns in vision tokens -- worse output
    // AND ~2GB RSS), so it always fits one tile; iOS runs only the 0.8B. The
    // larger macOS-only models get the tile/fit toggle.
    var allowsTiling: Bool {
        !isOS && modelName != Models.fallback && modelSupportsVision
    }

    // Sized from the catalog by the pending download's name -- the two were
    // only ever set together, so the name alone carries the state.
    var downloadSizeText: String {
        let bytes = downloadName.flatMap { name in
            ModelCatalog.source(name)?.bytes
        } ?? 0
        return ByteCountFormatter.string(fromByteCount: bytes,
                                         countStyle: .file)
    }

    // Compile ("optimize for this device") progress: compileDoneLoC
    // accumulates each program's model.mil LoC as it finishes its ANE compile
    // (Engine's onCompiledLoC), against compileTotalLoC computed up front.
    // `compiling` drives the blocking Optimizing view; it clears when the
    // whole set is ready.

    var compiling = false
    var compileDoneLoC = 0
    var compileTotalLoC = 0
    var compileFraction: Double {
        compileTotalLoC > 0
            ? min(1.0, Double(compileDoneLoC) / Double(compileTotalLoC)) : 0
    }

    // Non-nil when the set could not be built on this device: an older Neural
    // Engine can reject the trunk's compile ("Unable to build plan"). Surfaces
    // a clear message + Retry instead of an endless spinner.
    var loadError: String? = nil

    // True only when this set has never finished compiling on this install
    // (the slow one-time ANE compile). A later launch, with the OS's compile
    // cache warm, shows a brief "Loading" instead of the "Optimizing, may take
    // a while" copy. Defaults true so the first frame errs toward the
    // reassuring message.
    var firstCompile = true

    // Two-tier network access, split by WHAT LEAVES THE DEVICE. `wikipedia`
    // gates the Wikimedia-only tools (article lookup matched on-device +
    // today's headlines; nothing typed is ever sent). `webAccess` gates the
    // open-web tools (search / fetch / weather; the model's search terms and
    // IP-derived location go out). Both default ON (useful out of the box);
    // the toolbar button cycles airplane -> book -> globe, Settings holds
    // the two switches for fine control.
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

    // Transient toggle confirmation (Thinking / access tier), shown centered
    // for ~3s then dissolved: iOS has no hover help, so a toolbar state flip
    // needs visible confirmation; macOS gets the same flash.
    private(set) var hud: String?
    @ObservationIgnored private var hudTask: Task<Void, Never>?

    private func flashHUD(_ text: String) {
        hudTask?.cancel()
        hud = text
        hudTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            if !Task.isCancelled { hud = nil }
        }
    }

    // Swap the live session's runner so a tier change takes effect next
    // turn, like Thinking.
    func setAccess(wikipedia wiki: Bool, web: Bool) {
        wikipedia = wiki
        webAccess = web
        let s = session
        let r = toolRunner
        Task { await s?.setTools(r) }
    }
    // Reasoning-effort. Default on: the model reasons in <think> and the app
    // shows a collapsed grey disclosure of it above each answer.
    var thinking = true
    // Editable system prompt (Settings), persisted. Applied only when the
    // session is (re)built -- model load and New Chat -- never mid-turn, so it
    // cannot mismatch a live KV prefix.
    var systemPrompt: String = UserDefaults.standard
        .string(forKey: "systemPrompt") ?? ChatModel.defaultSystemPrompt {
        didSet {
            UserDefaults.standard.set(systemPrompt, forKey: "systemPrompt")
            // Every model's precooked prefix bakes the prompt; an edit makes
            // them all stale, so free the disk now (the stamps would reject
            // them anyway).
            ChatModel.wipePrecook()
        }
    }
    // Image handling (Settings), persisted PER MODEL: tile vs fit. Read by
    // sendVision for the larger models; the 0.8B always fits (see
    // allowsTiling). The default differs by model: a tiled image on the 9B /
    // 27B costs minutes of tower encode + prefill, so they default to Fit;
    // the 2B/4B afford tiles (measured better OCR).
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
    // Markdown transcript rendering (Settings > View), persisted. The docs
    // are maintained regardless of the toggle, so flipping it re-renders past
    // turns too.
    var renderMarkdown: Bool = UserDefaults.standard
        .object(forKey: "renderMarkdown") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(renderMarkdown,
                                      forKey: "renderMarkdown")
        }
    }
    // Ask before deleting a conversation from the sidebar trash (Settings,
    // persisted). Default on; the iOS swipe-to-delete is its own confirmation,
    // so this gates only the macOS trash button.
    var confirmDeleteConversation: Bool = UserDefaults.standard
        .object(forKey: "confirmDeleteConversation") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(confirmDeleteConversation,
                                      forKey: "confirmDeleteConversation")
        }
    }
    // Status bar detail (Settings / macOS View menu), persisted. Default None:
    // the whimsical "working" line now lives in the transcript, so the bar is
    // hidden until a user opts into the throughput stats.
    var statusBarMode: StatusBarMode = {
        let raw = UserDefaults.standard.string(forKey: "statusBarMode") ?? ""
        return StatusBarMode(rawValue: raw) ?? .off
    }() {
        didSet {
            UserDefaults.standard.set(statusBarMode.rawValue,
                                      forKey: "statusBarMode")
        }
    }
    // Settings is shown in-view (like the disclaimer), gated by this flag.
    var showSettings = false
    // Session debug view (the Option-hold ladybug), routed in-view the same
    // way. The trace behind it is ALWAYS recording -- drill-in is
    // retroactive, after something already went wrong.
    var showDebug = false
    var traceEvents: [TraceEvent] = []
    private let traceFile = TraceFile()
    // Wires MD render timing into Diag + starts the main-thread hang watchdog,
    // once, at model construction.
    private let instrument = Instrument.install()
    var tracePath: String { traceFile.path }
    // Ring cap so a marathon session cannot grow the array unbounded; the
    // on-disk transcript keeps everything.
    private static let traceCap = 2000
    // Overthink penalty: a static logit bias on reasoning branch-openers while
    // thinking, to curb over-reasoning. Robust across 0.5-4.0. TODO: Settings.
    // Paper: https://arxiv.org/abs/2606.00206
    private static let overthinkLambda: Float = 1.0

    // Thinking budget: T-shirt sizes anchored in TIME. The token cap is
    // derived per model AND device from the measured decode rate (persisted
    // after each turn), so "M is about 20 seconds" holds on an M3 Max and an
    // iPad alike, and stays true as the engine gets faster. Seeded with a
    // conservative rate until the first turn measures truth.
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

    // Measured decode rate for the current model on THIS device; a
    // conservative 20 t/s until the first turn records one.
    var measuredTG: Double {
        let v = UserDefaults.standard.double(forKey: Self.tgKey(modelName))
        return v > 0 ? v : 20
    }

    // EMA so one outlier turn (a tool-heavy round, a cold start) does not
    // yank the budget around.
    private func recordTG(_ tg: Double) {
        if tg > 0 {
            let old = UserDefaults.standard.double(
                forKey: Self.tgKey(modelName))
            let ema = old > 0 ? 0.7 * old + 0.3 * tg : tg
            UserDefaults.standard.set(ema, forKey: Self.tgKey(modelName))
        }
    }

    // The soft token cap for the current budget, rounded to 100 for display
    // honesty; the hard runaway backstop is twice this.
    var thinkTokenCap: Int {
        let raw = thinkBudget.seconds * measuredTG
        return max(200, Int((raw / 100).rounded()) * 100)
    }

    private let log = Logger(subsystem: "io.github.leok7v.gadeon",
                              category: "ui")
    private var chat: AneChat?
    // The unified-engine (ternary GGUF) backend, set instead of `chat` when a
    // GGUF model is active: a GPU MetalBackend driving the SAME ChatSession. Its
    // template / vocab / sampler recs all come from the one GGUF. Vision and the
    // ANE heavy/carry compile sets do not apply (guarded by `chat` being nil).
    private var ggufBackend: (any AgentBackend)?
    private var ggufTemplate = ""
    private var ggufVocabCount = 0
    // The multi-turn chat loop (jinja chat_template + O(delta) bookmark
    // continuation). One per conversation; recreated on model switch, reset on
    // New Chat.
    private var session: ChatSession?
    // The in-flight generation, held so Stop can cancel it. Cancelling ends
    // the reply stream (the decode loop breaks on Task.isCancelled) and the
    // send task clears busy. Generation is otherwise unbounded (runs to EOS).
    private var genTask: Task<Void, Never>?
    // Bytes of the mmapped weights (a CoreML set's weight.bin sum, or the
    // whole GGUF): file-backed evictable pages, not in phys_footprint, so
    // refreshStats reports them separately.
    private var weightsBytes: Int64 = 0
    // Compiles each downloaded .mlmodelc in-process WHILE the rest of the set
    // still streams (macOS and iOS), so the one-time "Optimizing" compile
    // overlaps the download.
    private let primer = Primer()
    // Learned prefill rate (tokens/sec) from the last turn, for the attachment
    // time estimate; 0 until the first turn measures it (then the estimate
    // shows a duration). Per-image vision token cost, a baseline until the first
    // image turn learns the tower's real merged-token count.
    private var lastPP = 0.0
    private var perImageTokens = 256
    // The set's sampling matrix (thinking on/off x text/vision), captured at
    // build so makeSession can rebuild the session (New Chat) without re-reading
    // the model directory.
    private var activePresets: SamplingPresets = .qwen35
    // Wall-clock start of the in-flight download or optimize, for the live
    // ETA.
    @ObservationIgnored private var phaseStart = Date()
    // The projected optimize FINISH time, EMA-smoothed across the per-program
    // estimates so it tracks BOTH directions -- a fast early phase (warm decode)
    // must not pin it under a slow later one (cold prefill) -- while damping the
    // per-program jitter. nil until the first estimate; reset when a build
    // begins.
    @ObservationIgnored private var optimizeFinish: Date?

    // EULA accepted: persist, then resolve the model. load() builds an on-disk
    // set or surfaces the download-consent panel for an absent one. The fetch
    // does NOT start here -- the App Store requires explicit consent, so it
    // begins only when the user taps Download on that panel, which then overlaps
    // the disclaimer shown next.
    func acceptEULA() {
        eulaAccepted = true
        UserDefaults.standard.set(true, forKey: "eulaAccepted")
        load(name: modelName)
    }

    func accept() {
        accepted = true
        UserDefaults.standard.set(true, forKey: "disclaimerAccepted")
    }

    // Resolve the pinned set for `name` on disk. Present and verified -> build
    // it straight on the ANE. Absent -> surface the download-consent panel
    // (the app ships no weights); confirmDownload then fetches it from the Hub
    // and builds.

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

    // Load a ternary GGUF on the GPU (MetalChat) off the main actor, then swap
    // in a ChatSession over its MetalBackend. No ANE compile, no vision, no
    // heavy/carry sets -- everything comes from the one file. Only the Sendable
    // backend + template + vocab + sampler recs cross back to the main actor
    // (the non-Sendable engine stays inside the backend).
    private func buildGguf(name: String, path: String) {
        compiling = true
        compileDoneLoC = 0
        compileTotalLoC = 0
        phaseStart = Date()
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        weightsBytes = (attrs?[.size] as? NSNumber)?.int64Value ?? 0
        loadError = nil
        // No ANE compile: the wait is a GGUF mmap + GPU upload, so the
        // screen says Loading (indeterminate), never Optimizing.
        firstCompile = false
        Task.detached { [weak self] in
            var built: (backend: any AgentBackend, template: String,
                        vocab: Int, presets: SamplingPresets)?
            do {
                let c = try MetalChat(ggufPath: path)
                built = (c.backend(), c.chatTemplate,
                         c.tokenizer.vocabCount, c.samplingPresets)
            } catch {
                built = nil
            }
            let vision = await built?.backend.supportsVision() ?? false
            await MainActor.run {
                if let b = built {
                    self?.chat = nil
                    self?.ggufBackend = b.backend
                    self?.ggufTemplate = b.template
                    self?.ggufVocabCount = b.vocab
                    self?.activePresets = b.presets
                    self?.modelSupportsVision = vision
                    self?.makeSession()
                    self?.compiling = false
                    self?.status = ""
                    self?.primeSession()
                } else {
                    self?.compiling = false
                    self?.loadError = Self.prepFailed
                }
            }
        }
    }

    // Retry after a failed build: wipe the whole compile cache first, so a
    // corrupt program (e.g. an OS reboot mid-compile) is thrown away and the
    // model recompiles cleanly from scratch, in-process -- no relaunch. A
    // truly unsupported ANE just fails again, but the button costs nothing.

    func retry() {
        loadError = nil
        clearCompileCache()
        load(name: modelName)
    }

    // Fetch the pinned set from the Hub (verified, atomic) with a progress
    // report, then build it. Stored permanently in Application Support and
    // excluded from backup, so it downloads once and later launches are
    // offline.

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
            // In-place download puts files at their final (cache-keyed)
            // paths, so the primer can compile each finished program while
            // later files stream.
            let setDir = dest.appendingPathComponent(src.revision)
            Task.detached { [weak self] in
                // Entry snapshot before the primer's first compile, so the
                // streamed set's programs land in its ownership claim.
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
                    // Repeated verification failures are corruption (a proxy,
                    // a disk problem), not connectivity -- say so.
                    failure = "download failed verification, try again"
                } catch {
                }
                await MainActor.run {
                    // Whatever is primed is primed; from here the build
                    // compiles the remainder itself, so the helper must not
                    // contend for the serial ANE compiler.
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

    // Build the engine for an on-disk set and wait for the WHOLE set (decode +
    // batched embedding + carry prefill) to finish compiling on the ANE before
    // enabling chat. First cold-cache launch shows "compiling on Neural
    // Engine", blocking Send until everything is ready; a warm cache is
    // instant.

    private func build(setDir: URL, attempt: Int = 0) {
        compiling = true
        compileDoneLoC = 0
        compileTotalLoC = 0
        phaseStart = Date()
        optimizeFinish = nil
        weightsBytes = Self.weightBytes(setDir)
        activePresets = Self.presets(setDir)
        loadError = nil
        let key = Self.compiledKey(setDir)
        // A container migration (every Xcode install / App Store update)
        // forces a cold recompile regardless of the compiled-once flag: the
        // e5 cache keys bind to container identity. Show the honest
        // Optimizing copy for it, not "Loading".
        firstCompile = !UserDefaults.standard.bool(forKey: key)
            || AneCache.shared.containerMigrated()
        Task.detached { [weak self] in
            // Entry snapshot BEFORE the first MLModel touch, so everything
            // this build compiles lands in the set's ownership claim.
            let cacheBefore = AneCache.shared.buildBegan()
            // Fresh install: the e5 daemon has not created its cache dir yet, so
            // a restore lands in a dir it does not trust. Compile the smallest
            // program first so the daemon creates+owns the dir, THEN relink the
            // shadow's inodes into it, so the real load below warm-hits them.
            Engine.primeCacheDir(setDir)
            AneCache.shared.restoreFromShadow()
            let total = Engine.compileTotalLoC(setDir)
            await MainActor.run { self?.compileTotalLoC = total }
            // Each program's model.mil LoC lands here as it finishes
            // compiling, so the Optimizing bar advances across the whole set,
            // not just at coarse phase edges.
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
                if built != nil {
                    self?.modelSupportsVision = vision
                    self?.makeSession()
                } else {
                    self?.buildFailed(setDir, attempt)
                }
            }
            if let built {
                do {
                    // Compile the WHOLE model before enabling chat: decode, then
                    // the heavy embed + one-block carry sets. Any failure is
                    // fatal -- no half-compiled partial -- so a corrupt program
                    // hits the failure view and Try Again clears the cache and
                    // recompiles. A fresh install can afford the splash.
                    try await built.engine.warmup()
                    try await built.loadHeavy()
                    try await built.loadCarry()
                    // Autodetect the MTP self-spec drafter: loads it only if the
                    // set ships it, silent no-op otherwise (plain decode).
                    await built.loadMTP()
                    // The tower compiled during essential load; drop its
                    // residency so text-only serving does not hold it (an image
                    // turn reloads it warm).
                    await built.engine.offloadVision()
                    AneCache.shared.buildEnded(setDir: setDir,
                                               before: cacheBefore)
                    await MainActor.run {
                        // Only the first (cold) compile times the real
                        // optimize; a warm relaunch is instant and must not
                        // overwrite it.
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

    // A build that failed to load: in hardlink cache mode a relinked bundle may
    // be corrupt or stale, so purge this set's cache + shadow and recompile
    // ONCE (attempt caps the retry, so a genuine unsupported-device failure
    // still surfaces). Otherwise the failure view + Try Again.
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

    // Persistent "already compiled once" key. The {sha} leaf identifies a
    // downloaded set; the mf0 graph .mil size is folded in so a re-emitted
    // bundled graph (Local debug) reads as a fresh compile, not a warm load;
    // the OS version is folded in because the e5 compile cache is keyed by OS
    // build -- after an OS update every model recompiles from scratch, and the
    // "Optimizing, this takes a while" screen must say so again.

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

    // Switch the active model (macOS picker). A downloaded set switches
    // straight away; an un-downloaded one surfaces the consent panel WITHOUT
    // tearing down the current model, so Cancel returns to it. No-op mid-reply
    // / same model.

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

    // Cancel a switch to an un-downloaded model: dismiss the consent panel. When
    // a model is loaded, it was never torn down, so the app returns to it. At
    // STARTUP the persisted choice may be un-downloaded with nothing loaded --
    // so if a downloaded set exists (e.g. the 0.8B), switch to it and load it,
    // rather than stranding the user on a mandatory download with no way back.

    func cancelDownload() {
        let fallback = ready ? nil : downloadedFallback()
        downloadName = nil
        if let fallback {
            commitSwitch(fallback)
            status = "loading model…"
            load(name: fallback)
        }
    }

    // Whether the download screen's Cancel has somewhere to go: back to a loaded
    // model, or to a downloaded set on disk. False only on the mandatory
    // first-run download (nothing on disk yet), which stays uncancellable.

    var canCancelDownload: Bool { ready || downloadedFallback() != nil }

    // A downloaded set to fall back to, preferring this device's start model,
    // else any set on disk that it is actually offered. nil when nothing is
    // downloaded.

    private func downloadedFallback() -> String? {
        isOnDisk(Models.start)
            ? Models.start : Models.all.first { isOnDisk($0) }
    }

    // Commit to `name`: tear down the current engine + conversation and make
    // it active. Taken only when a switch is actually chosen (a downloaded set
    // or a confirmed download), never while the consent panel is still
    // cancellable.

    private func commitSwitch(_ name: String) {
        genTask?.cancel()
        // A cook still running on the outgoing model's engine would hold that
        // whole mapping alive, and keep the GPU busy, for as long as it takes
        // to finish work whose result is about to be discarded.
        session?.requestStop()
        primer.cancel()
        chat = nil
        ggufBackend = nil
        session = nil
        // Pending images belong to the OLD model's vision path; the new one
        // may not have eyes at all. The capability re-probes after build.
        modelSupportsVision = false
        for img in attachedImages {
            input = AttachmentRefs.scrub(img.name, from: input)
        }
        attachedImages = []
        clampCaret()
        downloading = false
        compiling = false
        compileDoneLoC = 0
        loadError = nil
        messages = []
        generatedTitle = nil
        statsLabel = ""
        statsCompact = ""
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

    // Settings > Models: per-model disk state + selective delete / download.
    // diskRevision ticks on a delete so the pane re-reads the disk (isOnDisk
    // reads no observable state).
    private(set) var diskRevision = 0

    func isDownloaded(_ name: String) -> Bool { isOnDisk(name) }

    // Delete a downloaded set to free its disk space; it can be downloaded
    // again. The ACTIVE model is not deletable (the running engine mmaps it).
    // The compiled-once flags go with it (every size/OS variant for the sha),
    // so a re-download presents as a fresh install -- the Optimizing screen,
    // not a bare Loading.
    func deleteModel(_ name: String) {
        if name != modelName, !busy, !downloading {
            ChatModel.erase(name)
            diskRevision += 1
        }
    }

    // Everything one model owns on disk: the set, its precooked prefix, and
    // the compiled-once flags (every size / OS variant for its sha), so a
    // re-download presents as a fresh install -- the Optimizing screen, not a
    // bare Loading.
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

    // Downloaded sets this device is not offered: a choice persisted from a
    // roomier device, or one a gating change dropped (the 0.8B on a 3 GB
    // phone, whose Neural Engine cannot compile it). They are unreachable --
    // absent from the picker, and Settings > Models hides itself when only
    // one model is offered -- so without this their gigabytes have no route
    // off the device short of a reinstall. An offered model is never touched,
    // downloaded or not, and anything erased returns on demand.
    private static func pruneUnavailable() {
        let offered = Set(Models.all)
        // Empty means the device runs nothing at all, not that every set is
        // dead: pruning on that would wipe the store rather than one entry.
        if !offered.isEmpty {
            let names = (try? FileManager.default.contentsOfDirectory(
                atPath: Bundle.modelStore().path)) ?? []
            for name in names where !offered.contains(name) {
                erase(name)
            }
        }
    }

    // Fetch a model from Settings: the same consent -> download -> build flow
    // the composer picker uses (downloading implies switching to it).
    func requestDownload(_ name: String) {
        if !busy, !downloading, !isOnDisk(name),
           ModelCatalog.source(name) != nil {
            downloadName = name
        }
    }

    // Build a fresh ChatSession over the loaded model with the CURRENT system
    // prompt and reasoning-effort. Cheap (no model recompile), so a Settings
    // change lands on the next conversation. The engine is cleared separately
    // (newChat resets it through the new session's backend).

    // The MiniLM semantic-search model bundled inside the LLM package, if
    // present. Best-effort: absent -> no wikipedia_query tool (the app still
    // works, just no grounding).
    private static let minilmPath: String? = WikiSlugs.bundledModel?.path

    // The tool runner for the current tiers. get_current_time and calculator
    // are local, so they are offered even in airplane mode.
    private var toolRunner: (any ToolRunner)? {
        SafeToolRunner(slugsPath: wikipedia ? ChatModel.minilmPath : nil,
                       wikipedia: wikipedia, network: webAccess)
    }

    // The system prompt with the current date/time appended, so the model
    // answers "what day is it" directly without a tool. When Web access is off,
    // it also states that no web/Wikipedia lookup exists, so the model does not
    // fabricate a search tool for an online-sounding question -- get_current_time
    // stays usable. Rebuilt per session (New Chat), like the tool advertisement.
    // Regions wholly (or predominantly) south of the equator, so the stated
    // season flips there -- resolved from the locale, per the zero-config rule.
    private static let southernRegions: Set<String> = [
        "AU", "NZ", "AR", "CL", "UY", "PY", "BO", "PE", "BR", "ZA", "NA",
        "BW", "ZW", "ZM", "MZ", "MG", "LS", "SZ", "AO", "MW", "PG", "FJ",
        "NC", "WS", "TO", "VU", "SB",
    ]

    // The system prompt SPLITS for the precooked-prefix cache: systemStable
    // is everything whose bytes hold across sessions (persona, locale,
    // access tier, vision) -- what ChatSession precooks; systemDynamic is
    // the date/time tail that changes every minute and would poison the
    // cache stamp, laid separately after a restore.
    private var systemStable: String {
        // The host's locale, so the model answers in the user's units (the 2B
        // otherwise garbles C<->F) and knows the region for weather and
        // local questions.
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
        // A text-tuned checkpoint with a grafted tower (the 27B) reflexively
        // claims it cannot see images even while describing one; assert the
        // vision channel whenever the active model actually has it.
        if modelSupportsVision {
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
        // State the season outright: a small model does not reliably infer it
        // from the date (the 2B guessed "spring" in July).
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
        traceFile.append(e)
    }

    // Wire the fresh session's trace into the debug array + the on-disk
    // transcript, headed by the session's configuration. The vision
    // capability is set BEFORE makeSession (the system prompt reads it).
    private func hookTrace() {
        traceFile.note("=== \(modelName) thinking=\(thinkingActive) "
            + "wiki=\(wikipedia) web=\(webAccess) \(Date())")
        let s = session
        Task {
            await s?.setTrace { [weak self] e in
                Task { @MainActor in self?.recordTrace(e) }
            }
        }
        // Endpoint diagnostics (what the search backend ACTUALLY returned)
        // land beside the tool rounds in the debug view + transcript.log.
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

    // ---- precooked system-prefix cache ---------------------------------
    // One state file per model under Caches/precook: the rendered system +
    // tools prefix, prefilled once and restored at every session build
    // (model load / New Chat), so TTFT skips the prefix prefill. The stamp
    // inside the file self-invalidates when the prompt / tools tier /
    // locale / template change; the files are also wiped outright when the
    // user edits the system prompt (a 27B blob is ~300MB of stale bytes
    // otherwise).
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

    // Prime the fresh session from the model's precooked prefix, cooking it
    // on a miss. primeOrCook registers as the session's priming gate, so an
    // early user send WAITS it out (the same wait the fresh prefill would
    // have cost) instead of interleaving with it.
    private func primeSession(resetFirst: Bool = false) {
        if let s = session {
            let url = ChatModel.precookURL(modelName)
            try? FileManager.default.createDirectory(
                at: ChatModel.precookDir, withIntermediateDirectories: true)
            Task { await s.primeOrCook(at: url, resetFirst: resetFirst) }
        }
    }

    // New Chat rebuilds the session so an edited system prompt / thinking flag
    // takes effect, then resets the engine (dropping the prior KV + GDN state).

    func newChat() {
        commitCurrent()
        readOnly = false
        currentConversationId = nil
        generatedTitle = nil
        messages = []
        // Per-conversation trace: the outgoing one is saved by commitCurrent
        // above; the graph and debug view now belong to the fresh chat.
        traceEvents = []
        statsLabel = ""
        // The outgoing session may still be cooking its prefix, and the fresh
        // one resets the engine they SHARE -- so it has to be off the engine
        // first. Held in genTask so `busy` covers the swap: Send stays
        // disabled until the new session is the live one.
        let outgoing = session
        genTask = Task { @MainActor in
            await outgoing?.endPriming()
            makeSession()
            primeSession(resetFirst: true)
            genTask = nil
        }
    }

    // After the first substantive exchange, generate a short title on the
    // loaded model and re-commit with it. Routed through genTask so the
    // composer is disabled for its ~1s: makeTitle holds the engine, and a
    // concurrent send would interleave two turns on one KV. Once per
    // conversation.
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

    // The toolbar gear and ladybug TOGGLE their views: a second click on
    // the button whose view is up acts as Done, and opening one dismisses
    // the other (the routing shows debug first, so a stale flag would
    // shadow the requested view).

    func toggleSettings() {
        showSettings.toggle()
        if showSettings { showDebug = false }
    }

    func toggleDebug() {
        showDebug.toggle()
        if showDebug { showSettings = false }
    }

    // Flip reasoning-effort on the live session; it takes effect next turn.

    func toggleThinking() {
        if modelSupportsThinking {
            thinking.toggle()
            let s = session
            let on = thinking
            Task { await s?.setThinking(on) }
            flashHUD(thinking ? "Thinking: On" : "Thinking: Off")
        }
    }

    // Ask the running turn to end reasoning at the next token; the session
    // ignores it once the answer has begun, so this only forwards.

    func quickAnswer() {
        let s = session
        Task { await s?.requestQuickAnswer() }
    }

    // The composer's [+] / Photos / drop hand image bytes here; the next send
    // runs the vision path. `at` is the UTF-16 offset the inline reference is
    // inserted at (the caret). Adds one image (up to maxImages), each a
    // uniquely-named inline reference, like a doc.
    func attachImage(_ data: Data, name: String, at offset: Int) {
        // Skip an exact-content duplicate (same file dropped twice) -- it just
        // wastes context/RSS. A text-only model attaches nothing (the button
        // is disabled too; this covers drops and pickers).
        let dup = attachedImages.contains { $0.data == data }
        if modelSupportsVision, attachedImages.count < Self.maxImages, !dup {
            let unique = uniqueName(name)
            // 96px covers a ~24pt chip at up to 3x; decoded straight to size, so
            // even a huge source never materializes its full bitmap.
            let thumb = VisionPreprocess.thumbnail(data, maxPx: 96)
            attachedImages.append(
                ImageAttachment(name: unique, data: data, thumbnail: thumb))
            insertRef(unique, at: offset)
        }
    }

    // A dropped .txt/.md: keep the content, add a uniquely-named inline
    // reference (so two same-named files stay distinct) at `at`.
    func attachDoc(_ name: String, _ content: String, at offset: Int) {
        let capped = Self.capDoc(content)
        // Skip an exact-content duplicate (same file dropped twice).
        if !attachedDocs.contains(where: { $0.content == capped }) {
            let unique = uniqueName(name)
            attachedDocs.append(Doc(name: unique, content: capped))
            insertRef(unique, at: offset)
        }
    }

    // Bound a doc to maxDocBytes, cut on a character boundary and marked, so one
    // dropped file cannot balloon the prompt (the aggregate warning is separate).
    private static func capDoc(_ content: String) -> String {
        content.utf8.count > maxDocBytes
            ? String(content.prefix(maxDocBytes)) + "\n[... truncated at 8 KB]"
            : content
    }

    func clearImage(_ id: UUID) {
        if let img = attachedImages.first(where: { $0.id == id }) {
            input = AttachmentRefs.scrub(img.name, from: input)
        }
        attachedImages.removeAll { $0.id == id }
        clampCaret()
    }

    func clearDoc(_ id: UUID) {
        if let doc = attachedDocs.first(where: { $0.id == id }) {
            input = AttachmentRefs.scrub(doc.name, from: input)
        }
        attachedDocs.removeAll { $0.id == id }
        clampCaret()
    }

    // A reference edited out of the prompt drops its attachment (the chips are
    // driven by these, so the chip goes too), and any broken "@name" remnant is
    // scrubbed so no orphan text is left. Called on every input change; the
    // scrub is idempotent, so a self-triggered re-run settles.
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

    // A reference name unique across BOTH docs and images, so two attachments
    // (even a doc and image sharing a filename) stay distinct inline refs.
    private func uniqueName(_ name: String) -> String {
        var candidate = name
        var n = 2
        while attachedDocs.contains(where: { $0.name == candidate })
            || attachedImages.contains(where: { $0.name == candidate }) {
            candidate = "\(name) (\(n))"
            n += 1
        }
        return candidate
    }

    static let imageExts: Set<String> =
        ["png", "jpg", "jpeg", "heic", "heif", "gif", "webp", "bmp", "tiff"]
    static let docExts: Set<String> = ["txt", "md", "markdown", "text"]
    static let maxDocs = 6
    static let maxImages = 4
    static let maxDocBytes = 8192
    // Warn once the pending attachments cross this rough prefill-token estimate
    // (a few thousand tokens is a noticeable ingest).
    static let warnTokens = 4000

    // A drop: each image becomes a vision attachment (up to maxImages), each
    // .txt/.md a doc attachment (up to maxDocs), each inserted as an inline
    // reference at `offset` (the caret) and advancing it. Reads happen off the
    // URLs here (macOS: the user-selected-files entitlement covers drops).
    func handleDrop(_ urls: [URL], at offset: Int) {
        caret = max(0, min(offset, input.utf16.count))
        for url in urls where ready {
            let ext = url.pathExtension.lowercased()
            let scoped = url.startAccessingSecurityScopedResource()
            if Self.imageExts.contains(ext),
               attachedImages.count < Self.maxImages,
               let data = try? Data(contentsOf: url) {
                attachImage(data, name: url.lastPathComponent, at: caret)
            } else if Self.docExts.contains(ext),
                      attachedDocs.count < Self.maxDocs,
                      let text = try? String(contentsOf: url, encoding: .utf8) {
                attachDoc(url.lastPathComponent, text, at: caret)
            }
            if scoped { url.stopAccessingSecurityScopedResource() }
        }
    }

    // The prompt to send: each inline doc reference is replaced in place by that
    // file's fenced content (markdown-fenced so it does not bleed into chat
    // markup); the image reference is dropped (the image goes via the vision
    // path); stray sentinels are stripped. Both paths multi-block, so a big doc
    // is fine, bounded by the model context + KV RAM.
    private func promptFor(_ raw: String) -> String {
        AttachmentRefs.substitute(raw) { name in
            var out = "@\(name)"
            if let doc = self.attachedDocs.first(where: { $0.name == name }) {
                let md = doc.name.hasSuffix(".md")
                    || doc.name.hasSuffix(".markdown")
                out = "\n\n\(doc.name):\n```\(md ? "markdown" : "")\n"
                    + "\(doc.content)\n```\n\n"
            } else if self.attachedImages.contains(where: { $0.name == name }) {
                out = ""
            }
            return out
        }
    }

    func voice() {
        log.info("TBD: voice")
    }

    // Stop the in-flight generation (the Send button shows Stop while busy).
    // The engine stop flag is the reliable lever: the decode loop polls it and
    // ends even when AsyncStream cancellation does not propagate to the
    // producer. Cancelling the task is belt-and-suspenders. The partial answer
    // and state stay intact, so the conversation continues normally on the
    // next turn.

    func stop() {
        genTask?.cancel()
        // The engine stop flag (raised through the session's active backend) is
        // the reliable lever: nonisolated, so it lands even while a synchronous
        // forward holds the ChatSession/engine -- prefill rolls the turn back, a
        // decode-phase Stop keeps the partial answer. Covers BOTH the CoreML
        // (Engine actor) and Metal (MetalEngine, no actor hop) backends, where
        // task cancellation alone does not reliably reach the Metal forward.
        session?.requestStop()
    }

    // Destructive reset (Option+Command on macOS / 3s long-press on iOS):
    // forget the license acceptance and the model choice, delete every
    // downloaded set and the compile cache, then quit -- so the next launch
    // re-shows the disclaimer and re-downloads the 0.8B fallback from scratch.

    func factoryReset() {
        // The alert promises ALL settings go: wipe the whole persistent
        // domain (system prompt, view toggles, timing keys, ...), not a
        // hand-picked list that drifts as settings are added.
        let d = UserDefaults.standard
        if let id = Bundle.main.bundleIdentifier {
            d.removePersistentDomain(forName: id)
        }
        clearCompileCache()
        ChatModel.wipePrecook()
        try? FileManager.default.removeItem(at: Bundle.modelStore())
        // The cache keeper's claims + snapshot go with the cache they
        // describe.
        if let support = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first {
            try? FileManager.default.removeItem(
                at: support.appendingPathComponent("anecache"))
        }
        quitApp()
    }

    // Shown when the on-device build fails. Try Again wipes the cache and
    // recompiles; a persistent failure means the ANE is too old for the graph.
    static let prepFailed = "This model could not be prepared. Try Again "
        + "clears the compile cache and recompiles; if it keeps failing, this "
        + "device's Neural Engine is likely unsupported."

    // Delete the ANE compile cache AND the "already compiled" flags, so the
    // next build is a clean cold recompile (shown as "Optimizing"). Recovers
    // from a corrupt cache -- e.g. an OS reboot mid-compile -- and backs both
    // the factory reset and Try Again. Safe to call with no model loaded.
    private func clearCompileCache() {
        let d = UserDefaults.standard
        for k in d.dictionaryRepresentation().keys
            where k.hasPrefix("compiled.") {
            d.removeObject(forKey: k)
        }
        // Flush now: iOS quitApp is exit(0), a hard exit that never flushes.
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
        if !attachedImages.isEmpty, ready, !busy {
            sendVision()
        } else {
            sendText()
        }
    }

    // ---- empty-screen sample pills -----------------------------------
    // Two canned demos on a fresh conversation, sent exactly as if typed
    // (the same input + send path), so the transcript teaches the real
    // flow: the agentic research walk and the vision story.
//  static let sampleResearch = "Using Simple English Wikipedia, research "
//      + "Dark Matter and Dark Energy and summarize the current state of "
//      + "the art in this field."
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
    static let sampleEuler = "Using Euler's formula e^(ix) = cos(x) + "
        + "i*sin(x) and your calculator, explore: e^i (one radian around "
        + "the unit circle), Euler's identity e^(i*pi) + 1 = 0, i^i, "
        + "sqrt(i), and ln(-1). Compute each and explain briefly what it "
        + "means geometrically."
    // "compounded annually" summons the textbook (1 + r/nt)^(nt) formula,
    // which the 0.8B garbles into (1 + 5/22)^22 (or subtracts the
    // principal); "multiplies by 1.05" all but dictates 1000 * 1.05^22,
    // the expression every size gets right.
    static let sampleInterest = "A savings account starts with $1,000 and "
        + "earns 5% interest each year, so every year its balance "
        + "multiplies by 1.05. Use the calculator to find the balance "
        + "after 22 years."
    // The 0.8B fumbles even the hinted interest walk (overthinks itself
    // out of the right answer), so the phone model gets a question whose
    // numbers are ALL in the prompt with one operation between them --
    // every unit conversion tried (weeks/year, tonne/grams, minute/year)
    // derailed it in testing, as did thousands-commas in the given
    // numbers. Baker's percentages fit that shape: percent-of-flour per
    // ingredient, one multiply each.
    // ONE ingredient on purpose: with two, the 0.8B's percentages-sum-
    // to-100 prior computes the second as flour-minus-first (baker's
    // percentages exceeding 100% total is exactly the counterintuitive
    // bit). A bare "65% of 350 grams" is the shape it always gets right.
    static let sampleCookies = "Bakers weigh every ingredient as a "
        + "percentage of the flour weight. In my cookie recipe, butter "
        + "is 65% of the flour. I have 350 grams of flour. Use the "
        + "calculator to find how many grams of butter I need."

    // Whether the calculator pill runs the interest walk (2B and up) or
    // the chicken math (the 0.8B base model).
    var calcSampleIsInterest: Bool { modelName != Models.fallback }

    // The bundled demo picture (the beach scene); nil hides the pill.
    static let samplePicture: Data? = Bundle.main
        .url(forResource: "SamplePicture", withExtension: "jpeg")
        .flatMap { url in try? Data(contentsOf: url) }
    // Pill-sized preview decoded once, like the attach chips.
    static let samplePictureThumb: CGImage? = samplePicture.flatMap { data in
        VisionPreprocess.thumbnail(data, maxPx: 128)
    }

    // "Always show sample prompts" (Settings), OFF by default. Off, each
    // sample shows until it has been used once, then hides; on, every
    // applicable sample always shows. One switch -- no per-conversation
    // graduation, no sticky re-enable state.
    var alwaysShowSamples: Bool =
        UserDefaults.standard.bool(forKey: "alwaysShowSamples") {
        didSet {
            UserDefaults.standard.set(alwaysShowSamples,
                                      forKey: "alwaysShowSamples")
        }
    }
    private var usedSamples: Set<String> = Set(
        UserDefaults.standard.stringArray(forKey: "usedSamples") ?? [])

    // The samples applicable to the current model + access tier -- mirrors the
    // pill view's own gating: calc always, research when online, the picture
    // demo on a vision model.
    private var applicableSampleIds: [String] {
        var ids = ["calc"]
        if accessState != .offline { ids.append("research") }
        if modelSupportsVision, ChatModel.samplePictureThumb != nil {
            ids.append("picture")
        }
        return ids
    }

    // The pill area shows on an empty chat while at least one applicable
    // sample is visible (always-on, or not yet used).
    var showSamples: Bool {
        ready && !busy && messages.isEmpty
            && applicableSampleIds.contains { id in showSample(id) }
    }

    // One sample's visibility: always-on shows it regardless, else until used.
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
            attachImage(data, name: "beach.jpeg", at: 0)
            input += ChatModel.sampleStory
            send()
        }
    }

    // Display metadata per advertised tool; an unknown emitted name keeps its
    // raw text with a question-mark glyph (the grounded no-such-tool case,
    // exactly what the strip exists to surface).
    private static let toolGlyphs: [String: (label: String, symbol: String)] = [
        "get_current_time": ("Current Time", "clock"),
        "calculator": ("Calculator", "function"),
        "web_search": ("Web Search", "magnifyingglass"),
        "fetch_url": ("URL Fetch", "link"),
        "get_news": ("News", "newspaper"),
        "get_weather": ("Weather", "cloud.sun"),
        "wikipedia_query": ("Wikipedia", "books.vertical"),
    ]

    // Fold a session tool-round event into the live message: the start event
    // appends a running row, the completion fills its result. The index is
    // re-checked on the actor hop, and a completion never un-fills a result
    // (main-actor task order across two hops is not guaranteed).
    private func applyToolRound(_ event: ToolRoundEvent, at idx: Int) {
        if messages.indices.contains(idx) {
            let known = event.resolved
                .flatMap { name in ChatModel.toolGlyphs[name] }
            let args = event.params
                .map { param in "\(param.name): \"\(param.value)\"" }
                .joined(separator: "  ")
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

    // VL turn: preprocess every attached image, then stream the answer through
    // the ChatSession vision path so the images are carried -- a later text turn
    // still sees them. The tower's grid drives the tiling and the merged-token
    // count; reasoning follows the session's thinking toggle like a text turn.
    // Several images are each fit to one tile (distinct photos, not tiles of one
    // picture); a lone image keeps the tile/fit toggle.
    private func sendVision() {
        let raw = input
        // Docs inline in place, image references dropped (they go via vision).
        // No text left -> a neutral question the transcript never shows; the
        // bubble shows the "@name" references, or an image marker if bare.
        let body = promptFor(raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let ask = body.isEmpty ? VLPrompt.defaultPrompt : body
        // The strip below SHOWS the images, so their "@name" references
        // leave the displayed text (docs keep theirs); an image-only turn
        // shows the strip alone, no text bubble.
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
            statsLabel = ""
            // Bounded display decode (640px long edge, ~1.6MB each, at most
            // four): crisp at transcript sizes without retaining full
            // bitmaps for the life of the conversation.
            let previews = images.compactMap { img in
                VisionPreprocess.thumbnail(img.data, maxPx: 640)
            }
            messages.append(Message(fromUser: true, text: display,
                                    images: previews))
            messages.append(Message(fromUser: false, text: ""))
            let idx = messages.count - 1
            resetLiveBuffers()
            // Several images -> fit each (distinct photos); one image keeps the
            // toggle. iOS is always fit (allowsTiling false there anyway).
            let tiled = allowsTiling && visionMode == .tile && images.count == 1
            // The index is re-checked on every MainActor hop, like
            // applyToolRound: New Chat / rollback can shrink `messages`
            // between the hop's enqueue and its run.
            let onReasoning: @Sendable (String) -> Void = { piece in
                Task { @MainActor in
                    if self.prefilling { self.prefilling = false }
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
            genTask = Task {
                let phrases = phraseCycler()
                let ticker = statsTicker(session)
                // The thinking budget is time-anchored: re-derive the token
                // caps from the latest measured decode rate each turn.
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
                        if messages.indices.contains(idx) {
                            liveAnswer += piece
                            messages[idx].answerStream.append(piece)
                            flushLive(idx)
                        }
                    }
                    flushLive(idx, force: true)
                    finishDocs(idx)
                    // A prefill-phase Stop rolls the turn back in the session;
                    // drop the two bubbles it left (user + empty assistant).
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

    // A tiny aspect-preserving JPEG (<=128px, a few KB) of the turn's first
    // image, riding the .user trace event so the debug view shows WHAT was
    // attached. Decoded straight to size, never the full bitmap.
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
            input = ""
            caret = 0
            attachedDocs = []
            // Prefill runs before the first decoded token: the status bar
            // shows a cycling phrase (prefilling drives the shimmer), no stats
            // yet.
            prefilling = true
            statsLabel = ""
            messages.append(Message(fromUser: true, text: display))
            messages.append(Message(fromUser: false, text: ""))
            let idx = messages.count - 1
            resetLiveBuffers()
            // Reasoning streams on ChatSession's actor, so hop each piece to
            // the main actor to grow the message's think disclosure; the first
            // reasoning token also ends prefill. It comes before the answer
            // (think precedes content), so ordering holds.
            let onReasoning: @Sendable (String) -> Void = { piece in
                Task { @MainActor in
                    if self.prefilling { self.prefilling = false }
                    if self.consulting { self.consulting = false }
                    if self.messages.indices.contains(idx) {
                        self.liveReason += piece
                        self.messages[idx].reasoningStream.append(piece)
                        self.flushLive(idx)
                    }
                }
            }
            // A tool call starts the "consulting" stage (and re-enters prefill
            // while the fetched article is ingested); the first answer/reason
            // token clears it.
            let onTool: @Sendable (String) -> Void = { _ in
                Task { @MainActor in
                    self.consulting = true
                    self.prefilling = true
                }
            }
            let onToolRound: @Sendable (ToolRoundEvent) -> Void = { event in
                Task { @MainActor in self.applyToolRound(event, at: idx) }
            }
            genTask = Task {
                let phrases = phraseCycler()
                let ticker = statsTicker(session)
                // The thinking budget is time-anchored: re-derive the token
                // caps from the latest measured decode rate each turn.
                await session.setReasoningCaps(soft: thinkTokenCap,
                                               hard: thinkTokenCap * 2)
                let stream = session.reply(prompt, onReasoning: onReasoning,
                                           onTool: onTool,
                                           onToolRound: onToolRound)
                for await piece in stream {
                    if prefilling { prefilling = false }  // first token
                    if consulting { consulting = false }
                    if messages.indices.contains(idx) {
                        liveAnswer += piece
                        messages[idx].answerStream.append(piece)
                        flushLive(idx)
                    }
                }
                flushLive(idx, force: true)
                finishDocs(idx)
                phrases.cancel()
                ticker.cancel()
                genTask = nil
                prefilling = false
                consulting = false
                // A prefill-phase Stop rolls the turn back in the session; drop
                // the two bubbles it left (user + empty assistant).
                if await session.turnRolledBack {
                    if messages.count >= 2 { messages.removeLast(2) }
                } else {
                    await refreshStats(session)
                    recordTG(await session.lastMetrics.tg)
                    noteLoopStop(await session.lastMetrics, idx)
                    // Persist EVERY completed turn: title generation commits
                    // only once, so without this later turns are lost until the
                    // conversation is left (New Chat / switch).
                    commitCurrent()
                    maybeGenerateTitle()
                }
            }
        }
    }

    // The loop breaker fired mid-think and the turn committed NO answer:
    // flag the message so the transcript explains itself instead of showing
    // silence. A breaker-cut turn WITH content keeps its partial answer.
    private func noteLoopStop(_ m: TurnMetrics, _ idx: Int) {
        if m.endReason == "loop-breaker", messages.indices.contains(idx),
           messages[idx].text.isEmpty {
            messages[idx].loopStopped = true
        }
    }

    // Pull throughput + context metrics into the status bar: pp/tg t/s, total
    // ctx (💾), and the reasoning (🤔) / content (💬) split. Called on a ~400ms
    // ticker while streaming and once at the end, never per token. The memory
    // walk (task_vm_info) is charged only at this cadence.
    private func refreshStats(_ session: ChatSession) async {
        let t = await session.lastMetrics
        if t.pp > 0 { lastPP = t.pp }   // feeds the attachment time estimate
        let gib = 1_073_741_824.0
        // The mmapped weights are not in phys_footprint, so report their size
        // explicitly or the number reads as if the model were not loaded.
        let mem = weightsBytes > 0
            ? String(format: "   mem: %.1fGB + %.1fGB weights",
                     Self.footprintGiB(), Double(weightsBytes) / gib)
            : String(format: "   mem: %.1fGB", Self.footprintGiB())
        let speeds = "pp \(Self.rate(t.pp)) t/s   tg \(Self.rate(t.tg)) t/s"
        let tokens = "🤔 \(t.thinkTokens)   💬 \(t.contentTokens)"
        statsCompact = "\(tokens)\(mem)"
        statsLabel = "\(speeds)   ctx: \(t.ctx) 💾   \(tokens)\(mem)"
    }

    // t/s for the status bar: one decimal below 10 so a slow decode does not
    // read as a flat 0 t/s.
    private static func rate(_ v: Double) -> String {
        v < 10 ? String(format: "%.1f", v) : String(format: "%.0f", v)
    }

    // Poll the running throughput a few times a second (not per token) so the
    // status bar shows a live pp/tg rate while a reply streams. It idles until
    // the first token (status cleared), so prefill still shows "Thinking...".
    // The same cadence refreshes the live message's Markdown snapshots --
    // never per token; only the stream's open block re-parses.
    private func statsTicker(_ session: ChatSession) -> Task<Void, Never> {
        Task { @MainActor in
            while busy && !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(400))
                if busy && !prefilling {
                    refreshDocs()
                    await refreshStats(session)
                }
            }
        }
    }

    // Snapshot the live assistant message's Markdown streams into the docs
    // the transcript renders.
    private func refreshDocs() {
        if let idx = messages.indices.last, !messages[idx].fromUser {
            Instrument.timed("refreshDocs") {
                messages[idx].answerDoc = messages[idx].answerStream.snapshot()
                messages[idx].reasoningDoc =
                    messages[idx].reasoningStream.snapshot()
            }
        }
    }

    // Streamed pieces accumulate in these non-observable buffers and flush to
    // the visible message on a bounded cadence: a fast token stream would
    // otherwise mutate the @Observable model once per token, re-diffing the
    // whole (non-lazy) transcript ~25x/sec and stalling the main thread
    // (measured). The visible Markdown already updates on the 400ms doc ticker,
    // so per-token writes bought nothing. Reset at each turn's first token.
    @ObservationIgnored private var liveAnswer = ""
    @ObservationIgnored private var liveReason = ""
    @ObservationIgnored private var lastFlushNs: UInt64 = 0
    private static let flushIntervalNs: UInt64 = 100_000_000   // 100ms

    private func resetLiveBuffers() {
        liveAnswer = ""
        liveReason = ""
        lastFlushNs = 0
    }

    // Flush at most every flushIntervalNs; force at turn end. Equality-guarded
    // so an idle gap (a tool prefill with no yield) writes nothing.
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

    // Seal the streams at end of turn so the settled docs are the full parse
    // (finish resolves what a snapshot's open block could not).
    private func finishDocs(_ idx: Int) {
        if messages.indices.contains(idx), !messages[idx].fromUser {
            messages[idx].answerDoc = messages[idx].answerStream.finish()
            messages[idx].reasoningDoc =
                messages[idx].reasoningStream.finish()
        }
    }

    // The whimsical list to draw from right now: while a reply is still
    // ingesting the prompt it is the in-flight turn's stage (prefill / documents
    // / vision), and once tokens flow it is the reasoning verbs. Set at send.
    private var activePrefillStage: Whimsical.Stage = .prefill
    private var whimsicalStage: Whimsical.Stage {
        if consulting {
            return .consulting
        }
        return prefilling ? activePrefillStage : .reasoning
    }

    // Cycle the three playful phrases every ~5s while a reply runs (prefill
    // and decode), each a different word, so the transcript quip, the
    // reasoning label, and the status bar never show the same one.
    private func phraseCycler() -> Task<Void, Never> {
        Task { @MainActor in
            while busy && !Task.isCancelled {
                let p = Whimsical.trio(whimsicalStage)
                thinkStatus = p.first
                thinkLabel = p.second
                barStatus = p.third
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }

    // phys_footprint in GiB (Apple's pressure metric): dirty + wired anonymous
    // memory. Excludes the mmapped weights, which refreshStats adds
    // separately.
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

    // Sum of the set's weight.bin blobs: the mmapped weight bytes the status
    // bar reports as the "+ N.NGB weights" term.
    private static func weightBytes(_ setDir: URL) -> Int64 {
        let fm = FileManager.default
        var total: Int64 = 0
        let en = fm.enumerator(at: setDir,
                               includingPropertiesForKeys: [.fileSizeKey])
        if let en {
            for case let url as URL in en
                where url.lastPathComponent == "weight.bin" {
                let sz = (try? url.resourceValues(
                    forKeys: [.fileSizeKey]))?.fileSize ?? 0
                total += Int64(sz)
            }
        }
        return total
    }

    // The model's own sampling matrix (generation_config.json's
    // "_sampling_presets"), falling back to the Qwen3.5/QwenPaw card values for
    // a set that predates the matrix.

    private static func presets(_ setDir: URL) -> SamplingPresets {
        let url = setDir.appendingPathComponent("generation_config.json")
        return (try? SamplingPresets.from(generationConfig: url,
                                          fallback: .qwen35)) ?? .qwen35
    }

    // Live ETA for the in-flight download / optimize, from the phase's elapsed
    // time and its progress fraction; nil until there is enough to
    // extrapolate.

    var downloadETA: String? { Self.eta(phaseStart, downloadFraction) }

    // A monotonic countdown from the held finish time (tightenOptimizeETA), so
    // the shown remaining never jumps back up as per-program estimates jitter.
    var optimizeETA: String? {
        optimizeFinish.map { finish in
            Self.formatETA(max(0, finish.timeIntervalSinceNow))
        }
    }

    // Pull the projected optimize finish EARLIER when a fresh estimate warrants,
    // never later, so the shown ETA only counts down. Called as each program
    // finishes compiling (where the raw estimate jitters up and down).
    private func tightenOptimizeETA() {
        if compileFraction > 0.02 {
            let now = Date()
            let elapsed = now.timeIntervalSince(phaseStart)
            let candidate = elapsed * (1 - compileFraction) / compileFraction
            let current = optimizeFinish?.timeIntervalSince(now) ?? candidate
            // Ease toward each fresh remaining-time estimate (EMA) rather than
            // locking to the most optimistic one, so a warm early phase cannot
            // strand the ETA under a cold later phase.
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

    // "ETA: ~N seconds" under a minute, else "ETA: ~N minutes" ("~1 minute"
    // singular).
    static func formatETA(_ seconds: Double) -> String {
        let s = max(1, Int(seconds.rounded()))
        let m = Int((Double(s) / 60).rounded())
        let body = s < 60 ? "~\(s) seconds"
            : "~\(m) minute" + (m == 1 ? "" : "s")
        return "ETA: " + body
    }

    // Estimated (download, optimize) minutes for `name`: the model's own
    // measured times when it has them, else the 0.8B baseline scaled by byte
    // ratio. nil until the 0.8B has been through both phases once.

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
            // Download scales with bytes; compile time tracks op count,
            // superlinear in model size (~n^1.3 measured), so a linear byte
            // ratio undersells a big set's first-install optimize.
            result = (Self.minutes(dl > 0 ? dl : dl0 * ratio),
                      Self.minutes(opt > 0 ? opt : opt0 * pow(ratio, 1.3)))
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

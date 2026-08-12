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
        let url: URL?
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
        let thumbnail: CGImage?
    }

    struct ClipAttachment: Identifiable {
        let id = UUID()
        let name: String
        let url: URL
        let isVideo: Bool
        let thumbnail: CGImage?

        func spans(_ media: Gemma4Media,
                   onFrame: (@Sendable (VideoPeek) -> Void)? = nil)
            async throws -> [(ContentPart, SoftSpan)] {
            var out: [(ContentPart, SoftSpan)] = []
            if isVideo {
                var seen = 0
                out.append((.video, try await media.video(url: url) {
                    img, _ in
                    if let onFrame,
                       let peek = VideoPeek(index: seen, full: img) {
                        onFrame(peek)
                    }
                    seen += 1
                }))
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
        let id: Int
        let emitted: String          // the name the model asked for
        let label: String            // display name (resolved tool, or emitted)
        let symbol: String
        let args: String
        var result: String?
    }

    struct Message: Identifiable {
        let id = UUID()
        let fromUser: Bool
        var text: String
        var images: [CGImage] = []
        var clips: [URL] = []
        var docs: [DocRef] = []
        var toolRounds: [ToolRound] = []
        var reasoning = ""
        var placeholder = false
        var loopStopped = false
        var answerDoc = Markdown.Document.empty
        var reasoningDoc = Markdown.Document.empty
        let answerStream = MarkdownStream()
        let reasoningStream = MarkdownStream()
    }

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
    var readOnly = false
    var currentConversationId: UUID?
    var generatedTitle: String?
    var theme: AppTheme = {
        AppTheme(rawValue:
            UserDefaults.standard.string(forKey: "theme") ?? "") ?? .system
    }() {
        didSet { UserDefaults.standard.set(theme.rawValue, forKey: "theme") }
    }
    var textZoom: Int = ChatModel.clampZoom(
        UserDefaults.standard.integer(forKey: "textZoom")) {
        didSet { UserDefaults.standard.set(textZoom, forKey: "textZoom") }
    }
    static let zoomLimit = 2

    static func clampZoom(_ notches: Int) -> Int {
        min(max(notches, -zoomLimit), zoomLimit)
    }

    var textScale: CGFloat { ChatModel.zoomScale(textZoom) }

    static func zoomScale(_ notch: Int) -> CGFloat {
        1 + CGFloat(notch) / 10
    }

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
    var caret = 0
    var attachedImages: [ImageAttachment] = []
    var attachedDocs: [Doc] = []
    private(set) var converting = 0
    var attachedClips: [ClipAttachment] = []
    var status = "loading model..."
    var ready: Bool { session != nil && !compiling }
    var busy: Bool { genTask != nil }

    var lookingAt: VideoPeek?
    var watching = false
    var inTurn: Bool { busy || listening || speech.engaged }

    var voiceReady: Bool {
        lastTurnSpoken && canAttachAudio && !busy && !listening
            && !speech.engaged && input.isEmpty
    }

    private(set) var lastTurnSpoken = false
    var statsLabel = ""

    private(set) var modelShape: ModelShape?

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

    private static func tokens(_ n: Int) -> String {
        n >= 1024 && n % 1024 == 0 ? "\(n / 1024)K" : n.formatted(.number)
    }

    private static func weight(_ bytes: Int) -> String {
        let gb = Double(bytes) / 1_073_741_824
        return gb >= 1 ? String(format: "%.1f GB", gb)
                       : String(format: "%.0f MB",
                                Double(bytes) / 1_048_576)
    }
    let speech = VoiceSession()
    var listening = false
    var heardSeconds = 0.0
    var hearingSpeech = false
    var speechLevel = 0.0
    var prefilling = false
    var consulting = false
    var thinkStatus = "Thinking"
    var thinkLabel = "Thinking"
    var eulaAccepted = UserDefaults.standard.bool(forKey: "eulaAccepted")
    var accepted = UserDefaults.standard.bool(forKey: "disclaimerAccepted")
    private static func startModel() -> String {
        let saved = UserDefaults.standard.string(forKey: "modelName")
            ?? Models.start
        return Models.all.contains(saved) ? saved : Models.start
    }
    var modelName: String = ChatModel.startModel()
    var downloadName: String? = nil
    var downloading = false
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

    var canSend: Bool {
        let text = input.trimmingCharacters(
            in: .whitespacesAndNewlines)
        let has = !text.isEmpty || !attachedImages.isEmpty
            || !attachedDocs.isEmpty || !attachedClips.isEmpty
        return ready && !busy && has
    }

    var typing: Bool { !input.isEmpty }

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

    private(set) var modelSupportsVision = false

    private(set) var modelSupportsSoftTokens = false
    @ObservationIgnored var media: Gemma4Media?

    var canAttachImages: Bool {
        modelSupportsVision || modelSupportsSoftTokens
    }
    var canAttachAudio: Bool { modelSupportsSoftTokens }
    var canAttachVideo: Bool {
        modelSupportsSoftTokens
            && !attachedClips.contains { c in c.isVideo }
    }

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

    private(set) var modelSupportsThinking = true

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

    var compiling = false
    var compileDoneLoC = 0
    var compileTotalLoC = 0
    var compileFraction: Double {
        compileTotalLoC > 0
            ? min(1.0, Double(compileDoneLoC) / Double(compileTotalLoC)) : 0
    }

    var loadError: String? = nil

    var firstCompile = true

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
    var systemPrompt: String = UserDefaults.standard
        .string(forKey: "systemPrompt") ?? ChatModel.defaultSystemPrompt {
        didSet {
            UserDefaults.standard.set(systemPrompt, forKey: "systemPrompt")
            ChatModel.wipePrecook()
        }
    }
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
    var renderMarkdown: Bool = UserDefaults.standard
        .object(forKey: "renderMarkdown") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(renderMarkdown,
                                      forKey: "renderMarkdown")
        }
    }
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
    var showDebug = false
    var traceEvents: [TraceEvent] = []
    private let traceFile: TraceFile? = debugBuild ? TraceFile() : nil
    private let instrument = Instrument.install()
    var tracePath: String { traceFile?.path ?? "" }
    var diagPath: String { Diag.shared.path }
    private static let traceCap = 2000
    // A static logit bias on reasoning branch-openers while thinking. Robust
    // across 0.5-4.0. TODO: surface as a setting.
    private static let overthinkLambda: Float = 1.0

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

    var measuredTG: Double {
        let v = UserDefaults.standard.double(forKey: Self.tgKey(modelName))
        return v > 0 ? v : 20
    }

    private func recordTG(_ tg: Double) {
        if tg > 0 {
            let old = UserDefaults.standard.double(
                forKey: Self.tgKey(modelName))
            let ema = old > 0 ? 0.7 * old + 0.3 * tg : tg
            UserDefaults.standard.set(ema, forKey: Self.tgKey(modelName))
        }
    }

    var thinkTokenCap: Int {
        let raw = thinkBudget.seconds * measuredTG
        return max(200, Int((raw / 100).rounded()) * 100)
    }

    private let log = Logger(subsystem: "io.github.leok7v.gadeon",
                              category: "ui")
    private var chat: AneChat?
    private var ggufBackend: (any AgentBackend)?
    private var ggufTemplate = ""
    private var ggufVocabCount = 0
    private var session: ChatSession?
    private var genTask: Task<Void, Never>?
    private let primer = Primer()
    private var lastPP = 0.0
    private var perImageTokens = 256
    private var activePresets: SamplingPresets = .qwen35
    @ObservationIgnored private var phaseStart = Date()
    @ObservationIgnored private var optimizeFinish: Date?

    func acceptEULA() {
        eulaAccepted = true
        UserDefaults.standard.set(true, forKey: "eulaAccepted")
        load(name: modelName)
    }

    func accept() {
        accepted = true
        UserDefaults.standard.set(true, forKey: "disclaimerAccepted")
    }

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

    private func buildGguf(name: String, path: String) {
        compiling = true
        compileDoneLoC = 0
        compileTotalLoC = 0
        phaseStart = Date()
        loadError = nil
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
                if Gemma4Model.isGemma4(path: path) {
                    let c = try GemmaChat(ggufPath: path)
                    let gpu = try c.metalBackend()
                    built = (gpu, c.chatTemplate,
                             c.vocabCount, c.samplingPresets, c.shape)
                    media = c.media(ctx: gpu.ctx)
                } else {
                    let c = try MetalChat(ggufPath: path)
                    built = (c.backend(), c.chatTemplate,
                             c.tokenizer.vocabCount, c.samplingPresets,
                             c.shape)
                }
            } catch {
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
            let setDir = dest.appendingPathComponent(src.revision)
            Task.detached { [weak self] in
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
                    failure = "download failed verification, try again"
                } catch {
                }
                await MainActor.run {
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

    private func build(setDir: URL, attempt: Int = 0) {
        compiling = true
        compileDoneLoC = 0
        compileTotalLoC = 0
        phaseStart = Date()
        optimizeFinish = nil
        activePresets = Self.presets(setDir)
        loadError = nil
        let key = Self.compiledKey(setDir)
        firstCompile = !UserDefaults.standard.bool(forKey: key)
            || AneCache.shared.containerMigrated()
        Task.detached { [weak self] in
            let cacheBefore = AneCache.shared.buildBegan()
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
                    await built.loadMTP()
                    await built.engine.offloadVision()
                    AneCache.shared.buildEnded(setDir: setDir,
                                               before: cacheBefore)
                    await MainActor.run {
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

    // Named for this: the outgoing conversation is SAVED before the switch
    // clears it, exactly as New Chat saves before starting one.
    private func commitSwitch(_ name: String) {
        commitCurrent()
        genTask?.cancel()
        session?.requestStop()
        primer.cancel()
        chat = nil
        ggufBackend = nil
        session = nil
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
        // The identity goes with the messages, or the next conversation
        // commits into this one's id and overwrites it.
        messages = []
        traceEvents = []
        currentConversationId = nil
        readOnly = false
        generatedTitle = nil
        followupHint = ""
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

    private(set) var diskRevision = 0

    func isDownloaded(_ name: String) -> Bool { isOnDisk(name) }

    func deleteModel(_ name: String) {
        if name != modelName, !busy, !downloading {
            ChatModel.erase(name)
            diskRevision += 1
        }
    }

    private static func erase(_ name: String) {
        let fm = FileManager.default
        try? fm.removeItem(
            at: Bundle.modelStore().appendingPathComponent(name))
        for on in [true, false] {
            try? fm.removeItem(at: precookURL(name, thinking: on))
        }
        if let sha = ModelCatalog.source(name)?.revision {
            let d = UserDefaults.standard
            for k in d.dictionaryRepresentation().keys
            where k.hasPrefix("compiled.\(sha).") {
                d.removeObject(forKey: k)
            }
        }
    }

    private static func pruneUnavailable() {
        let offered = Set(Models.all)
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

    private static let minilmPath: String? = WikiSlugs.bundledModel?.path

    private var toolRunner: (any ToolRunner)? {
        SafeToolRunner(slugsPath: wikipedia ? ChatModel.minilmPath : nil,
                       wikipedia: wikipedia, network: webAccess)
    }

    private static let southernRegions: Set<String> = [
        "AU", "NZ", "AR", "CL", "UY", "PY", "BO", "PE", "BR", "ZA", "NA",
        "BW", "ZW", "ZM", "MZ", "MG", "LS", "SZ", "AO", "MW", "PG", "FJ",
        "NC", "WS", "TO", "VU", "SB",
    ]

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
        if canAttachAudio {
            s += "\nThe user may speak to you. Their speech reaches you "
                + "already encoded by your own audio tower, so you hear it "
                + "directly: never say you cannot process audio, and never "
                + "treat it as a transcript someone pasted. Spoken words are "
                + "the user talking TO you -- answer them as you would the "
                + "same words typed, and write them out only when asked to."
        }
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

    private static var precookDir: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? "app")
            .appendingPathComponent("precook", isDirectory: true)
    }

    // Keyed by the reasoning flag as well as the model: a template that spells
    // thinking inside its system block renders two different blocks, so they
    // are two caches rather than one that misses every time the flag moves.
    private static func precookURL(_ name: String, thinking: Bool) -> URL {
        precookDir.appendingPathComponent(
            name + (thinking ? ".think" : "") + ".ctx")
    }

    private static func ensurePrecookDir() {
        try? FileManager.default.createDirectory(
            at: precookDir, withIntermediateDirectories: true)
    }

    static func wipePrecook() {
        try? FileManager.default.removeItem(at: precookDir)
    }

    private func primeSession(resetFirst: Bool = false) {
        if let s = session {
            let on = thinkingActive
            primedThinking = on
            ChatModel.ensurePrecookDir()
            let url = ChatModel.precookURL(modelName, thinking: on)
            Task { await s.primeOrCook(at: url, resetFirst: resetFirst) }
        }
    }

    // The reasoning flag the live conversation's system block was rendered
    // with. Reasoning can always be turned OFF later by closing the channel as
    // it opens, but never ON: the marker some templates carry lives in that
    // block, which is in the KV and is subtracted from every later delta.
    @ObservationIgnored private var primedThinking = true

    func newChat() {
        followupHint = ""
        lastTurnSpoken = false
        speech.stopSpeaking()
        commitCurrent()
        readOnly = false
        currentConversationId = nil
        generatedTitle = nil
        messages = []
        traceEvents = []
        statsLabel = ""
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

    @ObservationIgnored private var metaTask: Task<Void, Never>?

    private func runMetaTurns() {
        let chars = messages.reduce(0) { sum, m in sum + m.text.count }
        let titled = !readOnly && generatedTitle == nil
            && messages.count >= 2 && chars > 200
        if let session, !readOnly, titled || offersFollowupHint {
            let running = metaTask
            metaTask = Task { @MainActor in
                await running?.value
                if titled, !Task.isCancelled {
                    let t = await session.makeTitle()
                    if !t.isEmpty {
                        generatedTitle = t
                        commitCurrent()
                    }
                }
                if offersFollowupHint, !Task.isCancelled {
                    followupHint = await session.makeFollowup()
                }
                metaTask = nil
            }
        }
    }

    private func drainMeta() async {
        if let running = metaTask {
            metaTask = nil
            session?.requestStop()
            running.cancel()
            await running.value
        }
    }

    var suggestFollowups: Bool = UserDefaults.standard
        .object(forKey: "suggestFollowups") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(suggestFollowups,
                                      forKey: "suggestFollowups")
            if !suggestFollowups { followupHint = "" }
        }
    }

    var offersFollowupHint: Bool {
        suggestFollowups && modelName != Models.fallback
    }

    var followupHint = ""

    func acceptFollowupHint() {
        if !followupHint.isEmpty {
            input = followupHint
            caret = input.utf16.count
        }
    }

    func toggleSettings() {
        showSettings.toggle()
        if showSettings { showDebug = false }
    }

    func openSettings() {
        showDebug = false
        showSettings = true
    }

    func toggleDebug() {
        showDebug.toggle()
        if showDebug { showSettings = false }
    }

    func toggleThinking() {
        if modelSupportsThinking {
            thinking.toggle()
            let s = session
            let on = thinking
            // An empty conversation can be re-primed onto the other system
            // block, which is the only way to OPEN reasoning; one already
            // under way takes the closing route instead.
            let fresh = messages.isEmpty
            ChatModel.ensurePrecookDir()
            let url = ChatModel.precookURL(modelName, thinking: on)
            let hud = thinkingFlash(on, fresh: fresh)
            if fresh { primedThinking = on }
            Task {
                await s?.setThinking(on)
                await s?.setSuppressReasoning(!on)
                if fresh { await s?.primeOrCook(at: url, resetFirst: true) }
            }
            flashHUD(hud)
        }
    }

    private func thinkingFlash(_ on: Bool, fresh: Bool) -> String {
        var out = on ? "Thinking: On" : "Thinking: Off"
        if on, !fresh, !primedThinking {
            out = "Thinking: On, from the next chat"
        }
        return out
    }

    func quickAnswer() {
        let s = session
        Task { await s?.requestQuickAnswer() }
    }

    func attachImage(_ data: Data, name: String, at offset: Int) {
        let dup = attachedImages.contains { $0.data == data }
        if canAttachImages, attachedImages.count < Self.maxImages, !dup {
            let unique = uniqueName(name)
            let thumb = VisionPreprocess.thumbnail(data, maxPx: 96)
            attachedImages.append(
                ImageAttachment(name: unique, data: data, thumbnail: thumb))
            insertRef(unique, at: offset)
        }
    }

    func attachClip(_ url: URL, isVideo: Bool, at offset: Int) {
        let allowed = isVideo ? canAttachVideo : canAttachAudio
        let dup = attachedClips.contains { c in c.url == url }
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

    private static func read(_ name: String, _ text: String?,
                             _ since: Date, _ limit: Int) {
        let bytes = text?.utf8.count ?? 0
        let cut = bytes > limit ? ", TRUNCATED to \(limit)" : ""
        Diag.shared.report(String(
            format: "attach document %@ -> %d bytes%@ (%.1fs)", name, bytes,
            cut, Date().timeIntervalSince(since)))
    }

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

    private static func keep(_ from: URL, _ name: String) -> URL? {
        let fm = FileManager.default
        let dir = attachments.appendingPathComponent(UUID().uuidString,
                                                     isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let to = dir.appendingPathComponent(name)
        return (try? fm.copyItem(at: from, to: to)) != nil ? to : nil
    }

    private static func markdown(of url: URL) async -> String? {
        await Task.detached(priority: .userInitiated) {
            url.pathExtension.lowercased() == "pdf"
                ? try? await Pdf2md.markdown(of: url)
                : try? Docs2md.markdown(of: url)
        }.value
    }

    private static func refs(_ docs: [Doc]) -> [DocRef] {
        docs.compactMap { doc in
            doc.url.map { url in
                DocRef(url: url, bytes: doc.content.utf8.count,
                       short: doc.short)
            }
        }
    }

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
    static let convertExts: Set<String> = Set(["pdf"] + Docs2md.readable)
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
    static let maxClips = 2
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

    @ObservationIgnored private var mic: Microphone?
    @ObservationIgnored private var gate: SpeechGate?
    @ObservationIgnored private let heard = HeardSpeech()
    @ObservationIgnored private var rateInUse: Double = 1
    @ObservationIgnored private var endOfTurn: Task<Void, Never>?
    @ObservationIgnored private var micPhrases: Task<Void, Never>?
    private static let endOfTurnSilence = 1.5

    func voice() {
        if listening { endListening() } else { beginListening() }
    }

    private func beginListening() {
        speech.stopSpeaking()
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

    private func watchForEndOfTurn() {
        endOfTurn?.cancel()
        var ticks = 0
        endOfTurn = Task { @MainActor in
            while listening && !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(60))
                heardSeconds = heard.seconds
                hearingSpeech = gate?.hearing ?? false
                speechLevel = Double(gate?.level ?? 0)
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
            flashHUD(Whimsical.current(.heard, hold: 0), prominent: true,
                     seconds: 2)
            sendSpoken(said)
        }
    }

    func stop() {
        speech.stopSpeaking()
        genTask?.cancel()
        session?.requestStop()
    }

    func factoryReset() {
        let d = UserDefaults.standard
        if let id = Bundle.main.bundleIdentifier {
            d.removePersistentDomain(forName: id)
        }
        clearCompileCache()
        ChatModel.wipePrecook()
        try? FileManager.default.removeItem(at: Bundle.modelStore())
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

    private func clearCompileCache() {
        let d = UserDefaults.standard
        for k in d.dictionaryRepresentation().keys
            where k.hasPrefix("compiled.") {
            d.removeObject(forKey: k)
        }
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
        followupHint = ""
        let attached = !attachedImages.isEmpty || !attachedClips.isEmpty
        if attached, ready, !busy {
            if modelSupportsSoftTokens {
                sendSoft()
            } else {
                sendVision()
            }
        } else {
            sendText()
        }
    }

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

    private static let spokenPrompt = "Reply to what I just said."

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
                await drainMeta()
                let began = Date()
                let phrases = phraseCycler()
                let ticker = statsTicker(session)
                await session.setReasoningCaps(soft: thinkTokenCap,
                                               hard: thinkTokenCap * 2)
                do {
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
                    await finishTurn(session, idx, began)
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
            if !images.isEmpty || !clips.isEmpty {
                Footprint.report("encoded \(images.count) img "
                    + "\(clips.count) clip")
            }
            return (parts, spans, widest)
        }.value
    }

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

    nonisolated private static func note(_ kind: String, _ name: String,
                                        _ rows: Int, _ since: Date) {
        Diag.shared.report(String(
            format: "attach %@ %@ -> %d soft tokens (%.1fs)",
            kind, name, rows, Date().timeIntervalSince(since)))
    }

    private static func attachmentFailed(_ error: any Error) -> String {
        error is Gemma4MediaError
            ? "\(error)" : "Could not process the attachment."
    }

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
    static let sampleInterest = "A savings account starts with $1,000 and "
        + "earns 5% interest each year, so every year its balance "
        + "multiplies by 1.05. Use the calculator to find the balance "
        + "after 22 years."
    static let sampleCookies = "Bakers weigh every ingredient as a "
        + "percentage of the flour weight. In my cookie recipe, butter "
        + "is 65% of the flour. I have 350 grams of flour. Use the "
        + "calculator to find how many grams of butter I need."

    var calcSampleIsInterest: Bool { modelName != Models.fallback }

    static let samplePicture: Data? = Bundle.main
        .url(forResource: "dogs-beach", withExtension: "jpg")
        .flatMap { url in try? Data(contentsOf: url) }

    static let sampleVideo: URL? = Bundle.main
        .url(forResource: "dogs-beach", withExtension: "mp4")
    static let samplePictureThumb: CGImage? = samplePicture.flatMap { data in
        VisionPreprocess.thumbnail(data, maxPx: 128)
    }

    static let sampleReport =
        "Which block gives the most fruit per tree, and what makes that "
        + "surprising?"

    static let samplePdf: URL? = Bundle.main
        .url(forResource: "harvest-report", withExtension: "pdf")

    static let sampleClipThumb: CGImage? = Bundle.main
        .url(forResource: "dogs-beach-poster", withExtension: "jpg")
        .flatMap { url in try? Data(contentsOf: url) }
        .flatMap { data in VisionPreprocess.thumbnail(data, maxPx: 128) }

    var alwaysShowSamples: Bool =
        UserDefaults.standard.bool(forKey: "alwaysShowSamples") {
        didSet {
            UserDefaults.standard.set(alwaysShowSamples,
                                      forKey: "alwaysShowSamples")
        }
    }
    private var usedSamples: Set<String> = Set(
        UserDefaults.standard.stringArray(forKey: "usedSamples") ?? [])

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

    var canOfferDocumentSample: Bool { ChatModel.samplePdf != nil }

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

    private static let toolGlyphs: [String: (label: String, symbol: String)] = [
        "get_current_time": ("Current Time", "clock"),
        "calculator": ("Calculator", "function"),
        "web_search": ("Web Search", "magnifyingglass"),
        "fetch_url": ("URL Fetch", "link"),
        "get_news": ("News", "newspaper"),
        "get_weather": ("Weather", "cloud.sun"),
        "wikipedia_query": ("Wikipedia", "books.vertical"),
    ]

    private func applyToolRound(_ event: ToolRoundEvent, at idx: Int) {
        if messages.indices.contains(idx) {
            let known = event.resolved
                .flatMap { name in ChatModel.toolGlyphs[name] }
            let args = event.params
                .map { param in "\(param.name): \"\(param.value)\"" }
                .joined(separator: "  ")
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

    private func sendVision() {
        let raw = input
        let body = promptFor(raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let ask = body.isEmpty ? VLPrompt.defaultPrompt : body
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
            let previews = images.compactMap { img in
                VisionPreprocess.thumbnail(img.data, maxPx: 640)
            }
            messages.append(Message(fromUser: true, text: display,
                                    images: previews))
            messages.append(Message(fromUser: false, text: ""))
            let idx = messages.count - 1
            resetLiveBuffers()
            let tiled = allowsTiling && visionMode == .tile && images.count == 1
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
                await drainMeta()
                let began = Date()
                let phrases = phraseCycler()
                let ticker = statsTicker(session)
                await session.setReasoningCaps(soft: thinkTokenCap,
                                               hard: thinkTokenCap * 2)
                do {
                    let grid = await session.visionGrid() ?? VisionGrid.canonical
                    perImageTokens = grid.mergedTokens
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
                    await finishTurn(session, idx, began)
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
                await drainMeta()
                let began = Date()
                let phrases = phraseCycler()
                let ticker = statsTicker(session)
                await session.setReasoningCaps(soft: thinkTokenCap,
                                               hard: thinkTokenCap * 2)
                let stream = session.reply(prompt, onReasoning: onReasoning,
                                           onTool: onTool,
                                           onToolRound: onToolRound)
                for await piece in stream {
                    if prefilling { prefilling = false }
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
                await finishTurn(session, idx, began)
            }
        }
    }

    // A turn that produced nothing at all takes its two bubbles with it; one
    // that spent tool rounds keeps them, so the transcript can say it reached
    // no answer instead of erasing the question.
    private func finishTurn(_ session: ChatSession, _ idx: Int,
                            _ since: Date) async {
        let outcome = await session.turnOutcome
        if outcome == .stopped {
            if messages.count >= 2 { messages.removeLast(2) }
        } else {
            await refreshStats(session)
            recordTG(await session.lastMetrics.tg)
            noteLoopStop(await session.lastMetrics, idx)
            commitCurrent()
            runMetaTurns()
        }
        reportTurn(await session.lastMetrics, outcome, idx, since)
    }

    // One line per turn, so a sweep across models and thinking modes diffs by
    // grepping rather than by reading a whole trace.
    private func reportTurn(_ m: TurnMetrics,
                            _ outcome: ChatSession.TurnOutcome,
                            _ idx: Int, _ since: Date) {
        Diag.shared.report(String(
            format: "[turn] %@ thinking=%@ outcome=%@ end=%@ tools=%@ "
                + "think=%d content=%d ctx=%d %.1fs",
            modelName, thinkingActive ? "on" : "off", outcome.rawValue,
            m.endReason, toolDigest(idx), m.thinkTokens, m.contentTokens,
            m.ctx, Date().timeIntervalSince(since)))
    }

    // How many calls and how many of them were DISTINCT. A model re-asking
    // one expression is what spends a whole tool budget without progress,
    // and a count alone cannot show it.
    private func toolDigest(_ idx: Int) -> String {
        var out = "none"
        let rounds = messages.indices.contains(idx)
            ? messages[idx].toolRounds : []
        if !rounds.isEmpty {
            var counts: [String: Int] = [:]
            for round in rounds { counts[round.label, default: 0] += 1 }
            let names = counts.sorted { a, b in a.value > b.value }
                .map { pair in "\(pair.key)x\(pair.value)" }
                .joined(separator: ",")
            let distinct = Set(rounds.map { r in r.label + r.args }).count
            out = "\(names)/\(distinct)distinct"
        }
        return out
    }

    private func noteLoopStop(_ m: TurnMetrics, _ idx: Int) {
        if m.endReason == "loop-breaker", messages.indices.contains(idx),
           messages[idx].text.isEmpty {
            messages[idx].loopStopped = true
        }
    }

    private func refreshStats(_ session: ChatSession) async {
        let t = await session.lastMetrics
        if t.pp > 0 { lastPP = t.pp }
        if t.ctx > 0 {
            let tokens = thinkingActive
                ? "🤔 \(t.thinkTokens) 💬 \(t.contentTokens)"
                : "💬 \(t.thinkTokens + t.contentTokens)"
            statsLabel = "⇄ \(t.ctx.formatted(.number))  \(tokens) "
                + String(format: "🐏 %.1fGB t/s: %.1f/%.1f",
                         Self.footprintGiB(), t.pp, t.tg)
        }
    }

    private func statsTicker(_ session: ChatSession) -> Task<Void, Never> {
        Task { @MainActor in
            while busy && !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(400))
                if busy {
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

    @ObservationIgnored private var liveAnswer = ""
    @ObservationIgnored private var liveReason = ""
    @ObservationIgnored private var lastFlushNs: UInt64 = 0
    private static let flushIntervalNs: UInt64 = 100_000_000

    private func resetLiveBuffers() {
        liveAnswer = ""
        liveReason = ""
        lastFlushNs = 0
    }

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

    private func finishDocs(_ idx: Int) {
        if messages.indices.contains(idx), !messages[idx].fromUser {
            messages[idx].answerDoc = messages[idx].answerStream.finish()
            messages[idx].reasoningDoc =
                messages[idx].reasoningStream.finish()
        }
    }

    private var activePrefillStage: Whimsical.Stage = .prefill
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
                try? await Task.sleep(for: .seconds(listening ? 2 : 5))
            }
        }
    }

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

    private static func presets(_ setDir: URL) -> SamplingPresets {
        let url = setDir.appendingPathComponent("generation_config.json")
        return (try? SamplingPresets.from(generationConfig: url,
                                          fallback: .qwen35)) ?? .qwen35
    }

    var downloadETA: String? { Self.eta(phaseStart, downloadFraction) }

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

final class HeardSpeech: @unchecked Sendable {
    private let lock = NSLock()
    private var said: [SpeechGate.Utterance] = []
    private var count = 0
    private var at = Date()

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

    var seconds: Double {
        lock.lock()
        defer { lock.unlock() }
        return said.reduce(0.0) { sum, u in sum + u.seconds }
    }

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

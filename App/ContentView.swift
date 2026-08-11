import AVKit
import MD
import SwiftUI

struct ContentView: View {

    @Bindable var model: ChatModel

    static let answerStyle = MarkdownStyle(
        bodySize: isOS ? 17 : 14,
        codeSize: isOS ? 13 : 12,
        headingSizes: isOS ? [25, 21, 19, 17, 17, 16]
                           : [22, 18, 16, 14, 14, 13])
    static let reasoningStyle = MarkdownStyle(
        bodySize: isOS ? 14 : 12,
        codeSize: isOS ? 12 : 10,
        headingSizes: isOS ? [18, 16, 15, 14, 14, 14]
                           : [16, 14, 13, 12, 12, 12],
        textColor: Color(white: 0.53),
        secondaryColor: Color(white: 0.6),
        highlightCode: false)
    // MarkdownStyle point sizes are fixed: the MD renderer builds
    // systemFont(ofSize:), which no content-size category touches.
    @ScaledMetric(relativeTo: .body) private var typeScale: CGFloat = 1

    private var textScale: CGFloat { typeScale * model.textScale }

    static func scaled(_ s: MarkdownStyle, by k: CGFloat) -> MarkdownStyle {
        var out = s
        out.bodySize = s.bodySize * k
        out.codeSize = s.codeSize * k
        out.headingSizes = s.headingSizes.map { size in size * k }
        out.blockSpacing = s.blockSpacing * k
        out.paragraphSpacing = s.paragraphSpacing * k
        out.listIndent = s.listIndent * k
        return out
    }
    // Not FocusState: the AppKit editor owns first responder and syncs this
    // both ways, which a FocusState binding cannot cross into an
    // NSViewRepresentable.
    @State private var promptFocused = false
    @State private var chatHeight: CGFloat = 0
    @State private var peek = ToolPeek()
    @State private var calloutSize: CGSize = .zero
    @ScaledMetric(relativeTo: .body) private var popScale: CGFloat = 1
    @State private var follow = true
    @State private var tailInset: CGFloat = 0
    @State private var tailCollapse: Task<Void, Never>?
    @ScaledMetric(relativeTo: .body) private var tailInsetSize: CGFloat = 64
    @State private var optionDown = false
    @State private var showModelInvite = false
    @State private var dropActive = false
    // A magnification is reported relative to the gesture's own start, so
    // `pinchFrom` holds the stop it began at.
    @State private var pinchFrom: Int?
    @State private var pinchShown: Int?

    // simultaneousGesture, or it takes the scroll view's own gestures with it
    // and the transcript stops scrolling.
    private var pinchZoom: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                let from = pinchFrom ?? model.textZoom
                if pinchFrom == nil { pinchFrom = from }
                let want = ChatModel.zoomScale(from) * value.magnification
                let notch = ChatModel.zoomNotch(nearest: want)
                if notch != pinchShown {
                    pinchShown = notch
                    model.flashZoom(notch)
                }
            }
            .onEnded { value in
                let from = pinchFrom ?? model.textZoom
                let want = ChatModel.zoomScale(from) * value.magnification
                model.textZoom = ChatModel.zoomNotch(nearest: want)
                pinchFrom = nil
                pinchShown = nil
            }
    }
    // Seeded from the ring holder rather than a placeholder: the read is
    // idempotent, so a view re-init cannot advance the rotation.
    @State private var optimizingPhrase = Whimsical.current(.optimizing)
    @State private var downloadingPhrase = Whimsical.current(.downloading)
    @State private var sidebarOpen = false
    @State private var findActive = false
    @State private var findController = MarkdownFindController()
    @State private var findQuery = ""
    @State private var findCount = 0
    @State private var findCurrent = 0
    @FocusState private var findFocused: Bool
    @State private var actionsExpanded = false
    @State private var actionsHovering = false

    var body: some View {
        content
            .modifier(OptionKeyMonitor(down: $optionDown))
            .preferredColorScheme(model.theme.scheme)
            .environment(\.appTextScale, model.textScale)
            // The default font too, so anything that never names one does not
            // sit at the system size while labelled text grows around it.
            .font(appTextFont(.body, model.textScale))
            // A first run starts the fetch at EULA acceptance instead, so it
            // is not resolved here.
            .onAppear {
                if Models.supported, model.eulaAccepted, model.accepted {
                    model.load(name: model.modelName)
                }
            }
    }

    @ViewBuilder
    private var content: some View {
        if model.showSettings {
            SettingsView(model: model,
                         onClose: { model.showSettings = false })
        } else if model.showDebug {
            DebugView(model: model, onClose: { model.showDebug = false })
        // A pending download takes over even with a model already loaded, or
        // the pick sets state behind the live chat with nothing on screen.
        } else if model.ready && model.downloadName == nil
            && !model.downloading {
            chatShell
        } else {
            onboarding
        }
    }

    private var chatShell: some View {
        ZStack(alignment: .leading) {
            chat
                .overlay { hudOverlay }
                .animation(.easeInOut(duration: 0.4), value: model.hud)
                .navigationTitle("")
                .modifier(ChatChrome(leading: foldersButton,
                                     center: centerBar,
                                     trailing: trailingButtons))
                .disabled(sidebarOpen)
            drawer
        }
        // Hidden: only the Cmd+N shortcut matters, and it has to survive the
        // top-bar button being disabled while the drawer is open.
        .background {
            Button("", action: openNewChat)
                .keyboardShortcut("n", modifiers: .command)
                .disabled(model.busy)
                .hidden()
        }
    }

    @Environment(\.horizontalSizeClass) private var sizeClass

    // nil on a regular layout, so the title bar inherits the toolbar's own
    // size rather than an override.
    private var barGlyph: Font? {
        sizeClass == .compact ? .system(size: 20 * model.textScale) : nil
    }

    private var foldersButton: some View {
        Button(action: toggleSidebar) {
            Image(systemName: "sidebar.leading")
        }
        .font(barGlyph)
        .help("Menu")
    }

    private var newChatButton: some View {
        Button(action: model.newChat) {
            Image(systemName: "square.and.pencil")
        }
        .help("New chat")
        .disabled(model.busy)
    }

    // Expanding hides the centered model picker, so the row takes over the
    // bar rather than shoving it.
    private var trailingButtons: some View {
        HStack(spacing: 12) {
            if !model.messages.isEmpty {
                if actionsExpanded {
                    Button { actionsExpanded = false } label: {
                        Image(systemName: "chevron.right.2")
                    }
                    .help("Hide actions")
                    TranscriptActions(
                        document: model.transcriptDocument,
                        title: model.transcriptTitle,
                        renderMarkdown: $model.renderMarkdown,
                        onFind: openFind, onDebug: debugAction)
                } else {
                    Button { actionsExpanded = true } label: {
                        Image(systemName: "chevron.left.2")
                    }
                    .help("More actions")
                }
            }
            newChatButton
        }
        .font(barGlyph)
        .onHover { inside in actionsHovering = inside }
        .animation(.easeInOut(duration: 0.2), value: actionsExpanded)
        .task(id: actionsIdle) { await collapseActionsAfterIdle() }
    }

    // nil hides the button.
    private var debugAction: (() -> Void)? {
        var out: (() -> Void)? = nil
        if model.statusLine { out = openDebug }
        return out
    }

    // Never on iOS, which has no hover to leave: the cluster stays open there
    // until it is tapped shut.
    private var actionsIdle: Bool {
        !isOS && actionsExpanded && !actionsHovering
    }

    private func collapseActionsAfterIdle() async {
        if actionsIdle {
            try? await Task.sleep(for: .seconds(3))
            if !Task.isCancelled { actionsExpanded = false }
        }
    }

    @ViewBuilder
    private var drawer: some View {
        if sidebarOpen {
            Color.black.opacity(0.3)
                .ignoresSafeArea()
                .transition(.opacity)
                .onTapGesture(perform: closeSidebar)
                .zIndex(1)
            Sidebar(model: model, theme: $model.theme,
                    onClose: closeSidebar, onOpen: openConversation,
                    onNewChat: openNewChat,
                    onSettings: openSettings)
                .frame(width: 300 * model.textScale)
                .frame(maxHeight: .infinity)
                .background(.bar)
                .transition(.move(edge: .leading))
                .zIndex(2)
        }
    }

    private var onboarding: some View {
        Group {
            if !model.eulaAccepted {
                EULAView(onAgree: model.acceptEULA)
            } else if let name = model.downloadName {
                // The App Store requires explicit download consent before any
                // fetch, so it comes before the disclaimer. A third-party
                // licence comes before both: the size question is ours to put,
                // the licence is not.
                if GemmaTerms.applies(to: name), !model.gemmaTermsAccepted {
                    GemmaTermsView(model: name,
                                   onAgree: model.acceptGemmaTerms,
                                   onCancel: model.cancelDownload)
                } else {
                    downloadConsent(name)
                }
            } else if !model.accepted {
                DisclaimerView(onAgree: model.accept)
            } else if model.downloading {
                downloadProgress
            } else if let err = model.loadError {
                failureView(err)
            } else if model.compiling {
                compileProgress
            } else {
                preparing
            }
        }
        .overlay { hudOverlay }
        .animation(.easeInOut(duration: 0.4), value: model.hud)
        .modifier(OnboardingTitle())
    }

    @ViewBuilder
    private var hudOverlay: some View {
        if let hud = model.hud {
            Text(hud.text)
                .font(hud.prominent ? .title2 : .headline)
                .padding(.horizontal, hud.prominent ? 30 : 22)
                .padding(.vertical, hud.prominent ? 20 : 14)
                .background(.regularMaterial,
                            in: RoundedRectangle(cornerRadius: 14))
                .overlay {
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(.separator, lineWidth: 0.5)
                }
                .transition(.opacity)
                .allowsHitTesting(false)
        }
    }

    @ViewBuilder
    private var centerBar: some View {
        if !actionsExpanded {
            if model.inTurn && isOS {
                Text(ContentView.appName)
                    .appFont(.headline)
                    .lineLimit(1)
                    .transition(.opacity)
            } else {
                modelPicker
            }
        }
    }

    private static let appName: String =
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName")
            as? String)
        ?? (Bundle.main.object(forInfoDictionaryKey: "CFBundleName")
            as? String) ?? "Gadeon"

    private var modelPicker: some View {
        Group {
            if Models.all.count < 2 {
                Button { showModelInvite = true } label: { modelPickerLabel }
                    .buttonStyle(.plain)
            } else {
                Menu {
                    ForEach(Models.all, id: \.self) { name in
                        Button(Models.display(name)) { model.switchModel(name) }
                    }
                } label: { modelPickerLabel }
                .menuIndicator(.hidden)
                .disabled(model.busy)
            }
        }
        .alert("More models on Mac", isPresented: $showModelInvite) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("This iPhone runs \(Models.display(model.modelName)) to stay "
               + "fast and light. On a Mac, Gadeon can run our larger 2B, 4B "
               + "and 9B models that are noticeably smarter, "
               + "with much sharper image understanding.")
        }
    }

    private var modelPickerLabel: some View {
        HStack(spacing: 4) {
            Text(Models.display(model.modelName))
                .appFont(.subheadline)
                .fontWeight(.semibold)
                .lineLimit(1)
            if Models.all.count >= 2 {
                Image(systemName: "chevron.down").appFont(.caption2)
            }
        }
    }

    private func openConversation(_ id: UUID) {
        hideSoftKeyboard()
        actionsExpanded = false
        // Reopening the LIVE conversation read-only would hide the composer
        // on its own chat.
        if !(id == model.currentConversationId && !model.readOnly) {
            model.openConversation(id)
        }
        closeSidebar()
    }

    private func openNewChat() {
        hideSoftKeyboard()
        actionsExpanded = false
        model.newChat()
        closeSidebar()
    }

    private func openSettings() {
        hideSoftKeyboard()
        model.showDebug = false
        model.showSettings = true
        closeSidebar()
    }

    private func openDebug() {
        hideSoftKeyboard()
        model.showSettings = false
        model.showDebug = true
        closeSidebar()
    }

    private func toggleSidebar() {
        if !sidebarOpen { hideSoftKeyboard() }
        withAnimation(.snappy) { sidebarOpen.toggle() }
    }

    private func closeSidebar() {
        withAnimation(.snappy) { sidebarOpen = false }
    }

    // The storage and backup note is required disclosure for an on-device
    // download this size.

    private func downloadConsent(_ name: String) -> some View {
        VStack(spacing: 16) {
            VStack(spacing: 4) {
                Text("Download").appFont(.title2).bold()
                Text(Models.display(name))
                    .appFont(.title3)
                    .foregroundStyle(.secondary)
            }
            Text("This model is about \(model.downloadSizeText).")
            if let est = model.estimatedMinutes(name) {
                Text(est.optimize > 0
                     ? "Estimated ~\(est.download) min download\n"
                       + "~\(est.optimize) min optimize"
                     : "Estimated ~\(est.download) min download")
                    .appFont(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Text("Stored on this device, excluded from iCloud and backup.\n"
               + "It downloads once.\n"
               + "Later launches need no network.")
                .appFont(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            VStack(spacing: 14) {
                Button("Download") { model.confirmDownload() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.defaultAction)
                if model.canCancelDownload {
                    Button("Cancel") { model.cancelDownload() }
                        .controlSize(.large)
                        .keyboardShortcut(.cancelAction)
                }
            }
            .padding(.top, 8)
        }
        .padding(32)
        .frame(maxWidth: 420)
    }

    private var downloadProgress: some View {
        VStack(spacing: 16) {
            ProgressView(value: model.downloadFraction)
                .frame(maxWidth: 320)
            VStack(spacing: 4) {
                Text(model.downloadTotal > 0
                     ? "Downloading" : "Preparing download…")
                    .appFont(.headline)
                if model.downloadTotal > 0 {
                    Text(Models.display(model.modelName))
                        .appFont(.subheadline)
                        .foregroundStyle(.secondary)
                    Text(model.downloadCounter)
                        .appFont(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            if model.downloadTotal > 0 {
                stageWhimsical(.downloading, $downloadingPhrase)
            }
            if let eta = model.downloadETA {
                Text(eta).appFont(.caption).foregroundStyle(.tertiary)
            }
        }
        .padding(32)
    }

    private var compileProgress: some View {
        VStack(spacing: 16) {
            // No LoC denominator means a GGUF load rather than an ANE
            // compile, so a spinner rather than a bar stuck at zero.
            Group {
                if model.compileTotalLoC > 0 {
                    ProgressView(value: model.compileFraction)
                } else {
                    ProgressView()
                }
            }
            .frame(maxWidth: 320)
            if model.firstCompile {
                VStack(spacing: 4) {
                    Text("Optimizing for your device…").appFont(.headline)
                    Text(Models.display(model.modelName))
                        .appFont(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Text("It will take a while.\nLater launches will be fast.")
                    .appFont(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                stageWhimsical(.optimizing, $optimizingPhrase)
            } else {
                VStack(spacing: 4) {
                    Text("Loading…").appFont(.headline)
                    Text(Models.display(model.modelName))
                        .appFont(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            if let eta = model.optimizeETA {
                Text(eta).appFont(.caption).foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 32)
        .padding(.horizontal, 24)
        .frame(maxWidth: 480)
    }

    // The ring holder advances at most every 7s and the read is idempotent,
    // so polling it costs nothing and duplicate tickers are harmless.

    private func stageWhimsical(_ stage: Whimsical.Stage,
                                _ phrase: Binding<String>) -> some View {
        // Every phrase must fit ONE line and never ellipsize, on an
        // iPhone-mini width and under Large Text alike.
        Text(phrase.wrappedValue)
            .appFont(.subheadline)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .minimumScaleFactor(0.4)
            .frame(maxWidth: .infinity)
            .modifier(Shimmer(active: true))
            .task {
                while !Task.isCancelled {
                    phrase.wrappedValue = Whimsical.current(stage)
                    try? await Task.sleep(for: .seconds(1))
                }
            }
    }

    // Some paths reach here without load() having run: accept() flips the
    // disclaimer gate without resolving the model. load() always transitions
    // out, so this fires once rather than spinning.

    private var preparing: some View {
        ProgressView()
            .padding(32)
            .onAppear {
                if !model.downloading, !model.compiling, !model.ready {
                    model.load(name: model.modelName)
                }
            }
    }

    private func failureView(_ message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .appFont(.largeTitle)
                .foregroundStyle(.secondary)
            Text(message)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button("Try Again") { model.retry() }
                .controlSize(.large)
        }
        .padding(32)
        .frame(maxWidth: 420)
    }


    // A diagnostic that deliberately drives a redraw per display refresh over
    // the transcript. Off unless GADEON_STALL=1, and it reports which so a
    // capture cannot be read against an app that was not running it.
    static let stallProbeOn: Bool = {
        let on = ProcessInfo.processInfo.environment["GADEON_STALL"] == "1"
        Instrument.note("[stall] probe \(on ? "ARMED" : "off")")
        return on
    }()

    private var chat: some View {
        transcript
            .simultaneousGesture(pinchZoom,
                                 including: isOS ? .all : .none)
            .modifier(Shimmer(active: ContentView.stallProbeOn))
            // A safe-area inset rather than an overlay: it composes with the
            // keyboard's own inset, so the soft keyboard cannot strand the
            // tail under the composer.
            .safeAreaInset(edge: .bottom, spacing: 0) {
                if !model.readOnly {
                    VStack(spacing: 0) {
                        Composer(model: model, focused: $promptFocused)
                        if model.statusLine {
                            Divider()
                            statusLine
                        }
                    }
                    // The transcript scrolls UNDER this inset and the card is
                    // translucent, so it needs an opaque ground of its own.
                    .background(.bar)
                }
            }
        // The editors refuse drops so this handler owns them, and no file
        // path is ever pasted as text.
        .dropDestination(for: URL.self) { urls, _ in
            model.handleDrop(urls, at: model.caret)
            return true
        } isTargeted: { dropActive = $0 }
        .overlay {
            if dropActive {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color.accentColor, lineWidth: 2)
                    .padding(4)
                    .allowsHitTesting(false)
            }
        }
        .overlay(alignment: .top) { if findActive { findBar } }
        .animation(.easeInOut(duration: 0.2), value: findActive)
    }

    private var findBar: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Find in transcript", text: $findQuery)
                .textFieldStyle(.plain)
                .focused($findFocused)
                .onSubmit { findStep(findController.findNext) }
                .onChange(of: findQuery) { _, q in
                    findCount = findController.find(q)
                    findCurrent = findController.currentMatch
                }
            if findCount > 0 {
                Text("\(findCurrent)/\(findCount)")
                    .appFont(.caption).monospacedDigit()
                    .foregroundStyle(.secondary)
            } else if !findQuery.isEmpty {
                Text("none").appFont(.caption).foregroundStyle(.secondary)
            }
            Button { findStep(findController.findPrevious) } label: {
                Image(systemName: "chevron.up")
            }
            .disabled(findCount == 0)
            Button { findStep(findController.findNext) } label: {
                Image(systemName: "chevron.down")
            }
            .disabled(findCount == 0)
            Button("Done", action: closeFind)
                .keyboardShortcut(.cancelAction)
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.bar)
        .overlay(alignment: .bottom) { Divider() }
        .transition(.move(edge: .top).combined(with: .opacity))
    }

    private func openFind() {
        findController.order = model.messages.map { m in m.id }
        findActive = true
        if !findQuery.isEmpty {
            findCount = findController.find(findQuery)
            findCurrent = findController.currentMatch
        }
        // A tick later, or the field is not yet in the hierarchy to take it.
        Task { @MainActor in findFocused = true }
    }

    private func closeFind() {
        findActive = false
        findController.clear()
        findFocused = false
    }

    private func findStep(_ navigate: () -> Int) {
        findCurrent = navigate()
        if isOS { hideSoftKeyboard() }
    }

    private var samplePills: some View {
        VStack(spacing: 12) {
            if model.accessState != .offline, model.showSample("research") {
                sampleCard(symbol: "books.vertical",
                           title: "Dark Matter Research",
                           subtitle: "Watch it research Wikipedia "
                               + "and the web",
                           thumb: nil,
                           action: model.runResearchSample)
            }
            if model.showSample("calc") {
                if optionDown {
                    sampleCard(symbol: "function",
                               title: "Euler's Formula",
                               subtitle: "Watch it do math with imaginary "
                                   + "numbers",
                               thumb: nil,
                               action: model.runEulerSample)
                } else if model.calcSampleIsInterest {
                    sampleCard(symbol: "percent",
                               title: "Compound Interest",
                               subtitle: "Watch it use the calculator",
                               thumb: nil,
                               action: model.runInterestSample)
                        .help("Try to hold the Option key \u{1F609}")
                } else {
                    sampleCard(symbol: "birthday.cake",
                               title: "Cookie Math",
                               subtitle: "Baker's percentages on the "
                                   + "calculator",
                               thumb: nil,
                               action: model.runCookiesSample)
                        .help("Try to hold the Option key \u{1F609}")
                }
            }
            if model.canAttachImages, model.showSample("picture"),
               ChatModel.samplePictureThumb != nil {
                sampleCard(symbol: "photo",
                           title: "Picture Understanding",
                           subtitle: "It writes a story about this photo",
                           thumb: ChatModel.samplePictureThumb,
                           action: model.runPictureSample)
            }
            if model.canOfferVideoSample, model.showSample("video") {
                sampleCard(symbol: "play.rectangle",
                           title: "Video Understanding",
                           subtitle: "Watch the frames it is looking at",
                           thumb: ChatModel.sampleClipThumb,
                           badge: "play.circle.fill",
                           action: model.runVideoSample)
            }
            if model.canOfferDocumentSample, model.showSample("document") {
                sampleCard(symbol: "doc.richtext",
                           title: "Document Understanding",
                           subtitle: "It reads the table inside a PDF",
                           thumb: nil,
                           action: model.runDocumentSample)
            }
        }
        .padding(24)
        // The transcript stack is leading-aligned, so the cards claim the
        // full width to centre within it.
        .frame(maxWidth: .infinity)
    }

    private func sampleCard(symbol: String, title: String, subtitle: String,
                            thumb: CGImage?, badge: String? = nil,
                            action: @escaping () -> Void)
        -> some View {
        Button(action: action) {
            HStack(spacing: 12) {
                if let thumb {
                    Image(decorative: thumb, scale: 1)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 44, height: 44)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay { badgeMark(badge) }
                } else {
                    Image(systemName: symbol)
                        .appFont(.title2)
                        .frame(width: 44, height: 44)
                        .foregroundStyle(Color.accentColor)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).appFont(.headline)
                    Text(subtitle)
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .padding(14)
            .frame(maxWidth: 340)
            .background(.regularMaterial,
                        in: RoundedRectangle(cornerRadius: 14))
            .overlay {
                RoundedRectangle(cornerRadius: 14)
                    .stroke(.separator, lineWidth: 0.5)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(SampleCardStyle())
    }

    // Points, not the app's text scale: the thumbnail it sits on has a fixed
    // 44pt frame, so a badge that followed the zoom would outgrow it.

    @ViewBuilder
    private func badgeMark(_ symbol: String?) -> some View {
        if let symbol {
            Image(systemName: symbol)
                .font(.system(size: 18))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.55), radius: 2)
        }
    }

    private var statusLine: some View {
        HStack {
            // The phone WRAPS rather than scaling to fit: a shrink-to-fit
            // line fights the zoom setting, and it does not touch the
            // interpolated Image runs, so only half the line would move.
            statusText
                .font(.system(size: statusPoints))
                .foregroundStyle(.secondary)
                .monospacedDigit()
                .lineLimit(isOS ? 2 : 1)
                .minimumScaleFactor(isOS ? 1 : 0.6)
                .modifier(Shimmer(active: model.prefilling))
            Spacer()
        }
        .padding(.leading, 16)
        .padding(.trailing, 12)
        .padding(.vertical, 4)
    }

    // A Text either way, with the symbols interpolated INSIDE the run:
    // lineLimit, minimumScaleFactor and the shimmer then apply to the whole
    // line, which an HStack of views would have broken.
    private var statusText: Text {
        model.statsLabel.isEmpty ? modelSummary : Text(model.statsLabel)
    }

    private var statusPoints: CGFloat {
        (appTextPoints(.footnote) + 1) * model.textScale
    }

    private var statusSymbol: Font {
        .system(size: statusPoints - 2)
    }

    private var modelSummary: Text {
        var out = Text("")
        for tower in model.modelTowers {
            let gap = tower.id == 0 ? "" : " "
            out = out + Text(gap)
                + Text(Image(systemName: tower.symbol)).font(statusSymbol)
                + Text(" \(tower.weight)")
        }
        for fact in model.modelFacts {
            out = out + Text(" \u{00B7} \(fact)")
        }
        return out
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                // NOT a LazyVStack: on iOS a lazy row taller than the viewport
                // blanks once its top scrolls far off-screen, and a long
                // answer is one such row.
                VStack(alignment: .leading, spacing: 12) {
                    let _ = Instrument.beat("transcript")
                    ForEach(model.messages) { m in
                        bubble(m).id(m.id)
                    }
                    // INSIDE the scroll, so the soft keyboard's half of the
                    // screen cannot put a card out of reach.
                    if model.showSamples { samplePills }
                    Color.clear
                        .frame(height: max(tailInset, 1))
                        .id("chat-tail")
                }
                .padding(12)
            }
            .onChange(of: model.busy) { _, busy in
                tailCollapse?.cancel()
                if busy {
                    withAnimation(.easeOut(duration: 0.25)) {
                        tailInset = tailInsetSize
                    }
                } else {
                    tailCollapse = Task { @MainActor in
                        try? await Task.sleep(for: .seconds(1.2))
                        if !Task.isCancelled {
                            withAnimation(.easeOut(duration: 0.35)) {
                                tailInset = 0
                            }
                        }
                    }
                }
            }
            // Toggle follow ONLY on a user-driven scroll, so a container
            // resize or our own animated jump never flips it.
            .onScrollPhaseChange { _, phase, context in
                if phase == .interacting || phase == .decelerating {
                    let g = context.geometry
                    follow = g.contentSize.height - g.visibleRect.maxY < 60
                }
            }
            .onChange(of: model.messages.last?.text) { _, _ in
                if follow { scrollToBottom(proxy) }
            }
            // Re-arms follow, which every other scroll trigger is gated on:
            // the keyboard raising shrinks the viewport, and a scrolled-up
            // transcript would otherwise strand its tail behind the composer.
            .onChange(of: promptFocused) { _, focused in
                if focused {
                    follow = true
                    withAnimation { scrollToBottom(proxy) }
                }
            }
            .onChange(of: model.messages.count) { _, _ in
                follow = true
                findController.order = model.messages.map { m in m.id }
            }
            .onAppear {
                // Align the match's own fraction of the message to the same
                // fraction of the viewport, so the exact line lands in view
                // (a long bubble's match is not left off-screen); nil centers.
                findController.scrollTo = { id, frac in
                    withAnimation {
                        proxy.scrollTo(id,
                            anchor: UnitPoint(x: 0.5, y: frac ?? 0.5))
                    }
                }
                findController.order = model.messages.map { m in m.id }
            }
            .onGeometryChange(for: CGFloat.self, of: { geo in
                geo.size.height
            }, action: { h in
                chatHeight = h
                if follow { scrollToBottom(proxy) }
            })
            .coordinateSpace(.named(transcriptSpace))
            .overlay { toolCallout }
            .overlay(alignment: .bottom) {
                VStack(spacing: 8) {
                    if isReasoningLive { quickAnswerButton }
                    if !follow { toBottomButton(proxy) }
                }
                .padding(.bottom, 10)
            }
        }
    }

    // Looked up rather than copied: a round opened while it is still running
    // has to fill in its result under the pointer.
    private var peekedRound: ChatModel.ToolRound? {
        var out: ChatModel.ToolRound? = nil
        if let id = peek.messageId,
           let m = model.messages.first(where: { msg in msg.id == id }) {
            out = m.toolRounds.first { r in r.id == peek.roundId }
        }
        return out
    }

    private var peekScale: CGFloat {
        min(popScale, 1.5) * model.textScale
    }

    // Quarter-speed on the Dynamic Type curve, so Large Text nudges the
    // monospaced dump rather than ballooning it.
    private var peekTextSize: CGFloat {
        11 * (1 + (popScale - 1) / 4) * model.textScale
    }

    @ViewBuilder
    private var toolCallout: some View {
        if let round = peekedRound {
            GeometryReader { geo in
                let below = peek.anchor.midY < geo.size.height / 2
                let width = min(440 * peekScale + 24, geo.size.width - 24)
                let shape = Callout(
                    tailAt: peek.anchor.minX + 24 - (geo.size.width - width) / 2,
                    tailOnBottom: !below)
                ZStack {
                    // With no pointer to move away, a touch elsewhere is the
                    // only way out.
                    if isOS {
                        Color.black.opacity(0.001)
                            .contentShape(Rectangle())
                            .onTapGesture { peek.close() }
                    }
                    ToolRoundDetail(round: round, size: peekTextSize,
                                    k: peekScale)
                        .padding(below ? .top : .bottom, Callout.tailHeight)
                        .frame(width: width)
                        .background(.regularMaterial, in: shape)
                        .overlay { shape.stroke(.separator, lineWidth: 0.5) }
                        .onGeometryChange(for: CGSize.self, of: { g in
                            g.size
                        }, action: { s in calloutSize = s })
                        .onHover { inside in
                            if inside { peek.keep() } else { peek.fade() }
                        }
                        // Positioned off its own measured height, so it stays
                        // hidden until there is one rather than opening at a
                        // guessed spot and jumping.
                        .opacity(calloutSize.height > 0 ? 1 : 0)
                        .position(x: geo.size.width / 2,
                                  y: calloutY(below, geo.size.height))
                }
            }
        }
    }

    // Clamped into the transcript, so a callout that would run off the top or
    // bottom slides back in rather than being cut.
    private func calloutY(_ below: Bool, _ height: CGFloat) -> CGFloat {
        let h = calloutSize.height
        let want = below ? peek.anchor.maxY + 6 + h / 2
                         : peek.anchor.minY - 6 - h / 2
        return min(max(want, h / 2 + 6), max(height - h / 2 - 6, h / 2 + 6))
    }

    private var isReasoningLive: Bool {
        var live = false
        if model.busy, let last = model.messages.last, !last.fromUser {
            live = last.text.isEmpty && !last.reasoning.isEmpty
        }
        return live
    }

    private var quickAnswerButton: some View {
        Button(action: model.quickAnswer) {
            Label("Quick Answer", systemImage: "bolt.fill")
                .appFont(.caption)
                .fontWeight(.semibold)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(.thinMaterial, in: Capsule())
                .overlay(Capsule().stroke(.separator, lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }

    // The tail spacer, not the last message: while the inset is open, that is
    // what keeps the streaming text lifted above the composer.
    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        proxy.scrollTo("chat-tail", anchor: .bottom)
    }

    private func toBottomButton(_ proxy: ScrollViewProxy) -> some View {
        Button {
            follow = true
            withAnimation { scrollToBottom(proxy) }
        } label: {
            Image(systemName: "arrow.down.circle.fill")
                .appFont(.title)
                .symbolRenderingMode(.hierarchical)
        }
        .buttonStyle(.plain)
    }

    private func bubble(_ m: ChatModel.Message) -> some View {
        HStack {
            if m.fromUser { Spacer(minLength: 40) }
            VStack(alignment: m.fromUser ? .trailing : .leading,
                   spacing: 4) {
                if !m.images.isEmpty {
                    imageStrip(m.images)
                }
                ForEach(m.clips, id: \.self) { url in
                    ClipPlayer(url: url)
                }
                ForEach(m.docs, id: \.self) { doc in
                    DocChip(doc: doc)
                }
                if !m.fromUser, !m.reasoning.isEmpty {
                    ReasoningView(text: m.reasoning,
                                  doc: m.reasoningDoc,
                                  style: ContentView.scaled(
                                      ContentView.reasoningStyle,
                                      by: textScale),
                                  markdown: model.renderMarkdown,
                                  active: isThinking(m),
                                  label: isThinking(m)
                                         ? model.thinkLabel : "Thoughts",
                                  maxHeight: chatHeight / 3)
                }
                if !m.fromUser, !m.toolRounds.isEmpty {
                    ToolCallStrip(messageId: m.id, rounds: m.toolRounds,
                                  peek: peek, live: isLive(m))
                }
                if !m.fromUser, isPrefilling(m) {
                    prefillWhimsical
                } else if let answer = answerText(m) {
                    answerBubble(m, answer)
                } else if isAnswerless(m) {
                    noAnswerNote
                } else if m.loopStopped {
                    loopStopNote
                }
            }
            if !m.fromUser { Spacer(minLength: 40) }
        }
    }

    @ViewBuilder
    private func imageStrip(_ images: [CGImage]) -> some View {
        switch images.count {
        case 1:
            Image(decorative: images[0], scale: 1)
                .resizable()
                .scaledToFit()
                .frame(maxWidth: 300, maxHeight: 260)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        case 2, 3:
            HStack(spacing: 6) {
                ForEach(0..<images.count, id: \.self) { i in
                    stripThumb(images[i], 108)
                }
            }
        case 4:
            VStack(spacing: 6) {
                HStack(spacing: 6) {
                    stripThumb(images[0], 132)
                    stripThumb(images[1], 132)
                }
                HStack(spacing: 6) {
                    stripThumb(images[2], 132)
                    stripThumb(images[3], 132)
                }
            }
        default:
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(0..<images.count, id: \.self) { i in
                        stripThumb(images[i], 96)
                    }
                }
            }
        }
    }

    private func stripThumb(_ cg: CGImage, _ side: CGFloat) -> some View {
        Image(decorative: cg, scale: 1)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: side, height: side)
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // Offered ONLY to the turn it came from: speech follows one answer, and a
    // stale highlight on an earlier bubble would point at the wrong words.
    private func spoken(_ m: ChatModel.Message) -> String? {
        m.id == model.messages.last?.id ? model.speech.spokenText : nil
    }

    private func answerBubble(_ m: ChatModel.Message,
                              _ answer: String) -> some View {
        Group {
            if !m.fromUser && model.renderMarkdown
                && !m.answerDoc.items.isEmpty {
                // The transcript owns the scroll, so scrolls: false.
                MarkdownTextView(m.answerDoc,
                                 style: ContentView.scaled(
                                     ContentView.answerStyle, by: textScale),
                                 find: findController, findId: m.id,
                                 scrolls: false, speaking: spoken(m))
            } else if !m.fromUser {
                // The doc trails the raw text by one ticker beat, so this
                // covers the gap as well as plain-text mode.
                PlainTextView(answer,
                              style: ContentView.scaled(
                                  ContentView.answerStyle, by: textScale),
                              find: findController, findId: m.id,
                              scrolls: false, speaking: spoken(m))
            } else {
                // The ANSWER's body size, not .body, so a question and its
                // reply read the same at every zoom stop on both platforms.
                Text(answer)
                    .font(.system(
                        size: ContentView.answerStyle.bodySize * textScale))
                    .textSelection(.enabled)
            }
        }
        .padding(10)
        .background(m.fromUser ? Color.accentColor.opacity(0.2)
                               : Color.gray.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var prefillWhimsical: some View {
        Text(model.thinkStatus)
            .appFont(.caption)
            .foregroundStyle(.secondary)
            .modifier(Shimmer(active: true))
    }

    // nil while the turn has no content, so a stopped turn shows no bubble
    // and the reasoning header carries the live state.

    private func answerText(_ m: ChatModel.Message) -> String? {
        m.text.isEmpty ? nil : m.text
    }

    // A settled turn that made tool calls and committed no answer. The
    // session commits nothing there by design, so the transcript has to say
    // so rather than show Thoughts over silence.

    private func isAnswerless(_ m: ChatModel.Message) -> Bool {
        !m.fromUser && !m.toolRounds.isEmpty && !isLive(m)
    }

    // Everything transient about a bubble hangs off this. A state that
    // outlives its turn is a stale label, or -- where it drives an
    // animation -- a commit on every display refresh for the rest of the
    // conversation.

    private func isLive(_ m: ChatModel.Message) -> Bool {
        model.busy && m.id == model.messages.last?.id
    }

    private var noAnswerNote: some View {
        Text("The model made tool calls but gave no final answer.")
            .appFont(.caption)
            .italic()
            .foregroundStyle(.secondary)
    }

    private var loopStopNote: some View {
        Text("The model began repeating itself and was stopped before it "
           + "reached an answer. Try rephrasing.")
            .appFont(.caption)
            .italic()
            .foregroundStyle(.secondary)
    }

    private func isPrefilling(_ m: ChatModel.Message) -> Bool {
        isLive(m) && model.prefilling
    }

    // `isLive` and not "text is empty" alone: an interrupted turn keeps an
    // empty answer, so the marquee would never stop.

    private func isThinking(_ m: ChatModel.Message) -> Bool {
        isLive(m) && m.text.isEmpty
    }

}

// Built on APPEAR, never at init: a bubble is re-evaluated on every streamed
// token, and an AVPlayer per evaluation would be a decoder per token. A
// restored path can be stale, so the file is tested once when the row
// appears and named rather than offered when it has gone.

private struct ClipPlayer: View {

    let url: URL
    @State private var player: AVPlayer?
    @State private var playable = true

    var body: some View {
        Group {
            if playable {
                ClipSurface(player: player)
                    .frame(maxWidth: 300, minHeight: 170, maxHeight: 260)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                HStack(spacing: 6) {
                    Image(systemName: "film")
                    Text("Video: \(url.lastPathComponent)")
                        .lineLimit(1).truncationMode(.middle)
                }
                .appFont(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .onAppear {
            playable = FileManager.default.fileExists(atPath: url.path)
            if playable, player == nil { player = AVPlayer(url: url) }
        }
        .onDisappear { player?.pause() }
    }
}

private struct SampleCardStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .opacity(configuration.isPressed ? 0.55 : 1)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.12),
                       value: configuration.isPressed)
    }
}

private struct ReasoningView: View {
    let text: String
    let doc: Markdown.Document
    let style: MarkdownStyle
    let markdown: Bool
    let active: Bool
    let label: String
    let maxHeight: CGFloat
    @State private var expanded = false
    @State private var innerFollow = true

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            header
            if expanded {
                reasoning
            }
        }
    }

    // The ticker is a SIBLING of the Button, never inside its label: a
    // GeometryReader in a label swallows the leftover width (which can go
    // negative before layout settles), and a TimelineView there may skip
    // scheduling.

    private var header: some View {
        HStack(spacing: 6) {
            Button {
                expanded.toggle()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: expanded
                          ? "chevron.down" : "chevron.right")
                        .appFont(.caption2)
                        .foregroundStyle(.secondary)
                    Text(label)
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .fixedSize()
            if active && !expanded {
                ThinkingTicker(text: text)
            }
        }
    }

    private var reasoning: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Group {
                    if markdown && !doc.items.isEmpty {
                        MarkdownView(doc, style: style)
                    } else {
                        Text(text)
                            .appFont(.callout)
                            .foregroundStyle(Color(white: 0.53))
                            .textSelection(.enabled)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .id("reason")
            }
            .frame(maxHeight: maxHeight)
            .onScrollPhaseChange { _, phase, context in
                if phase == .interacting || phase == .decelerating {
                    let g = context.geometry
                    let d = g.contentSize.height - g.visibleRect.maxY
                    innerFollow = d < 40
                }
            }
            .onChange(of: text) { _, _ in
                if innerFollow { proxy.scrollTo("reason", anchor: .bottom) }
            }
        }
    }
}

// A tool row reports its frame in this space and the callout is placed in the
// same one, so the two agree while the transcript scrolls under them.
private let transcriptSpace = "transcript"

// ONE owner, because the row and the callout both open and close it: a
// pointer travelling from one to the other leaves both, and two independent
// timers would race to dismiss what the user is reaching for.
@MainActor @Observable final class ToolPeek {
    private(set) var messageId: UUID?
    private(set) var roundId = 0
    private(set) var anchor: CGRect = .zero
    @ObservationIgnored private var dismiss: Task<Void, Never>?

    func showing(_ message: UUID, _ round: Int) -> Bool {
        messageId == message && roundId == round
    }

    func show(_ message: UUID, _ round: Int, at frame: CGRect) {
        dismiss?.cancel()
        messageId = message
        roundId = round
        anchor = frame
    }

    func toggle(_ message: UUID, _ round: Int, at frame: CGRect) {
        if showing(message, round) {
            close()
        } else {
            show(message, round, at: frame)
        }
    }

    func track(_ message: UUID, _ round: Int, at frame: CGRect) {
        if showing(message, round) { anchor = frame }
    }

    func keep() { dismiss?.cancel() }

    func close() {
        dismiss?.cancel()
        messageId = nil
    }

    // The gap between the row and its callout is not a decision to dismiss,
    // so leaving either only starts a grace period.
    func fade() {
        dismiss?.cancel()
        dismiss = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(300))
            if !Task.isCancelled { messageId = nil }
        }
    }
}

private struct Callout: Shape {
    let tailAt: CGFloat
    let tailOnBottom: Bool
    private static let radius: CGFloat = 12
    private static let tail: CGFloat = 9

    static var tailHeight: CGFloat { tail }

    func path(in rect: CGRect) -> Path {
        var p = Path()
        let body = CGRect(x: rect.minX,
                          y: rect.minY + (tailOnBottom ? 0 : Callout.tail),
                          width: rect.width,
                          height: rect.height - Callout.tail)
        p.addRoundedRect(in: body, cornerSize: CGSize(width: Callout.radius,
                                                      height: Callout.radius))
        let span = Callout.radius + Callout.tail
        let x = min(max(tailAt, body.minX + span), body.maxX - span)
        let edge = tailOnBottom ? body.maxY : body.minY
        let tip = tailOnBottom ? rect.maxY : rect.minY
        p.move(to: CGPoint(x: x - Callout.tail, y: edge))
        p.addLine(to: CGPoint(x: x, y: tip))
        p.addLine(to: CGPoint(x: x + Callout.tail, y: edge))
        p.closeSubpath()
        return p
    }
}

private struct ToolCallStrip: View {
    let messageId: UUID
    let rounds: [ChatModel.ToolRound]
    let peek: ToolPeek
    // A stopped or failed turn leaves a round's result nil for good, so the
    // shimmer needs this as well: an indefinite animation drives a view-graph
    // update and a synchronous render-server round trip on every display
    // refresh, for the rest of the conversation.
    let live: Bool
    @State private var frames: [Int: CGRect] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Image(systemName: "wrench.and.screwdriver")
                    .appFont(.caption2)
                Text("Using tools")
            }
            .appFont(.caption)
            .foregroundStyle(.secondary)
            ForEach(rounds) { round in row(round) }
        }
    }

    private func row(_ round: ChatModel.ToolRound) -> some View {
        HStack(spacing: 6) {
            Text(round.id == rounds.last?.id ? "└─" : "├─")
                .appFont(.caption).monospaced()
            Image(systemName: round.symbol)
                .appFont(.caption2)
                .foregroundStyle(errored(round) ? Color.orange
                                                : Color.secondary)
            Text(round.label)
                .appFont(.caption)
            if !round.args.isEmpty {
                Text(round.args)
                    .appFont(.caption2)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .foregroundStyle(.secondary)
        .modifier(Shimmer(active: live && round.result == nil))
        .contentShape(Rectangle())
        .onGeometryChange(for: CGRect.self, of: { geo in
            geo.frame(in: .named(transcriptSpace))
        }, action: { frame in
            frames[round.id] = frame
            peek.track(messageId, round.id, at: frame)
        })
        .onHover { inside in
            if inside {
                peek.show(messageId, round.id,
                          at: frames[round.id] ?? .zero)
            } else {
                peek.fade()
            }
        }
        .onTapGesture {
            peek.toggle(messageId, round.id, at: frames[round.id] ?? .zero)
        }
    }

    private func errored(_ round: ChatModel.ToolRound) -> Bool {
        round.result?.hasPrefix("error") == true
    }
}

private struct ToolRoundDetail: View {
    let round: ChatModel.ToolRound
    let size: CGFloat
    let k: CGFloat
    @State private var contentHeight: CGFloat = 20

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: round.symbol)
                Text(round.emitted).bold()
                if !round.emitted.isEmpty && round.label != round.emitted {
                    Text("-> \(round.label)")
                        .foregroundStyle(.secondary)
                }
            }
            .font(.system(size: size))
            if !round.args.isEmpty {
                Text(round.args)
                    .font(.system(size: size, design: .monospaced))
                    .textSelection(.enabled)
            }
            ScrollView {
                Text(round.result ?? "running")
                    .font(.system(size: size, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .onGeometryChange(for: CGFloat.self, of: { geo in
                        geo.size.height
                    }, action: { h in contentHeight = h })
            }
            .frame(minWidth: 260 * k, idealWidth: 440 * k, maxWidth: 440 * k)
            .frame(height: min(max(contentHeight, 20), 280 * k))
        }
        .padding(12)
    }
}

private struct ThinkingTicker: View {
    let text: String

    var body: some View {
        // No fixedSize: it would pin the Text to its full intrinsic width and
        // spill past the row.
        Text(String(text.suffix(120)))
            .appFont(.caption)
            .foregroundStyle(Color(white: 0.53))
            .lineLimit(1)
            .truncationMode(.head)
            .frame(maxWidth: .infinity, alignment: .trailing)
            .mask(LinearGradient(
                stops: [.init(color: .clear, location: 0),
                        .init(color: .black, location: 0.12)],
                startPoint: .leading, endPoint: .trailing))
            .padding(.trailing, 8)
    }
}

// Explicit per-frame opacity, never an implicit repeatForever animation: an
// in-flight implicit animation bleeds into unrelated changes on the same
// view, cross-fading text into overlapping glyphs.

private struct Shimmer: ViewModifier {
    let active: Bool

    func body(content: Content) -> some View {
        if active {
            TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { ctx in
                let t = ctx.date.timeIntervalSinceReferenceDate
                content.opacity(0.85 + 0.15 * sin(t * 2 * .pi / 3.6))
            }
        } else {
            content
        }
    }
}

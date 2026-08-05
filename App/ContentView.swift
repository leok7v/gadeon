import MD
import SwiftUI

struct ContentView: View {

    @Bindable var model: ChatModel

    // Markdown styles per channel (MD/INTEGRATION.md), based per platform:
    // the user bubble is the system .body (17pt iOS/iPadOS, 13pt macOS),
    // and the answer must read the SAME size as the question -- a base
    // hardcoded to one platform renders smaller than the question on the
    // other. macOS keeps the touch-above 14; chat-scaled headings (the
    // default 28pt h1 shouts in a bubble) ride body-relative increments;
    // reasoning sits smaller in the disclosure grey. Code sits 2pt below
    // each body: monospaced reads optically larger at equal point size.
    static let answerStyle = MarkdownStyle(
        bodySize: isOS ? 17 : 14,
        codeSize: isOS ? 15 : 12,
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
    // Dynamic Type: MarkdownStyle point sizes are FIXED (the MD renderer
    // builds systemFont(ofSize:), which the content-size category never
    // touches), so the transcript would freeze while every system-styled
    // label around it scales -- a mixed soup under Larger Text. Scale the
    // style by the body text style's own curve so both move together.
    @ScaledMetric(relativeTo: .body) private var typeScale: CGFloat = 1

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
    // Plain State, not FocusState: the AppKit prompt editor owns
    // first-responder and syncs this flag both ways, which a FocusState
    // binding can't cross into an NSViewRepresentable.
    @State private var promptFocused = false
    // Live transcript height, so a reasoning disclosure can cap itself at a
    // third of it and scroll the rest.
    @State private var chatHeight: CGFloat = 0
    // The tool-row detail is drawn INSIDE the transcript, not in a system
    // popover: an NSPopover is its own window, free to hang past the app's
    // frame, and a screenshot or a screen recording of the window cuts off
    // whatever hung out. Everything shown about a turn has to be inside the
    // picture of that turn.
    @State private var peek = ToolPeek()
    @State private var calloutSize: CGSize = .zero
    // The callout's monospaced dump scales with Dynamic Type; grow the frame
    // along, capped, so Large Text cannot push it past the transcript.
    @ScaledMetric(relativeTo: .body) private var popScale: CGFloat = 1
    // Autoscroll follows the newest tokens until the user scrolls up; then it
    // stops and a jump-to-bottom button appears until they opt back in.
    @State private var follow = true
    // Breathing room under the last bubble while a reply streams, so the live
    // text is not pinned against the composer edge (where Quick Answer and
    // the jump-to-bottom button also sit); collapses a beat after the turn
    // ends. The spacer is the autoscroll target, so following keeps the tail
    // lifted by the inset.
    @State private var tailInset: CGFloat = 0
    @State private var tailCollapse: Task<Void, Never>?
    @ScaledMetric(relativeTo: .body) private var tailInsetSize: CGFloat = 64
    // Option held (macOS): swaps the calculator sample for the Euler one.
    // Always false on iOS (no modifier key).
    @State private var optionDown = false
    @State private var showModelInvite = false
    // Highlights the whole chat area while a file is dragged over it.
    @State private var dropActive = false
    // The optimizing / downloading screens' cycling whimsical phrases,
    // seeded from the ring holder (an idempotent read) so no placeholder
    // word ever renders and a view re-init cannot advance the rotation.
    @State private var optimizingPhrase = Whimsical.current(.optimizing)
    @State private var downloadingPhrase = Whimsical.current(.downloading)
    // The menu sidebar slides in as an overlay drawer, toggled by the top bar's
    // folders button. A custom top bar + drawer, not a NavigationSplitView, so
    // iOS injects no back chevron and no grouped toolbar pill, and macOS keeps
    // its native title-bar toolbar.
    @State private var sidebarOpen = false
    // Find in Chat: an inline bar over the transcript driving Find across the
    // per-message bubbles (each registers with this controller), so matches
    // scroll into view and highlight in the real transcript -- no separate view.
    @State private var findActive = false
    @State private var findController = MarkdownFindController()
    @State private var findQuery = ""
    @State private var findCount = 0
    @State private var findCurrent = 0
    @FocusState private var findFocused: Bool
    // I1 PROTOTYPE (uncommitted): the trailing tools cluster collapses to a
    // single chevron and expands on demand, auto-collapsing after an idle beat
    // (macOS pointer only). Borrowed from md.too's ContentView-macOS.
    @State private var actionsExpanded = false
    @State private var actionsHovering = false

    var body: some View {
        content
            .modifier(OptionKeyMonitor(down: $optionDown))
            .preferredColorScheme(model.theme.scheme)
            // A returning user who cleared both onboarding gates resolves the
            // model on appear (build if on disk, else its download). A first run
            // instead starts the fetch at EULA acceptance (acceptEULA), so it is
            // not loaded here. A too-small device downloads nothing.
            .onAppear {
                if Models.supported, model.eulaAccepted, model.accepted {
                    model.load(name: model.modelName)
                }
            }
    }

    // Settings and Debug are hosted IN the content area (not a sheet / cover),
    // so they get the full window width and read as part of the app rather than
    // a modal panel floating over the chat; each closes with its own Done. The
    // chat (top bar + transcript + composer) exists only once the model is
    // ready; onboarding, download, and the one-time Optimizing compile render
    // full-screen with no top bar.
    @ViewBuilder
    private var content: some View {
        if model.showSettings {
            SettingsView(model: model,
                         onClose: { model.showSettings = false })
        } else if model.showDebug {
            DebugView(model: model, onClose: { model.showDebug = false })
        // A pending or in-flight download takes over the screen even when a
        // model is already loaded: choosing an un-downloaded model from the
        // picker sets downloadName, which must surface its consent panel (then
        // progress) -- otherwise the pick silently sets state behind the live
        // chat and the app appears stuck on the current model.
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
        // Global New Chat (Cmd+N), reachable even with the drawer open -- the
        // top-bar button is disabled then. Hidden; only the shortcut matters.
        .background {
            Button("", action: openNewChat)
                .keyboardShortcut("n", modifiers: .command)
                .disabled(model.busy)
                .hidden()
        }
    }

    @Environment(\.horizontalSizeClass) private var sizeClass

    // nil on a regular layout, so the title bar keeps INHERITING the
    // toolbar's own size. Naming a number there is what shrank the iPad: the
    // system default is larger than any figure that looked right written
    // down, and an override applies everywhere it is not needed.
    private var barGlyph: Font? {
        sizeClass == .compact ? .system(size: 20) : nil
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

    // Trailing title-bar cluster (I1 PROTOTYPE): New Chat is always present;
    // the transcript-tools menu hides behind a chevron and reveals on tap,
    // auto-collapsing after an idle beat so the title bar stays quiet. The
    // chevron only appears once there is a transcript to act on.
    // The tools live in the title bar: collapsed to a chevron, expanded to the
    // action row. Expanding hides the centered model picker (centerBar) so the
    // row takes over the bar with no reflow, rather than shoving the picker.
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
        // The title-bar glyphs are the smallest targets in the app and the
        // ones furthest from the holding hand; a narrow screen needs them
        // bigger for the same reason the composer's controls do.
        .font(barGlyph)
        .onHover { inside in actionsHovering = inside }
        .animation(.easeInOut(duration: 0.2), value: actionsExpanded)
        .task(id: actionsIdle) { await collapseActionsAfterIdle() }
    }

    // The trace is offered only where the status line is: both are the
    // diagnostic surface, so one switch decides whether the app shows its
    // workings at all. nil hides the button.
    private var debugAction: (() -> Void)? {
        var out: (() -> Void)? = nil
        if model.statusLine { out = openDebug }
        return out
    }

    // Expanded and the pointer has left the cluster (macOS only -- iOS has no
    // hover, so its cluster stays open until tapped shut).
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
                .frame(width: 300)
                .frame(maxHeight: .infinity)
                .background(.bar)
                .transition(.move(edge: .leading))
                .zIndex(2)
        }
    }

    // Everything before a chat exists: the onboarding gates, download consent +
    // progress, load failure, and the one-time Optimizing compile. Full-screen.
    private var onboarding: some View {
        Group {
            if !Models.supported {
                unsupportedView
            } else if !model.eulaAccepted {
                EULAView(onAgree: model.acceptEULA)
            } else if let name = model.downloadName {
                // Explicit download consent BEFORE the disclaimer (App Store
                // requires approval; the fetch starts on Download, then overlaps
                // the disclaimer).
                //
                // A model under a third-party licence asks for that first: the
                // size question is ours to put, the licence is not.
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

    // Centered, non-interactive flash confirming a per-turn toggle (Thinking /
    // access tier); dissolves after ~3s (ChatModel.hud).
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

    // The title-bar model picker (moved from the composer): "Gadeon: <label>",
    // tap to open the model menu; a single-model phone taps to the "more on
    // Mac" note instead of a dead menu. Cosmetic label; the internal id is
    // unchanged.
    // The centered title-bar item, hidden while the tools are expanded so the
    // action row takes over the bar without shoving it.
    //
    // The app's own name while a turn is being worked, the model picker
    // between turns. A recording is made of the working stretch, which is
    // where the name earns its place; the moment the user has something to
    // decide, the control they might want is the one that belongs there --
    // and the picker carries the only route to the more-models invite.
    //
    // iOS ONLY. A Mac already shows the name in the menu bar and the window
    // title, so swapping it in here says nothing twice over and costs the
    // picker its place.
    @ViewBuilder
    private var centerBar: some View {
        if !actionsExpanded {
            if model.inTurn && isOS {
                Text(ContentView.appName)
                    .font(.headline)
                    .lineLimit(1)
                    .transition(.opacity)
            } else {
                modelPicker
            }
        }
    }

    // From the bundle, so it cannot drift from what the installer shows.
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
            // Names the model this device actually got: the invite fires only
            // where exactly one is offered, and which one that is varies by
            // device (a 3 GB iPhone gets the GPU-run Bonsai 1.7B, not the
            // 0.8B, whose graphs its Neural Engine cannot compile).
            Text("This iPhone runs \(Models.display(model.modelName)) to stay "
               + "fast and light. On a Mac, Gadeon can run our larger 2B, 4B "
               + "and 9B models that are noticeably smarter, "
               + "with much sharper image understanding.")
        }
    }

    private var modelPickerLabel: some View {
        HStack(spacing: 4) {
            // The model's name alone: the app's is already in the macOS menu
            // bar, and on the phone there is no width to spend on it either.
            // Smaller than the title so a long name fits.
            Text(Models.display(model.modelName))
                .font(.subheadline)
                .fontWeight(.semibold)
                .lineLimit(1)
            if Models.all.count >= 2 {
                Image(systemName: "chevron.down").font(.caption2)
            }
        }
    }

    // Sidebar routing: open a saved conversation / Settings / debug in the
    // detail and close the drawer behind it. Settings and debug reuse the
    // existing in-body flags.
    private func openConversation(_ id: UUID) {
        hideSoftKeyboard()
        actionsExpanded = false
        // Tapping the LIVE session just closes the drawer -- do not reopen it
        // as a read-only copy (which would hide the composer on its own chat).
        if !(id == model.currentConversationId && !model.readOnly) {
            model.openConversation(id)
        }
        closeSidebar()
    }

    // Start a fresh conversation from the sidebar (or ⌘N) and close the drawer
    // -- the same "pick a destination" gesture as opening an existing one.
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

    // Resign first responder on open so the soft keyboard does not linger
    // beside the drawer.
    private func toggleSidebar() {
        if !sidebarOpen { hideSoftKeyboard() }
        withAnimation(.snappy) { sidebarOpen.toggle() }
    }

    private func closeSidebar() {
        withAnimation(.snappy) { sidebarOpen = false }
    }

    // Shown on a device with too little RAM for even the 0.8B (iOS <=3GB): a
    // clear notice, and nothing is downloaded.

    private var unsupportedView: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("Device not supported").font(.title2).bold()
            Text("This device does not have enough memory to run the on-device "
               + "model. At least 4 GB of RAM is required.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding(32)
        .frame(maxWidth: 420)
    }

    // Model not on disk yet: ask before spending the bytes. The storage +
    // backup note is required disclosure for a multi-hundred-MB on-device
    // download.

    private func downloadConsent(_ name: String) -> some View {
        VStack(spacing: 16) {
            VStack(spacing: 4) {
                Text("Download").font(.title2).bold()
                Text(name)
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
            Text("This model is about \(model.downloadSizeText).")
            if let est = model.estimatedMinutes(name) {
                Text(est.optimize > 0
                     ? "Estimated ~\(est.download) min download\n"
                       + "~\(est.optimize) min optimize"
                     : "Estimated ~\(est.download) min download")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Text("Stored on this device, excluded from iCloud and backup.\n"
               + "It downloads once.\n"
               + "Later launches need no network.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            // Stacked, not side by side, so "Cancel Download" does not read as
            // one phrase. Cancel appears whenever there is a set to fall back to
            // -- a loaded model OR a downloaded one on disk (e.g. a persisted 2B
            // choice at startup returns to the on-disk 0.8B). Only the first-run
            // download, with nothing on disk, is uncancellable.
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
                    .font(.headline)
                if model.downloadTotal > 0 {
                    Text(model.modelName)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Text(model.downloadCounter)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            if model.downloadTotal > 0 {
                stageWhimsical(.downloading, $downloadingPhrase)
            }
            if let eta = model.downloadETA {
                Text(eta).font(.caption).foregroundStyle(.tertiary)
            }
        }
        .padding(32)
    }

    // Blocking model-preparation step. On the FIRST compile of a set (the slow
    // one-time ANE build) it explains itself; once the set is already compiled
    // a later launch just shows "Loading" and the bar, no explanation. The bar
    // is driven by real per-program model.mil LoC, so it tracks actual
    // progress.

    private var compileProgress: some View {
        VStack(spacing: 16) {
            // No LoC denominator (a GGUF load, not an ANE compile) -> an
            // indeterminate spinner instead of a bar stuck at zero.
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
                    Text("Optimizing for your device…").font(.headline)
                    Text(model.modelName)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Text("It will take a while.\nLater launches will be fast.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                stageWhimsical(.optimizing, $optimizingPhrase)
            } else {
                VStack(spacing: 4) {
                    Text("Loading…").font(.headline)
                    Text(model.modelName)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            if let eta = model.optimizeETA {
                Text(eta).font(.caption).foregroundStyle(.tertiary)
            }
        }
        // Wider than the other panels and slimmer side padding: the one-line
        // whimsical phrase needs the horizontal room (it ellipsized at 420).
        .padding(.vertical, 32)
        .padding(.horizontal, 24)
        .frame(maxWidth: 480)
    }

    // A cycling stage phrase (optimizing / downloading), shimmering like the
    // transcript's working line but at a full-screen tempo -- the ring
    // holder advances at most every 7s; this task just polls it (an
    // idempotent read, so duplicate tickers are harmless). The .task lives
    // on the Text, so it runs only while its screen is up.

    private func stageWhimsical(_ stage: Whimsical.Stage,
                                _ phrase: Binding<String>) -> some View {
        // A comfortable base size with a DEEP scale floor: every phrase must
        // fit ONE line, never ellipsize, on an iPhone-mini width AND under
        // Large Text on an iPad. The floor makes that per-phrase: short
        // phrases render full subheadline everywhere; only the longest
        // (~64 chars) shrink, and only on narrow screens.
        Text(phrase.wrappedValue)
            .font(.subheadline)
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

    // The neutral gap between appearing and load() deciding download-vs-build.
    // Self-heal: some paths reach here without load() having run -- notably a
    // restart that accepts the disclaimer AFTER a download was killed (accept()
    // flips the gate but does not resolve the model) -- so resolve it here
    // rather than spin forever. load() always transitions out (consent, build,
    // or error), so this fires once.

    private var preparing: some View {
        ProgressView()
            .padding(32)
            .onAppear {
                if !model.downloading, !model.compiling, !model.ready {
                    model.load(name: model.modelName)
                }
            }
    }

    // The set downloaded but could not be built on this device (e.g. an older
    // Neural Engine rejects the trunk compile). Clear message + Retry, never a
    // silent spinner.

    private func failureView(_ message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
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


    // A DIAGNOSTIC, and it deliberately reintroduces the defect this app was
    // freezing on. Off unless GADEON_STALL is set in the environment.
    //
    // It reuses `Shimmer` rather than rolling an animation, because the KIND
    // of animation is the whole thing. `withAnimation(.repeatForever)` on a
    // layer property -- opacity, scale -- is handed to the render server once
    // and interpolated there, so it drives no per-frame work in the app at
    // all. `TimelineView(.animation)` re-evaluates and REDRAWS its content
    // every tick, which is what turns each display refresh into a view-graph
    // update, a commit, and a synchronous round trip.
    //
    // Applied over the TRANSCRIPT so the redraw covers the tall single
    // surface that is the suspected cost, which is the pairing the original
    // freeze had: something committing per refresh, over something expensive
    // to commit.
    // Says so in the diagnostics, because two captures were taken against an
    // app that turned out not to be running it and nothing on screen or in
    // the log could settle which. An instrument that cannot prove it was
    // armed is not an instrument.
    static let stallProbeOn: Bool = {
        let on = ProcessInfo.processInfo.environment["GADEON_STALL"] == "1"
        Instrument.note("[stall] probe \(on ? "ARMED" : "off")")
        return on
    }()

    private var chat: some View {
        transcript
            .modifier(Shimmer(active: ContentView.stallProbeOn))
            .overlay { if model.showSamples { samplePills } }
            // The composer rides as a bottom safe-area inset, so the transcript
            // insets its scrollable content beneath it -- the last bubble always
            // scrolls clear of the composer, and the inset composes with the
            // keyboard safe area, so the soft keyboard never strands the tail
            // under the composer. A reopened (read-only) chat shows NO composer:
            // the missing input, plus New Chat in the top bar, is the whole
            // story, so there is no bottom bar to explain it.
            .safeAreaInset(edge: .bottom, spacing: 0) {
                if !model.readOnly {
                    VStack(spacing: 0) {
                        Composer(model: model, focused: $promptFocused)
                        if model.statusLine {
                            Divider()
                            statusLine
                        }
                    }
                    // Opaque: the transcript scrolls UNDER this inset, and the
                    // composer card is translucent, so without it a scrolled
                    // bubble reads straight through the prompt field.
                    .background(.bar)
                }
            }
        // Drop images / .txt / .md anywhere over the conversation; each becomes
        // an inline @reference at the caret. The editors refuse drops so this
        // handler owns them, with no file-path paste.
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

    // Find bar pinned to the transcript top. It drives Find across the
    // per-message bubbles through findController (wired to the ScrollViewReader
    // below), so each match scrolls into view and highlights in the real
    // bubble -- no separate view.
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
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            } else if !findQuery.isEmpty {
                Text("none").font(.caption).foregroundStyle(.secondary)
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
        // Focus after the bar materializes, or the field is not yet in the
        // hierarchy to receive it.
        Task { @MainActor in findFocused = true }
    }

    private func closeFind() {
        findActive = false
        findController.clear()
        findFocused = false
    }

    // Navigate to a match and, on iOS, drop the keyboard so the scrolled-to
    // line is not hidden behind it (the find bar itself stays visible).
    private func findStep(_ navigate: () -> Int) {
        findCurrent = navigate()
        if isOS { hideSoftKeyboard() }
    }

    // Fresh-conversation demo pills: equal cards inviting a first
    // message, gone the moment a conversation exists (and back after New
    // Chat). The picture card shows the actual bundled photo -- a generic
    // glyph would leave the user guessing what got sent. The research
    // pill hides in airplane mode (a tool demo without tools demos the
    // wrong thing); the picture pill needs a vision-capable model. The
    // calculator pill runs LOCAL, so it demos even offline: compound
    // interest by default (well within even the 0.8B's reach -- Euler
    // math sends the phone model rambling), and holding Option swaps in
    // the Euler's Formula walk for nerds; the tooltip winks at it.

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
        }
        .padding(24)
    }

    private func sampleCard(symbol: String, title: String, subtitle: String,
                            thumb: CGImage?, action: @escaping () -> Void)
        -> some View {
        Button(action: action) {
            HStack(spacing: 12) {
                if let thumb {
                    Image(decorative: thumb, scale: 1)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 44, height: 44)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    Image(systemName: symbol)
                        .font(.title2)
                        .frame(width: 44, height: 44)
                        .foregroundStyle(Color.accentColor)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline)
                    Text(subtitle)
                        .font(.caption)
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
        .buttonStyle(.plain)
    }

    // The status line under the prompt: the turn's numbers, or the working
    // phrase while a prefill runs, or load status.

    private var statusLine: some View {
        HStack {
            // caption2 on iOS + a scale floor: the line must FIT rather than
            // truncate -- the t/s term trails, and it is the one being read.
            Text(statusText)
                .font(isOS ? .caption2 : .caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
                .lineLimit(1)
                .minimumScaleFactor(0.6)
                .modifier(Shimmer(active: model.prefilling))
            Spacer()
        }
        .padding(.leading, 16)
        .padding(.trailing, 12)
        .padding(.vertical, 4)
    }

    // Numbers whenever a turn has produced any, and what the model IS before
    // that. No playful phrase: this line is where someone looks for facts,
    // and a quip here competes with the transcript's own -- which is the
    // surface written for it. Shimmer (driven by `prefilling`) is what says
    // "working" from then on.

    private var statusText: String {
        model.statsLabel.isEmpty ? model.modelSummary : model.statsLabel
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                // A plain VStack, not LazyVStack: on iOS a lazy row taller than
                // the viewport blanks once its top scrolls far off-screen, and a
                // long answer is one such row. A conversation is bounded, so
                // laying every bubble out eagerly is affordable.
                VStack(alignment: .leading, spacing: 12) {
                    let _ = Instrument.beat("transcript")
                    ForEach(model.messages) { m in
                        bubble(m).id(m.id)
                    }
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
            // resize (composer growing on a paste) or our own animated jump
            // never flips it -- deriving follow from raw geometry left it
            // stuck off.
            .onScrollPhaseChange { _, phase, context in
                if phase == .interacting || phase == .decelerating {
                    let g = context.geometry
                    follow = g.contentSize.height - g.visibleRect.maxY < 60
                }
            }
            .onChange(of: model.messages.last?.text) { _, _ in
                if follow { scrollToBottom(proxy) }
            }
            // Taking the composer means the user is about to write, so re-arm
            // follow and bring the newest turn back. Without it, the keyboard
            // raising shrinks the viewport while a scrolled-up transcript
            // stays put, and whatever was above the composer ends up behind
            // it -- the height change alone cannot fix that, since every other
            // scroll trigger is gated on `follow`, which is false for exactly
            // the reader this hits.
            .onChange(of: promptFocused) { _, focused in
                if focused {
                    follow = true
                    withAnimation { scrollToBottom(proxy) }
                }
            }
            // A new turn (send / New Chat) re-arms follow, so sending always
            // snaps to the bottom even if the user had scrolled up. The find
            // controller's message order tracks the transcript.
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
            // Transcript height changes only when the composer grows or the
            // window resizes; re-pin to the bottom then (if following), so a
            // long paste does not push the latest turn under the composer.
            .onGeometryChange(for: CGFloat.self, of: { geo in
                geo.size.height
            }, action: { h in
                chatHeight = h
                if follow { scrollToBottom(proxy) }
            })
            .coordinateSpace(.named(transcriptSpace))
            .overlay { toolCallout }
            // Quick Answer (while reasoning) and scroll-to-bottom (while
            // scrolled up) share this bottom spot; stacked so both stay usable
            // if they happen to show at once.
            .overlay(alignment: .bottom) {
                VStack(spacing: 8) {
                    if isReasoningLive { quickAnswerButton }
                    if !follow { toBottomButton(proxy) }
                }
                .padding(.bottom, 10)
            }
        }
    }

    // The peeked row's CURRENT state, looked up rather than copied: a round
    // opened while it is still running has to fill in its result under the
    // pointer.
    private var peekedRound: ChatModel.ToolRound? {
        var out: ChatModel.ToolRound? = nil
        if let id = peek.messageId,
           let m = model.messages.first(where: { msg in msg.id == id }) {
            out = m.toolRounds.first { r in r.id == peek.roundId }
        }
        return out
    }

    private var peekScale: CGFloat { min(popScale, 1.5) }

    // A raw-data debug view wants DENSITY: one step below caption, scaling at
    // quarter speed of the Dynamic Type curve so Large Text nudges it instead
    // of ballooning the monospaced dump.
    private var peekTextSize: CGFloat { 11 * (1 + (popScale - 1) / 4) }

    // Centered across the transcript, so the panel is whole inside it at any
    // window size, with the tail naming the row it came from. It sits below a
    // row in the upper half and above one in the lower half, which keeps the
    // row itself out from under it.

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
                    // No pointer to move away, so a touch anywhere else is
                    // the only way out; a mouse leaving the callout already
                    // dismisses it and a scrim would only block the rows.
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
                        // Placed off its own measured height, so it draws
                        // only once there is one -- otherwise the first open
                        // lands at a guessed spot and jumps.
                        .opacity(calloutSize.height > 0 ? 1 : 0)
                        .position(x: geo.size.width / 2,
                                  y: calloutY(below, geo.size.height))
                }
            }
        }
    }

    // Clamped into the transcript, which is the whole point of hosting it
    // here: a callout that would run off the top or bottom slides back in
    // rather than being cut.
    private func calloutY(_ below: Bool, _ height: CGFloat) -> CGFloat {
        let h = calloutSize.height
        let want = below ? peek.anchor.maxY + 6 + h / 2
                         : peek.anchor.minY - 6 - h / 2
        return min(max(want, h / 2 + 6), max(height - h / 2 - 6, h / 2 + 6))
    }

    // The live assistant turn is reasoning: thoughts have streamed, no answer
    // yet.
    private var isReasoningLive: Bool {
        var live = false
        if model.busy, let last = model.messages.last, !last.fromUser {
            live = last.text.isEmpty && !last.reasoning.isEmpty
        }
        return live
    }

    // Skip the rest of the reasoning: asks the session to inject </think> next
    // (a no-op once it has left <think>).
    private var quickAnswerButton: some View {
        Button(action: model.quickAnswer) {
            Label("Quick Answer", systemImage: "bolt.fill")
                .font(.caption)
                .fontWeight(.semibold)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(.thinMaterial, in: Capsule())
                .overlay(Capsule().stroke(.separator, lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }

    // The tail spacer is the target (not the last message): while the inset
    // is open, following keeps the streaming text lifted above the composer.
    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        proxy.scrollTo("chat-tail", anchor: .bottom)
    }

    // Shows just above the composer once autoscroll stops (the user scrolled
    // up). Tapping it re-enables follow and jumps to the newest tokens.

    private func toBottomButton(_ proxy: ScrollViewProxy) -> some View {
        Button {
            follow = true
            withAnimation { scrollToBottom(proxy) }
        } label: {
            Image(systemName: "arrow.down.circle.fill")
                .font(.title)
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
                if !m.fromUser, !m.reasoning.isEmpty {
                    ReasoningView(text: m.reasoning,
                                  doc: m.reasoningDoc,
                                  style: ContentView.scaled(
                                      ContentView.reasoningStyle,
                                      by: typeScale),
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

    // The turn's attached images above the prompt text: one image large but
    // bounded, two or three as a row of squares, four as a 2x2 grid, more
    // (should maxImages ever grow) a scrollable row -- always visible,
    // never the whole screen.

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

    // A square crop: fill then clip, so any aspect lands as a clean tile.
    private func stripThumb(_ cg: CGImage, _ side: CGFloat) -> some View {
        Image(decorative: cg, scale: 1)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: side, height: side)
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // The answer bubble: Markdown for an assistant turn when the toggle is on
    // AND the doc has content (the doc trails the raw text by one ticker
    // beat, so plain text bridges the gap); user input is never Markdown.

    // The sentence the voice is on, offered ONLY to the turn it came from:
    // speech follows one answer, and a stale highlight left on an earlier
    // bubble would point at the wrong conversation entirely.
    private func spoken(_ m: ChatModel.Message) -> String? {
        m.id == model.messages.last?.id ? model.speech.spokenText : nil
    }

    private func answerBubble(_ m: ChatModel.Message,
                              _ answer: String) -> some View {
        Group {
            if !m.fromUser && model.renderMarkdown
                && !m.answerDoc.items.isEmpty {
                // One self-sizing single-surface text view per answer, so a
                // drag selects across paragraphs and snaps around whole
                // tables / code (Bridges' atomic snapping), and Find can scroll
                // + highlight matches in the real bubble. The transcript owns
                // scroll (scrolls: false).
                MarkdownTextView(m.answerDoc,
                                 style: ContentView.scaled(
                                     ContentView.answerStyle, by: typeScale),
                                 find: findController, findId: m.id,
                                 scrolls: false, speaking: spoken(m))
            } else if !m.fromUser {
                // Plain-text mode (or the brief gap before the Markdown doc is
                // ready): the raw text on the SAME findable single surface, so
                // Find works with Markdown rendering off.
                PlainTextView(answer,
                              style: ContentView.scaled(
                                  ContentView.answerStyle, by: typeScale),
                              find: findController, findId: m.id,
                              scrolls: false, speaking: spoken(m))
            } else {
                Text(answer).textSelection(.enabled)
            }
        }
        .padding(10)
        .background(m.fromUser ? Color.accentColor.opacity(0.2)
                               : Color.gray.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // While the engine ingests the prompt (before the first reasoning or
    // content token) show the whimsical stage phrase (prompt / documents /
    // image) in the SAME small style as the reasoning disclosure header, as a
    // single line -- once tokens flow it is replaced by that header (thinking)
    // or the answer, never stacked above them.

    private var prefillWhimsical: some View {
        Text(model.thinkStatus)
            .font(.caption)
            .foregroundStyle(.secondary)
            .modifier(Shimmer(active: true))
    }

    // nil while the turn has no content, so an empty (e.g. stopped) turn shows
    // no bubble; the reasoning header / prefill line carries the live state.

    private func answerText(_ m: ChatModel.Message) -> String? {
        m.text.isEmpty ? nil : m.text
    }

    // A SETTLED assistant turn that made tool calls yet committed no answer
    // (the tool-call budget ran out with the model still calling, or it
    // stopped inside <think> after its tool rounds). The session rightly
    // commits nothing -- fabricated content must not enter the KV -- so the
    // transcript says so instead of showing Thoughts over silence. A stopped
    // plain turn (no tool rounds) keeps its quiet no-bubble behavior.

    private func isAnswerless(_ m: ChatModel.Message) -> Bool {
        !m.fromUser && !m.toolRounds.isEmpty && !isLive(m)
    }

    // The turn being generated right now. Everything transient about a
    // bubble hangs off this: a state that outlives its turn is either a stale
    // label or, where it drives an animation, a commit on every display
    // refresh for the rest of the conversation.

    private func isLive(_ m: ChatModel.Message) -> Bool {
        model.busy && m.id == model.messages.last?.id
    }

    private var noAnswerNote: some View {
        Text("The model made tool calls but gave no final answer.")
            .font(.caption)
            .italic()
            .foregroundStyle(.secondary)
    }

    private var loopStopNote: some View {
        Text("The model began repeating itself and was stopped before it "
           + "reached an answer. Try rephrasing.")
            .font(.caption)
            .italic()
            .foregroundStyle(.secondary)
    }

    // The live turn is still ingesting the prompt: busy, the last message, and
    // no token decoded yet (prefilling clears on the first reasoning/content
    // piece). Distinct from isThinking, which stays true through the reasoning
    // stream (where the disclosure header, not the prefill line, is shown).

    private func isPrefilling(_ m: ChatModel.Message) -> Bool {
        isLive(m) && model.prefilling
    }

    // True only for the message being generated right now, before its answer
    // begins -- so the live marquee runs while thinking and stops the instant
    // content arrives, generation ends, or Stop is pressed (an interrupted
    // turn keeps an empty answer, so "text empty" alone would never clear).

    private func isThinking(_ m: ChatModel.Message) -> Bool {
        isLive(m) && m.text.isEmpty
    }

}

// Collapsed grey (#888) disclosure of an assistant turn's <think> reasoning.
// Expanded, it scrolls inside a cap of a third of the chat height, so a long
// chain of thought never pushes the answer off screen.

private struct ReasoningView: View {
    let text: String
    let doc: Markdown.Document
    let style: MarkdownStyle
    let markdown: Bool
    let active: Bool
    let label: String
    let maxHeight: CGFloat
    @State private var expanded = false
    // The expanded view follows its own bottom while streaming, until the user
    // scrolls up inside it to read back.
    @State private var innerFollow = true

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            header
            if expanded {
                reasoning
            }
        }
    }

    // The Button wraps ONLY the chevron + word (fixed size); the ticker is its
    // SIBLING in the HStack, not inside the label. A bare GeometryReader in
    // the label swallowed the leftover width (could go negative before layout
    // settled), and a TimelineView in a button label may skip scheduling.

    private var header: some View {
        HStack(spacing: 6) {
            Button {
                expanded.toggle()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: expanded
                          ? "chevron.down" : "chevron.right")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(label)
                        .font(.caption)
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
                            .font(.callout)
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

// The turn's tool rounds as a compact tree between the reasoning disclosure
// and the answer: one row per round (glyph, tool, arguments), the running row
// shimmering, an "error:" result tinting its glyph orange. Hovering a row
// (macOS pointer) or tapping it (iOS) pops the raw invocation and the full
// tool result. Hover dismissal has a short grace so the pointer can travel
// INTO the popover to scroll a long result without it vanishing.

// The transcript's own coordinate space: a tool row reports where it sits in
// it and the callout is placed in the same one, so the two agree while the
// transcript scrolls under them.
private let transcriptSpace = "transcript"

// Which tool row is showing its detail, and where that row currently sits in
// the transcript. ONE owner, because the row and the callout both open and
// close it: a pointer travelling from one to the other leaves both, and two
// independent timers would race to dismiss what the user is reaching for.
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

    // The row moved under a scroll or a growing answer; the callout follows it
    // rather than hanging where the row used to be.
    func track(_ message: UUID, _ round: Int, at frame: CGRect) {
        if showing(message, round) { anchor = frame }
    }

    func keep() { dismiss?.cancel() }

    func close() {
        dismiss?.cancel()
        messageId = nil
    }

    // The gap between the row and its callout is not a decision to dismiss,
    // so leaving either one only starts a grace period.
    func fade() {
        dismiss?.cancel()
        dismiss = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(300))
            if !Task.isCancelled { messageId = nil }
        }
    }
}

// A rounded body with a tail on one long edge. The tail marks WHICH row the
// detail belongs to, which a centered panel otherwise cannot say.
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
    // Whether this turn is still generating. A round's result is filled only
    // by its completion event, so a turn that is stopped, cancelled or fails
    // between the two leaves one nil for good -- and the shimmer keyed off it
    // is an INDEFINITE animation, which drives a view-graph update and a
    // synchronous render-server round trip on every display refresh for the
    // rest of the conversation's life, in an app that looks idle.
    let live: Bool
    // Where each row sits, so opening one knows what the callout points at.
    @State private var frames: [Int: CGRect] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Image(systemName: "wrench.and.screwdriver")
                    .font(.caption2)
                Text("Using tools")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            ForEach(rounds) { round in row(round) }
        }
    }

    private func row(_ round: ChatModel.ToolRound) -> some View {
        HStack(spacing: 6) {
            Text(round.id == rounds.last?.id ? "└─" : "├─")
                .font(.caption.monospaced())
            Image(systemName: round.symbol)
                .font(.caption2)
                .foregroundStyle(errored(round) ? Color.orange
                                                : Color.secondary)
            Text(round.label)
                .font(.caption)
            if !round.args.isEmpty {
                Text(round.args)
                    .font(.caption2)
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

// Raw debug view: the emitted name (and the resolved tool when the fuzzy
// match changed it), the full arguments, and the exact result string the
// model received -- what the KV saw, not a paraphrase. The result area HUGS
// its content up to a cap, so a three-line calculator answer gets a small
// popup while a fetched article scrolls inside the capped height.

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

// A static one-line glimpse of the newest reasoning: the tail is right-aligned
// (last tokens always visible, updating in place), the left edge fades under a
// gradient mask. No scroll -- at ~20 tok/s a live scroll is too fast to read,
// so this is texture; the animated phrase label carries the liveliness.

private struct ThinkingTicker: View {
    let text: String

    var body: some View {
        // No fixedSize: that pins the Text to its full intrinsic width and it
        // spills past the row. Head truncation keeps the newest tail while
        // fitting the strip, and the gradient fades the left where the
        // ellipsis would sit. suffix caps the string fed to Text; the frame
        // shows only what fits.
        Text(String(text.suffix(120)))
            .font(.caption)
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

// Gentle opacity pulse for a transient status. EXPLICIT per-frame opacity
// (TimelineView), never an implicit repeatForever animation: an in-flight
// implicit animation bleeds into unrelated changes on the same view -- it
// cross-faded old and new phrase text into overlapping glyphs and oscillated
// the layout on width changes. Shallow (1 -> 0.7) and slow (~3.6s a cycle):
// it breathes, and the text stays readable through the whole pulse.

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

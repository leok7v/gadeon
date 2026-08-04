import LLM
import SwiftUI

// Settings shown in-view (routed by model.showSettings), not a sheet.

struct SettingsView: View {

    @Bindable var model: ChatModel
    let onClose: () -> Void
    @State private var category: Category = .systemPrompt
    // The set a tapped trash asks to delete; non-nil drives the confirm alert.
    @State private var deleteName: String?
    // Factory Reset lives in Misc, revealed only while Option is held (a
    // destructive action kept off the default surface). iOS has no Option key,
    // so its monitor is a no-op there.
    @State private var optionDown = false
    @State private var confirmReset = false
    @State private var confirmClear = false
    // The rail's labels scale with Dynamic Type; a fixed 180pt rail
    // hyphenates them ("Sys-tem Promp-t") under Larger Text, so the rail
    // and its icon slots scale on the same curve (width capped so the rail
    // never eats half the pane).
    @ScaledMetric(relativeTo: .body) private var railWidth: CGFloat = 180
    @ScaledMetric(relativeTo: .body) private var railIcon: CGFloat = 20

    enum Category: String, CaseIterable, Identifiable {
        case systemPrompt = "System Prompt"
        case models = "Models"
        case voice = "Voice"
        case vision = "Vision"
        case view = "View"
        case misc = "Misc"
        var id: String { rawValue }
        var symbol: String {
            switch self {
            case .systemPrompt: return "text.bubble"
            case .models: return "internaldrive"
            case .voice: return "waveform"
            case .vision: return "eye"
            case .view: return "paintbrush"
            case .misc: return "slider.horizontal.3"
            }
        }
    }

    // Done sits top-trailing (the HIG spot for settings dismissal; a
    // bottom-prominent button is the call-to-action pattern and belongs to
    // the disclaimer, not here), mirrored by the debug view's header.
    // Return and Esc both close.
    var body: some View {
        Group {
            if isOS {
                compactBody
            } else {
                regularBody
            }
        }
        .modifier(OptionKeyMonitor(down: $optionDown))
        .alert("Factory reset?", isPresented: $confirmReset) {
            Button("Proceed", role: .destructive) { model.factoryReset() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This deletes every downloaded model and all settings, then "
               + "quits. The next launch shows the terms again and re-downloads "
               + "the 0.8B model from scratch.")
        }
        .alert("Clear all conversations?", isPresented: $confirmClear) {
            Button("Clear", role: .destructive) {
                model.clearAllConversations()
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Deletes every saved conversation from this device. This "
               + "cannot be undone.")
        }
    }

    // iPhone (compact): an Apple-style grouped list that drills into each pane,
    // Done top-right -- not the desktop rail, which is cramped on a phone.
    private var compactBody: some View {
        NavigationStack {
            List {
                ForEach(categories) { item in
                    NavigationLink {
                        paneScroll(item).navigationTitle(item.rawValue)
                    } label: {
                        Label(item.rawValue, systemImage: item.symbol)
                    }
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done", action: onClose)
                }
            }
        }
    }

    // iPad / Mac: the persistent rail + detail.
    private var regularBody: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Settings", systemImage: "gearshape")
                    .font(.headline)
                Spacer()
                Button("Done", action: onClose)
                    .keyboardShortcut(.defaultAction)
                Button("", action: onClose)
                    .keyboardShortcut(.cancelAction)
                    .hidden()
            }
            .padding(12)
            Divider()
            HStack(spacing: 0) {
                rail
                Divider()
                paneScroll(category)
            }
        }
    }

    // Vision appears only when the active model actually offers the
    // tile/fit choice; a pane that switches nothing (the 0.8B, iOS) just
    // confuses. Models hides when this device ships only the base model
    // (nothing to pick or manage).
    private var categories: [Category] {
        Category.allCases.filter { c in
            (c != .vision || model.allowsTiling)
                && (c != .models || Models.all.count > 1)
                && (c != .voice || model.speech.available)
        }
    }

    private var rail: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(categories) { item in railRow(item) }
            Spacer()
        }
        .padding(12)
        .frame(width: min(railWidth, 300))
    }

    private func railRow(_ item: Category) -> some View {
        let selected = category == item
        return Button {
            category = item
        } label: {
            HStack(spacing: 8) {
                Image(systemName: item.symbol).frame(width: railIcon)
                Text(item.rawValue)
                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(selected ? Color.accentColor.opacity(0.15) : .clear,
                        in: RoundedRectangle(cornerRadius: 7))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(selected ? Color.accentColor : Color.primary)
    }

    private func paneScroll(_ item: Category) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                pane(item)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private func pane(_ item: Category) -> some View {
        switch item {
        case .systemPrompt: systemPromptPane
        case .models: modelsPane
        case .voice: voicePane
        case .vision: visionPane
        case .view: viewPane
        case .misc: miscPane
        }
    }

    private var voicePane: some View {
        @Bindable var speech = model.speech
        return VStack(alignment: .leading, spacing: 10) {
            title("Voice")
            explain("Replies are read aloud on this device; nothing is sent "
                + "anywhere. Tap a voice to hear it. Opening the microphone "
                + "while a reply is being read stops the reading, so you can "
                + "interrupt without waiting.")
            Picker("Speak", selection: $speech.mode) {
                ForEach(VoiceSession.Mode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            explain(speech.mode.detail)
            ForEach(Speech.voices) { v in voiceRow(v) }
            Divider().padding(.vertical, 4)
            HStack {
                Text("Speed")
                Slider(value: $speech.speed, in: 0.7...1.5, step: 0.05)
                Text(String(format: "%.2fx", speech.speed))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func voiceRow(_ v: SpeechVoice) -> some View {
        let picked = v.name == model.speech.voiceName
        return Button {
            model.speech.voiceName = v.name
            model.speech.preview(v)
        } label: {
            HStack(spacing: 8) {
                Image(systemName: picked ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(picked ? Color.accentColor : .secondary)
                Text(v.name)
                Spacer()
                Image(systemName: "play.circle")
                    .foregroundStyle(.secondary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var systemPromptPane: some View {
        VStack(alignment: .leading, spacing: 10) {
            title("System Prompt")
            explain("Prepended to every conversation to steer tone and "
                + "behaviour. Takes effect on the next New Chat.")
            TextEditor(text: $model.systemPrompt)
                .font(.body)
                .frame(minHeight: 160)
                .padding(6)
                .overlay {
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.separator, lineWidth: 0.5)
                }
            Button("Reset to default") {
                model.systemPrompt = ChatModel.defaultSystemPrompt
            }
            .disabled(model.systemPrompt == ChatModel.defaultSystemPrompt)
        }
    }

    private var modelsPane: some View {
        VStack(alignment: .leading, spacing: 10) {
            title("Models")
            explain("Tap a downloaded model's ring to make it active. Each "
                + "model downloads once and is stored on this device; deleting "
                + "frees its disk space and it can be downloaded again. The "
                + "active model cannot be deleted.")
            ForEach(Models.all, id: \.self) { name in modelRow(name) }
                .id(model.diskRevision)
        }
        .alert("Delete \(deleteName ?? "")?", isPresented: Binding(
            get: { deleteName != nil },
            set: { shown in if !shown { deleteName = nil } }
        )) {
            Button("Delete", role: .destructive) {
                if let name = deleteName { model.deleteModel(name) }
                deleteName = nil
            }
            Button("Cancel", role: .cancel) { deleteName = nil }
        } message: {
            Text("Frees the space it uses; you can download it again later.")
        }
    }

    private func modelRow(_ name: String) -> some View {
        let downloaded = model.isDownloaded(name)
        let active = name == model.modelName
        let bytes = ModelCatalog.source(name)?.bytes ?? 0
        return HStack(spacing: 8) {
            Text(name)
            if active {
                Image(systemName: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(Color.accentColor)
            }
            Spacer()
            Text(ByteCountFormatter.string(fromByteCount: bytes,
                                           countStyle: .file))
                .font(.caption)
                .foregroundStyle(.secondary)
            rowButton(name, downloaded: downloaded, active: active)
        }
        .padding(.vertical, 2)
    }

    // ONE glyph per row: trash for a downloaded set (disabled on the active
    // one), arrow-down for a fetchable one (routes into the same consent ->
    // download -> switch flow as the composer picker).

    private func rowButton(_ name: String, downloaded: Bool,
                           active: Bool) -> some View {
        let idle = !model.busy && !model.downloading
        return Group {
            if downloaded {
                Button { deleteName = name } label: {
                    Image(systemName: "trash")
                        .frame(width: 22, height: 22)
                }
                .buttonStyle(.plain)
                .foregroundStyle(idle && !active
                                 ? Color.red : Color.secondary.opacity(0.4))
                .disabled(active || !idle)
                .help(active ? "The active model cannot be deleted"
                             : "Delete from this device")
            } else {
                Button { model.requestDownload(name) } label: {
                    Image(systemName: "arrow.down.circle")
                        .frame(width: 22, height: 22)
                }
                .buttonStyle(.plain)
                .foregroundStyle(idle ? Color.accentColor : Color.secondary)
                .disabled(!idle)
                .help("Download and switch to this model")
            }
        }
    }

    private var visionPane: some View {
        VStack(alignment: .leading, spacing: 10) {
            title("Vision")
            explain("How an attached image is fed to the model.")
            Picker("Image handling", selection: $model.visionMode) {
                ForEach(ChatModel.VisionMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            explain(model.visionMode.detail)
        }
    }

    private var viewPane: some View {
        VStack(alignment: .leading, spacing: 10) {
            title("View")
            Toggle("Status line", isOn: $model.statusLine)
                .toggleStyle(.switch)
            explain("A line under the composer with the turn's context size, "
                + "token counts, memory and speed.")
            Toggle("Markdown", isOn: $model.renderMarkdown)
                .toggleStyle(.switch)
            explain("Render replies and reasoning as formatted Markdown "
                + "(headings, lists, code, tables). Off shows plain text.")
            Toggle("Always show sample prompts", isOn: $model.alwaysShowSamples)
                .toggleStyle(.switch)
            explain("Sample prompts are tappable examples on an empty chat. "
                + "They show until you have tried each one; turn this on to "
                + "always show them.")
        }
    }

    private var miscPane: some View {
        VStack(alignment: .leading, spacing: 10) {
            title("Misc")
            // Thinking drives the session's enable_thinking, so it applies to
            // EVERY turn -- text, dropped .txt/.md documents (a prompt in
            // themselves), and images (which now always carry a prompt). Takes
            // effect next turn, like the toolbar toggle it mirrors.
            Toggle("Thinking", isOn: Binding(
                get: { model.thinkingActive },
                set: { on in if on != model.thinking { model.toggleThinking() } }
            ))
            .toggleStyle(.switch)
            .disabled(!model.modelSupportsThinking)
            explain(model.modelSupportsThinking
                ? "The model works through the problem step by step before "
                    + "it answers. Answers get better; replies take longer. "
                    + "The reasoning shows above each answer, and Quick "
                    + "Answer skips the rest of it."
                : "\(Models.display(model.modelName)) answers directly and "
                    + "has no step-by-step mode, so this does nothing here.")
            // The budget only means something while the model can think; a
            // token cap for reasoning that never happens is noise.
            if model.modelSupportsThinking {
                Picker("Thinking budget", selection: $model.thinkBudget) {
                    ForEach(ChatModel.ThinkBudget.allCases) { size in
                        Text(size.rawValue).tag(size)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                // The token cap behind this is derived per model and device,
                // which is the reason the setting is stated in SECONDS at
                // all. Naming the count and the checkpoint answers a question
                // no one reading this pane is asking.
                explain("How long the model may think before it is nudged to "
                    + "answer: about \(Int(model.thinkBudget.seconds)) "
                    + "seconds.")
            }
            Toggle("Wikipedia", isOn: Binding(
                get: { model.wikipedia },
                set: { on in
                    model.setAccess(wikipedia: on, web: model.webAccess)
                }
            ))
            .toggleStyle(.switch)
            explain("Look up facts in Wikipedia. Your question is matched to "
                + "an article on this device; nothing you type is sent "
                + "anywhere. The app only downloads the matched article and "
                + "today's headlines from wikipedia.org.")
            Toggle("Web access", isOn: Binding(
                get: { model.webAccess },
                set: { on in
                    model.setAccess(wikipedia: model.wikipedia, web: on)
                }
            ))
            .toggleStyle(.switch)
            explain("Let the model search and read the web, and check the "
                + "weather. Search terms based on your question are sent to "
                + "the search engine, and pages the model picks are fetched. "
                + "Weather uses your approximate location.")
            // TODO: surface the overthink penalty (ChatModel.overthinkLambda,
            // fixed 1.0): a per-token logit bias on reasoning branch-openers
            // ("Wait", "However", ...) applied only while thinking, nudging
            // the chain of thought toward the answer sooner. Robust across
            // 0.5-4.0 (arxiv 2606.00206); a slider would trade reasoning
            // depth for speed.
            Toggle("Confirm before deleting a conversation",
                   isOn: $model.confirmDeleteConversation)
                .toggleStyle(.switch)
            explain("Ask before the sidebar trash deletes a conversation. "
                + "Off deletes immediately.")
            if !ConversationStore.shared.list.isEmpty {
                Divider().padding(.vertical, 4)
                Button(role: .destructive) { confirmClear = true } label: {
                    Label("Clear all conversations", systemImage: "trash")
                }
                explain("Delete every saved conversation from this device.")
            }
            if optionDown {
                Divider().padding(.vertical, 4)
                Button(role: .destructive) { confirmReset = true } label: {
                    Label("Factory Reset", systemImage: "trash")
                }
                explain("Deletes every downloaded model and all settings, then "
                    + "quits. Revealed only while Option is held.")
            }
        }
    }

    // The compact path pushes each pane with a navigation title already
    // naming it, so the pane's own heading is the same word a second time
    // directly beneath the first. Only the rail layout, which has no
    // navigation bar to carry it, needs one.
    @ViewBuilder
    private func title(_ text: String) -> some View {
        if !isOS {
            Text(text).font(.title3).bold()
        }
    }

    private func explain(_ text: String) -> some View {
        Text(text)
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

}

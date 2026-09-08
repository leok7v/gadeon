import Chat
import LLM
import SwiftUI

struct SettingsView: View {

    @Bindable var model: ChatModel
    let onClose: () -> Void
    @State private var deleteName: String?
    @State private var confirmReset = false
    @State private var confirmClear = false
    @ScaledMetric(relativeTo: .body) private var railWidth: CGFloat = 180
    @ScaledMetric(relativeTo: .body) private var railIcon: CGFloat = 20
    @State private var draftZoom: Int?
    @State private var gateRevision = 0
    @State private var path: [Category] = []

    private var notch: Int { draftZoom ?? model.textZoom }
    private var draftScale: CGFloat { 1 + CGFloat(notch) / 10 }

    private func dismiss() {
        if let draftZoom { model.textZoom = draftZoom }
        onClose()
    }

    enum Category: String, CaseIterable, Identifiable {
        case systemPrompt = "System Prompt"
        case models = "Models"
        case voice = "Voice"
        case view = "View"
        case intelligence = "Intelligence"
        case misc = "Misc"
        case diagnostics = "Diagnostics"
        case about = "About"
        var id: String { rawValue }
        var symbol: String {
            switch self {
            case .systemPrompt: return "text.bubble"
            case .models: return "internaldrive"
            case .voice: return "waveform"
            case .view: return "paintbrush"
            case .intelligence: return "brain"
            case .misc: return "slider.horizontal.3"
            case .diagnostics: return "stethoscope"
            case .about: return "info.circle"
            }
        }
    }

    var body: some View {
        Group {
            if isOS {
                compactBody
            } else {
                regularBody
            }
        }
        .modifier(OptionKeyMonitor(down: $model.optionDown))
        .alert("Factory reset?", isPresented: $confirmReset) {
            Button("Proceed", role: .destructive) { model.factoryReset() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This deletes every downloaded model and all settings, then "
               + "quits. The next launch shows the terms again and downloads "
               + "a model from scratch.")
        }
        .alert("Clear all conversations?", isPresented: $confirmClear) {
            Button("Clear", role: .destructive) {
                model.clearAllConversations()
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Moves every saved conversation to the trash, where it "
               + "stays for 30 days.")
        }
    }

    private var compactBody: some View {
        NavigationStack(path: $path) {
            List {
                ForEach(categories) { item in
                    NavigationLink(value: item) {
                        Label(item.rawValue, systemImage: item.symbol)
                    }
                }
            }
            .navigationDestination(for: Category.self) { item in
                paneScroll(item).navigationTitle(item.rawValue)
            }
            .navigationTitle("Settings")
            .toolbar { DoneToolbar(action: dismiss) }
        }
        .onAppear {
            if model.settingsCategory != .systemPrompt {
                path = [model.settingsCategory]
            }
        }
    }

    private var regularBody: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Settings", systemImage: "gearshape")
                    .appFont(.headline)
                Spacer()
                DoneButton(action: dismiss)
                EscapeToClose(action: dismiss)
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

    private var categories: [Category] {
        Category.allCases.filter { c in
            (c != .models || Models.all.count > 1)
                && (c != .voice || model.speech.available)
                && (c != .diagnostics || model.statusLine)
        }
    }

    private var rail: some View {
        let current = category
        return VStack(alignment: .leading, spacing: 4) {
            ForEach(categories) { item in railRow(item, current) }
            Spacer()
        }
        .padding(12)
        .frame(width: min(railWidth, 300) * model.textScale)
    }

    private func railRow(_ item: Category,
                         _ current: Category) -> some View {
        let selected = current == item
        return Button {
            model.settingsCategory = item
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
        case .view: viewPane
        case .intelligence: intelligencePane
        case .misc: miscPane
        case .diagnostics: diagnosticsPane
        case .about: aboutPane
        }
    }

    private var aboutPane: some View {
        VStack(alignment: .leading, spacing: 10) {
            title("About")
            creditRow(Credits.app)
            Divider().padding(.vertical, 4)
            explain("Built on the work below. Each entry says who made it, "
                + "the terms it arrives under, and what this app changed.")
            ForEach(Credits.all) { item in creditRow(item) }
            Divider().padding(.vertical, 4)
            licence("Apache License 2.0", Licence.apache2)
            licence("GNU General Public License v3", Licence.gpl3)
        }
    }

    private func creditRow(_ item: Credit) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Text(item.name).bold()
                Text(item.terms)
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(item.author).appFont(.callout)
            if let link = item.link {
                Link(item.source, destination: link).appFont(.caption)
            }
            explain(item.changed)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 4)
    }

    private func licence(_ name: String, _ text: String) -> some View {
        DisclosureGroup(name) {
            ScrollView {
                Text(text)
                    .appFont(.caption2)
                    .monospaced()
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 6)
            }
            .frame(maxHeight: 260)
        }
    }

    private var voicePane: some View {
        @Bindable var speech = model.speech
        return VStack(alignment: .leading, spacing: 10) {
            title("Voice")
            explain("Replies are read aloud on this device; nothing is sent "
                + "anywhere. Tap a voice to hear it. Opening the microphone "
                + "stops the reading, so you can interrupt at any time.")
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
            explain("Instructions the assistant reads before every "
                + "conversation, for tone and behaviour. Applies from the "
                + "next new chat.")
            TextEditor(text: $model.systemPrompt)
                .appFont(.body)
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

    private var category: Category {
        categories.contains(model.settingsCategory)
            ? model.settingsCategory : .systemPrompt
    }

    private var unlocked: Bool { model.statusLine && model.optionDown }

    private var listed: [String] { Models.offered(unlocked: unlocked) }

    private var modelsPane: some View {
        VStack(alignment: .leading, spacing: 10) {
            title("Models")
            explain("Each model downloads once and stays on this device. "
                + "Deleting one frees the space, and it can be downloaded "
                + "again. The model in use cannot be deleted. To delete it, "
                + "switch to another model first.")
            ForEach(listed, id: \.self) { name in
                modelRow(name)
            }
            .id(model.diskRevision)
            if unlocked {
                explain("Every model this Mac can run, shown while Option is "
                    + "held with Debug on. Normally only the ones its memory "
                    + "allows are offered.")
            }
        }
        .alert("Delete \(Models.display(deleteName ?? "", among: listed))?",
               isPresented: Binding(
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
            Text(Models.display(name, among: listed))
            if active {
                Image(systemName: "checkmark.circle.fill")
                    .appFont(.caption)
                    .foregroundStyle(Color.accentColor)
            }
            Spacer()
            Text(ByteCountFormatter.string(fromByteCount: bytes,
                                           countStyle: .file))
                .appFont(.caption)
                .foregroundStyle(.secondary)
            rowButton(name, downloaded: downloaded, active: active)
        }
        .padding(.vertical, 2)
    }

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
                .help(active ? "Switch to another model first"
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

    private var imageRows: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Images")
            Picker("Image detail", selection: $model.imageBudget) {
                ForEach(ChatModel.ImageBudget.allCases) { size in
                    Text(size.rawValue).tag(size)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            explain("How closely the assistant looks at a picture: about "
                    + "\(model.imageBudget.tokens) tokens per picture, at "
                    + "its own shape. Larger sees finer detail and takes "
                    + "longer.")
        }
    }

    private var documentRows: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Documents")
            Picker("Document size", selection: $model.docBudget) {
                ForEach(ChatModel.DocBudget.allCases) { size in
                    Text(size.rawValue).tag(size)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            explain("How much of an attached document the assistant reads: "
                    + "up to about \(model.docBudget.pages) pages of text. "
                    + "Anything past that is left out, and the chat says so.")
        }
    }

    private var viewPane: some View {
        VStack(alignment: .leading, spacing: 10) {
            title("View")
            Toggle("Markdown", isOn: $model.renderMarkdown)
                .toggleStyle(.switch)
            explain("Show replies with headings, lists, code and tables.")
            Toggle("Always show sample prompts", isOn: $model.alwaysShowSamples)
                .toggleStyle(.switch)
            explain("Keep the examples on an empty chat after you have tried "
                + "them all.")
            Text("Zoom")
            textSizeRow
        }
    }

    private var textSizeRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Text("a").font(.system(size: 11))
                VStack(spacing: 3) {
                    Slider(value: zoomNotch,
                           in: -Double(ChatModel.zoomLimit)
                               ... Double(ChatModel.zoomLimit),
                           step: 1)
                    detents
                    Button("Reset Zoom") { draftZoom = 0 }
                        .disabled(notch == 0)
                }
                .frame(maxWidth: 220)
                Text("A").font(.system(size: 21))
                Spacer()
            }
            draftSample
        }
    }

    private var zoomNotch: Binding<Double> {
        Binding(get: { Double(notch) },
                set: { value in
                    draftZoom = ChatModel.clampZoom(Int(value))
                })
    }

    private var detents: some View {
        HStack(spacing: 0) {
            ForEach(-ChatModel.zoomLimit ... ChatModel.zoomLimit,
                    id: \.self) { notch in
                if notch > -ChatModel.zoomLimit { Spacer(minLength: 0) }
                Capsule()
                    .fill(notch == 0 ? Color.accentColor : Color.secondary)
                    .frame(width: notch == 0 ? 2 : 1,
                           height: notch == 0 ? 9 : 5)
            }
        }
        .padding(.horizontal, isOS ? 13 : 5)
        .allowsHitTesting(false)
    }

    private var draftSample: some View {
        Text("Text and controls throughout the app, on top of your "
            + "device's own text size.")
            .font(appTextFont(.callout, draftScale))
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var intelligencePane: some View {
        VStack(alignment: .leading, spacing: 10) {
            title("Intelligence")
            if model.modelSupportsThinking { thinkingRows }
            if model.canSuggestFollowups {
                Toggle("Suggest follow-up questions",
                       isOn: $model.suggestFollowups)
                    .toggleStyle(.switch)
                explain("Suggest a sensible next question as a hint in the "
                    + "message box. "
                    + (isOS ? "Swipe right to ask it."
                            : "Tab or the right arrow asks it."))
            }
            Toggle("Wikipedia", isOn: Binding(
                get: { model.wikipedia },
                set: { on in
                    model.setAccess(wikipedia: on, web: model.webAccess)
                }
            ))
            .toggleStyle(.switch)
            explain("Let the assistant look things up in Wikipedia. The "
                + "lookup runs locally on your device, so what you type "
                + "stays here. Only the matching article is downloaded.")
            Toggle("Web Access", isOn: Binding(
                get: { model.webAccess },
                set: { on in
                    model.setAccess(wikipedia: model.wikipedia, web: on)
                }
            ))
            .toggleStyle(.switch)
            explain("Let the assistant search, fetch and read the Internet. "
                + "Your search terms leave the device, and the weather uses "
                + "your rough location.")
            if model.canAttachImages { imageRows }
            documentRows
        }
    }

    private var thinkingRows: some View {
        VStack(alignment: .leading, spacing: 10) {
            Toggle("Thinking", isOn: Binding(
                get: { model.thinkingActive },
                set: { on in
                    if on != model.thinking { model.toggleThinking() }
                }
            ))
            .toggleStyle(.switch)
            explain("Think before answering. Slower, but better on hard "
                + "questions. The thinking shows above each answer, and "
                + "Quick Answer cuts it short.")
            Picker("Thinking budget", selection: $model.thinkBudget) {
                ForEach(ChatModel.ThinkBudget.allCases) { size in
                    Text(size.rawValue).tag(size)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            explain("How long to think before answering: about "
                + "\(Int(model.thinkBudget.seconds)) seconds.")
            if model.modelSupportsReasoningEffort { reasoningEffortRow }
        }
    }

    private var miscPane: some View {
        VStack(alignment: .leading, spacing: 10) {
            title("Misc")
            Toggle("Confirm before deleting a conversation",
                   isOn: $model.confirmDeleteConversation)
                .toggleStyle(.switch)
            explain("Ask before a conversation is deleted. Off deletes at "
                + "once.")
            Toggle("Debug", isOn: $model.statusLine)
                .toggleStyle(.switch)
            explain("Show a status bar under the message box with context, "
                + "speed and memory, add a session details button to the "
                + "chat actions, and reveal the Diagnostics pane.")
            if !ConversationStore.shared.list.isEmpty {
                Divider().padding(.vertical, 4)
                Button(role: .destructive) { confirmClear = true } label: {
                    Label("Clear all conversations", systemImage: "trash")
                }
                .disabled(model.busy)
                explain("Move every saved conversation to the trash.")
            }
            if unlocked {
                Divider().padding(.vertical, 4)
                Button(role: .destructive) { confirmReset = true } label: {
                    Label("Factory Reset", systemImage: "trash")
                }
                explain("Delete every downloaded model and all settings, "
                    + "then quit.")
            }
        }
    }

    private var reasoningEffortRow: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Reasoning Effort")
            Picker("Reasoning effort", selection: Binding(
                get: { model.reasoningEffort },
                set: { level in model.setReasoningEffort(level) }
            )) {
                ForEach(ChatModel.ReasoningEffort.allCases) { level in
                    Text(level.rawValue).tag(level)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            explain("How hard to think. High thinks longer and more "
                + "carefully but can go in circles. Low answers sooner. "
                + "Applies from the next new chat.")
        }
    }

    private var diagnosticsPane: some View {
        VStack(alignment: .leading, spacing: 10) {
            title("Diagnostics")
            explain("Extra detail for the diagnostics log. A switch that is "
                + "off costs nothing. The session details screen says where "
                + "the log is.")
            ForEach(DiagGate.allCases) { gate in gateRow(gate) }
                .id(gateRevision)
        }
    }

    private func gateRow(_ gate: DiagGate) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Toggle(gate.label, isOn: Binding(
                get: { gate.on },
                set: { on in
                    gate.set(on)
                    gateRevision += 1
                }
            ))
            .toggleStyle(.switch)
            explain(gate.detail)
        }
    }

    @ViewBuilder
    private func title(_ text: String) -> some View {
        if !isOS {
            Text(text).appFont(.title3).bold()
        }
    }

    private func explain(_ text: String) -> some View {
        Text(text)
            .appFont(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

}

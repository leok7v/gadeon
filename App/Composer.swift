import SwiftUI

// The prompt card: a growing PromptEditor with the model picker, the [+] image
// attach + reserved mic buttons, and Send along its bottom edge. Return sends and
// Shift+Return inserts a newline (handled in PromptEditor, since a vertical
// TextField would otherwise swallow Return).

struct Composer: View {

    @Bindable var model: ChatModel
    @Binding var focused: Bool
    // The prompt text (preferredFont .body) scales with Dynamic Type; the
    // control glyphs and the model label are fixed point sizes, so scale
    // them by the same curve or they read as a different UI under Larger
    // Text.
    @ScaledMetric(relativeTo: .body) private var controlSize: CGFloat = 15
    @ScaledMetric(relativeTo: .body) private var labelSize: CGFloat = 13
    @ScaledMetric(relativeTo: .body) private var slotSize: CGFloat = 22
    @ScaledMetric(relativeTo: .body) private var sendSize: CGFloat = 24

    // Ten lines ~ a third of a default window. A hard line cap, not a
    // GeometryReader fraction (which re-measures on every resize and makes the
    // card twitch while the window is dragged).
    private static let maxLines = 10
    private static let caveat = "Chat is AI and can make mistakes."

    var body: some View {
        VStack(spacing: 6) {
            card
            footnote
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 6)
        // Focus the field once the composer appears, so the window opens
        // ready to type. The delay lets the view settle; setting focus at
        // raw appear does not take.
        .onAppear {
            Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(250))
                focused = true
            }
        }
    }

    private var card: some View {
        VStack(spacing: 8) {
            ForEach(model.attachedImages) { img in imageChip(img) }
            ForEach(model.attachedDocs) { doc in docChip(doc) }
            if let warning = model.attachmentWarning { warningBanner(warning) }
            // macOS lets you compose the NEXT message while a reply streams
            // (Send stays blocked -- submitReturn gates on canSend, which needs
            // !busy, and the Send button is Stop meanwhile). iOS keeps the
            // field disabled during generation.
            PromptEditor(text: $model.input, focused: $focused,
                         caret: $model.caret, disabled: isOS && model.busy,
                         minLines: 2, maxLines: Composer.maxLines,
                         onSubmit: submitReturn,
                         onDropFiles: { model.handleDrop($0, at: model.caret) })
                .overlay(alignment: .topLeading) {
                    if model.input.isEmpty {
                        Text("Write a message…")
                            .font(.body)
                            .foregroundStyle(.tertiary)
                            .padding(.top, 2)
                            .allowsHitTesting(false)
                    }
                }
                // Editing a reference out of the prompt drops its attachment.
                .onChange(of: model.input) { _, _ in
                    model.reconcileAttachments()
                }
            controls
        }
        .padding(10)
        .background(.quinary,
                    in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .stroke(.separator, lineWidth: 0.5)
        }
    }

    // Plain Return submits; Shift+Return inserts a caret-position line break
    // inside the AppKit editor (native), so this only handles the send.
    // Guarded by canSend, so a Return on an empty / whitespace-only prompt
    // does nothing.

    private func submitReturn() {
        if model.canSend { model.send() }
    }

    private var controls: some View {
        HStack(spacing: 8) {
            AttachButton(model: model)
            accessButton
            thinkingButton
            Spacer()
            // The mic keeps its slot so Send never shifts as text arrives;
            // it just fades out once there is text (future buttons can sit
            // beside it in the reserved gap).
            reserved("microphone", model.voice)
                .opacity(model.typing ? 0 : 1)
            sendButton
        }
        .font(.system(size: controlSize))
    }

    // Per-turn Web access (airplane -> Wikipedia -> Internet) and Thinking sit
    // next to Send: both are decisions about THIS message. State-showing icons;
    // the app flashes a HUD to confirm each toggle (no hover on iOS). The model
    // picker moved to the title bar.

    private var accessButton: some View {
        let icon: String
        let color: Color
        switch model.accessState {
        case .offline: icon = "airplane"; color = .orange
        case .wikipedia: icon = "books.vertical"; color = .teal
        case .full: icon = "globe"; color = .accentColor
        }
        return Button(action: model.cycleAccess) {
            Image(systemName: icon)
                .foregroundStyle(color)
                .frame(width: slotSize, height: slotSize)
        }
        .buttonStyle(.plain)
        .help("Web access")
    }

    // Disabled, not hidden, on a model that does not reason: the slot stays put
    // so the row does not reflow when switching models, and the tooltip says
    // why rather than leaving a dead control unexplained.
    private var thinkingButton: some View {
        let on = model.thinkingActive
        // Built as a String, not inline: .help infers LocalizedStringKey, and
        // `+` does not concatenate two of those.
        let tip: String
        if model.modelSupportsThinking {
            tip = on ? "Thinking on" : "Thinking off"
        } else {
            tip = Models.display(model.modelName)
                + " answers directly; it does not think step by step"
        }
        return Button(action: model.toggleThinking) {
            Image(systemName: on ? "lightbulb.fill" : "lightbulb")
                .foregroundStyle(on ? Color.yellow : Color.secondary)
                .frame(width: slotSize, height: slotSize)
        }
        .buttonStyle(.plain)
        .disabled(!model.modelSupportsThinking)
        .help(tip)
    }

    private func reserved(_ symbol: String,
                          _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .frame(width: slotSize, height: slotSize)
        }
        .buttonStyle(.plain)
        .foregroundStyle(.tertiary)
        .disabled(true)
        .help("Coming soon")
    }

    private func imageChip(_ img: ChatModel.ImageAttachment) -> some View {
        HStack(spacing: 6) {
            imageThumb(img)
            Text(img.name)
                .lineLimit(1).truncationMode(.middle)
            Spacer()
            Button { model.clearImage(img.id) } label: {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.plain)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    // A ~24pt preview decoded at attach (cross-platform CGImage, no NSImage/
    // UIImage split); falls back to the photo glyph if the decode failed. The
    // preview is the only way to tell several attached photos apart -- iOS
    // library picks all arrive named "image".

    @ViewBuilder
    private func imageThumb(_ img: ChatModel.ImageAttachment) -> some View {
        if let cg = img.thumbnail {
            Image(decorative: cg, scale: 1)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: 24, height: 24)
                .clipShape(RoundedRectangle(cornerRadius: 4))
        } else {
            Image(systemName: "photo").frame(width: 24, height: 24)
        }
    }

    // A heads-up above the field when the pending attachments are large enough
    // to make the next turn's ingest a noticeable wait.
    private func warningBanner(_ text: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(text).lineLimit(2)
            Spacer()
        }
        .font(.caption)
        .foregroundStyle(.orange)
    }

    private func docChip(_ doc: ChatModel.Doc) -> some View {
        HStack(spacing: 6) {
            // A text file has no meaningful preview; the glyph is sized into the
            // same 24pt slot as an image thumbnail so the chip rows line up.
            Image(systemName: "doc.text").frame(width: 24, height: 24)
            Text(doc.name).lineLimit(1).truncationMode(.middle)
            Spacer()
            Button { model.clearDoc(doc.id) } label: {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.plain)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    // Send becomes Stop while a reply streams (generation is unbounded, so the
    // user needs an out). No .keyboardShortcut(.defaultAction) -- it would
    // register a second Return handler racing PromptEditor's.

    private var sendButton: some View {
        let icon = model.busy ? "stop.fill" : "arrow.up"
        let tint: Color = sendLive ? .white : .secondary
        return Button(action: fire) {
            Image(systemName: icon)
                .font(.system(size: labelSize, weight: .semibold))
                .frame(width: sendSize, height: sendSize)
                .foregroundStyle(tint)
                .background(sendBackground,
                            in: RoundedRectangle(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .disabled(!sendLive)
    }

    // Stop while a reply streams, else send. A plain method, not a
    // method-reference ternary in the Button, which overloads inference.

    private func fire() {
        if model.busy {
            model.stop()
        } else {
            model.send()
        }
    }

    private var sendLive: Bool { model.busy || model.canSend }

    private var sendBackground: Color {
        sendLive ? .accentColor : Color.secondary.opacity(0.18)
    }

    // The AI caveat is always shown; the Shift+Return hint only on macOS (iOS
    // has no such key) and only before the first keystroke.

    private var footnote: some View {
        Text(model.typing || isOS
             ? Composer.caveat
             : "Shift+Return for a new line.  " + Composer.caveat)
            .font(.caption2)
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity)
    }

}

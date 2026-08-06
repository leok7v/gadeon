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
    @ScaledMetric(relativeTo: .body) private var baseControl: CGFloat = 15
    @ScaledMetric(relativeTo: .body) private var baseLabel: CGFloat = 13
    @ScaledMetric(relativeTo: .body) private var baseSlot: CGFloat = 22
    @ScaledMetric(relativeTo: .body) private var baseSend: CGFloat = 24

    // A narrow screen gets BIGGER controls, which is the opposite of the
    // usual instinct. A tablet is held in two hands with the row under a
    // steady thumb; a phone is held in one, and a 22pt slot is well under
    // what a thumb reliably finds. Size class rather than device model, so an
    // iPad in a narrow split view -- just as cramped -- gets it too.
    @Environment(\.horizontalSizeClass) private var sizeClass
    private var touch: CGFloat { sizeClass == .compact ? 1.35 : 1.0 }

    // The reach allowance times the app's text-size setting. The metrics above
    // carry the platform's Dynamic Type, which moves on iOS and is flat on
    // macOS, so this is what sizes the controls there.
    private var scale: CGFloat { touch * model.textScale }

    private var controlSize: CGFloat { baseControl * scale }
    private var labelSize: CGFloat { baseLabel * scale }
    private var slotSize: CGFloat { baseSlot * scale }
    private var sendSize: CGFloat { baseSend * scale }

    // The AppKit / UIKit editor builds its font from a point size, so it is
    // the one thing in the card that cannot read its size off the
    // environment. It gets handed the same factor as everything else.
    @ScaledMetric(relativeTo: .body) private var editorType: CGFloat = 1
    private var editorScale: CGFloat { editorType * model.textScale }

    // The placeholder sits ON the editor, so it has to be sized off the same
    // number rather than off .body, which would drift from it at every stop
    // but the middle one.
    private var editorFont: Font {
        .system(size: PromptEditor.points(editorScale))
    }

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
                if focusOnAppear { focused = true }
            }
        }
    }

    // Focus raises the soft keyboard, which takes half a phone -- and on an
    // empty chat the sample cards ARE the invitation, so opening ready to
    // type covers the thing the screen is there to offer. macOS keeps the
    // focus unconditionally: no keyboard rises there, so it costs nothing.
    private var focusOnAppear: Bool {
        !isOS || !model.showSamples
    }

    private var card: some View {
        VStack(spacing: 8) {
            ForEach(model.attachedImages) { img in imageChip(img) }
            ForEach(model.attachedClips) { clip in clipChip(clip) }
            ForEach(model.attachedDocs) { doc in docChip(doc) }
            if let warning = model.attachmentWarning { warningBanner(warning) }
            // macOS lets you compose the NEXT message while a reply streams
            // (Send stays blocked -- submitReturn gates on canSend, which needs
            // !busy, and the Send button is Stop meanwhile). iOS keeps the
            // field disabled during generation.
            PromptEditor(text: $model.input, focused: $focused,
                         caret: $model.caret, disabled: isOS && model.busy,
                         minLines: 2, maxLines: Composer.maxLines,
                         scale: editorScale,
                         onSubmit: submitReturn,
                         onDropFiles: { model.handleDrop($0, at: model.caret) })
                .overlay(alignment: .topLeading) {
                    if model.input.isEmpty {
                        Text("Write a message…")
                            .font(editorFont)
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
        // The film strip rides BETWEEN the card's fill and its content: above
        // the fill so it is visible, below the controls so they stay legible
        // over it. It fills the card, and all four of its edges are feathered,
        // so it reads as the card's own backdrop rather than as a picture
        // parked on one edge of it.
        .background {
            ZStack {
                RoundedRectangle(cornerRadius: 12).fill(.quinary)
                FilmStrip(model: model, quiet: !composing)
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .stroke(.separator, lineWidth: 0.5)
        }
    }

    // Whether the composer is in USE, which is what retires the film strip.
    //
    // The two platforms need different signals and neither one works for
    // both. On iOS focus means the soft keyboard is up and has taken half
    // the screen. On macOS the field is focused from launch and stays that
    // way, so focus says nothing at all -- keying off it there would hide
    // the strip permanently -- and what marks the composer as in use is
    // whether anything has been typed into it.
    private var composing: Bool {
        isOS ? focused : !model.input.isEmpty
    }

    // Plain Return submits; Shift+Return inserts a caret-position line break
    // inside the AppKit editor (native), so this only handles the send.
    // Guarded by canSend, so a Return on an empty / whitespace-only prompt
    // does nothing.

    private func submitReturn() {
        if model.canSend { model.send() }
    }

    // A live voice exchange ADDS a transport row above the ordinary controls,
    // so the two or three things that matter at arm's length are big enough to
    // hit. It sits above rather than overlaying: an overlay would cover the
    // text field, and typing mid-conversation is common.
    //
    // The trigger is the exchange, not the preference -- someone who leaves
    // "speak replies" on permanently gets the transport only while there is
    // something to transport.
    private var inVoiceExchange: Bool {
        model.listening || model.speech.engaged || model.voiceReady
    }

    // The two rows carry different KINDS of control and that is why both are
    // present. The transport is momentary and about this sound: pause it,
    // resume it, stop the turn. The row below is modal and outlives the turn:
    // whether replies are spoken at all, and whether the microphone is open.
    // Swapping one for the other stranded the modes exactly when a listener
    // most wants them -- someone interrupted mid-reply has to silence the
    // voice and close the mic, and going to Settings for that means leaving
    // the conversation.

    private var controls: some View {
        VStack(spacing: 8) {
            if inVoiceExchange { transport }
            standardControls
        }
    }

    // The big mic is the ARM'S-LENGTH affordance: it belongs to the moment
    // when it is the speaker's move, or when the mic is open and the ring
    // needs somewhere to live. While a reply is being READ it would be a
    // second microphone next to the small one, so it stands down and the row
    // is what it says it is -- pause and stop for the sound.
    private var showBigMic: Bool { model.listening || model.voiceReady }

    private var transport: some View {
        HStack(spacing: transportGap) {
            Spacer()
            if showBigMic {
                // Orange is the system's OWN microphone-in-use colour on both
                // platforms (the menu-bar pill, the privacy dot), so an open
                // mic needs no convention of ours learned. Red would also
                // collide with Stop beside it. Green at a turn boundary: the
                // reply is done and it is the speaker's move, which a resting
                // accent circle does not say.
                transportButton(
                    model.listening ? "microphone.fill" : "microphone",
                    model.listening ? "Stop and send" : "Speak",
                    micTint,
                    listening: model.listening,
                    action: model.voice)
            }
            if model.speech.paused {
                transportButton("play.fill", "Resume", .accentColor) {
                    model.speech.resume()
                }
                // Silences the VOICE and nothing else -- the answer goes on
                // being written and can still be read. Cancelling the turn is
                // the Send button in the row below, which shows a stop glyph
                // whenever generation is running, so the two live on separate
                // controls instead of one glyph meaning both.
                //
                // Red, not grey: a muted fill on a large control reads as
                // disabled rather than as secondary.
                transportButton("stop.fill", "Stop speaking", .red) {
                    model.speech.stopSpeaking()
                }
            } else if model.speech.engaged {
                // Keyed to the turn, not to `speaking`: sound stops between
                // sentences, and a Pause that vanished in those gaps would be
                // unhittable exactly when it is wanted.
                transportButton("pause.fill", "Pause", .accentColor) {
                    model.speech.pause()
                }
            }
            Spacer()
        }
        .padding(.top, 2)
    }

    // A 60pt circle in a Mac window reads as a phone app pretending, so the
    // desktop keeps a restrained size and only the vocabulary changes.
    private var micTint: Color {
        let tint: Color
        if model.listening {
            tint = .orange
        } else if model.voiceReady {
            tint = .green
        } else {
            tint = .accentColor
        }
        return tint
    }

    private var transportSize: CGFloat { isOS ? slotSize * 2.6 : slotSize * 1.5 }
    private var transportGap: CGFloat { isOS ? 28 : 14 }

    private func transportButton(_ symbol: String, _ label: String,
                                 _ tint: Color, listening: Bool = false,
                                 action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: transportSize * 0.4, weight: .medium))
                .foregroundStyle(.white)
                .frame(width: transportSize, height: transportSize)
                .background(tint, in: Circle())
                .background {
                    if listening {
                        HeardRing(size: transportSize,
                                  level: model.speechLevel,
                                  hearing: model.hearingSpeech)
                    }
                }
        }
        .buttonStyle(.plain)
        .help(label)
        .accessibilityLabel(label)
    }

    private var standardControls: some View {
        HStack(spacing: 8) {
            AttachButton(model: model)
            accessButton
            thinkingButton
            Spacer()
            // Both stay VISIBLE while typing. They used to fade out once
            // there was text, which hid them in the state that needs them
            // most: someone who breaks off a spoken exchange to type is
            // exactly the person reaching for "stop reading to me" and "close
            // the mic". A recording control that hides itself is also how a
            // microphone gets left open.
            // HIDDEN, not disabled, where the device cannot speak at all: the
            // house rule keeps a dead control visible so its tooltip can say
            // why, but that is for a state which can change, and on a 3 GB
            // phone this one never will. Settings drops its Voice pane on the
            // same flag.
            if model.speech.available { speakerButton }
            micButton
            sendButton
        }
        .font(.system(size: controlSize))
    }

    // Speak replies: a persistent MODE, not a per-message action, so a voice
    // conversation is entered once. It also shows that sound is still
    // trailing a finished transcript -- the wave fills while the voice reads,
    // and tapping it then is the quiet way out.

    // One meaning: it turns speech on and off. Silencing just THIS reply is
    // the transport's Stop and the microphone, both of which are on screen
    // whenever there is something to silence -- and a button that meant
    // "be quiet" while sound played and "switch off" the rest of the time
    // read as a setting that would not stay off.

    // The filled glyph carries "sound is playing" on its own, WITHOUT a
    // symbol effect. An indefinite effect (.variableColor, .pulse) keeps the
    // SwiftUI display link running, so every display refresh becomes a view
    // graph update, a CA commit, and a synchronous round trip to the render
    // server -- and the whole conversation lives in that one surface group,
    // so the round trip grows with it. Bound to a flag that stays true for as
    // long as sound is queued, that outruns the render server and the window
    // stops updating altogether.

    private var speakerButton: some View {
        let on = model.speech.enabled
        let live = model.speech.speaking
        let tip = on ? "Replies are spoken" : "Speak replies"
        return Button { model.speech.enabled.toggle() } label: {
            Image(systemName: live ? "speaker.wave.2.fill"
                                   : (on ? "speaker.wave.2" : "speaker.slash"))
                .foregroundStyle(on ? Color.accentColor : .secondary)
                .frame(width: slotSize, height: slotSize)
        }
        .buttonStyle(.plain)
        .help(tip)
    }

    // Dictation. A tap opens the mic, a second tap closes it and sends what
    // was SAID -- the gate drops the pauses, so a long think in the middle of
    // a sentence costs nothing. Disabled on a model with no ears.

    private var micButton: some View {
        Button(action: model.voice) {
            Image(systemName: model.listening
                  ? "microphone.fill" : "microphone")
                .foregroundStyle(model.listening ? Color.orange : .secondary)
                .frame(width: slotSize, height: slotSize)
                .symbolEffect(.pulse, isActive: model.listening)
        }
        .buttonStyle(.plain)
        .disabled(!model.canAttachAudio || (model.busy && !model.listening))
        .help(model.listening ? "Stop and send" : "Speak")
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
        .appFont(.caption)
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
        .appFont(.caption)
        .foregroundStyle(.orange)
    }

    // Sound and video have no thumbnail worth the decode, so the glyph
    // carries the kind; the row otherwise matches the image and doc chips.
    private func clipChip(_ clip: ChatModel.ClipAttachment) -> some View {
        HStack(spacing: 6) {
            Image(systemName: clip.isVideo ? "film" : "waveform")
                .frame(width: 24, height: 24)
            Text(clip.name).lineLimit(1).truncationMode(.middle)
            Spacer()
            Button { model.clearClip(clip.id) } label: {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.plain)
        }
        .appFont(.caption)
        .foregroundStyle(.secondary)
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
        .appFont(.caption)
        .foregroundStyle(.secondary)
    }

    // Send becomes Stop while a reply streams (generation is unbounded, so the
    // user needs an out). No .keyboardShortcut(.defaultAction) -- it would
    // register a second Return handler racing PromptEditor's.
    //
    // It says nothing about the VOICE. Pause, Resume and Stop for the sound
    // live in the transport row, which is on screen whenever there is sound to
    // control; offering them here as well gave one glyph two meanings and left
    // the user guessing whether Stop would silence the voice or discard the
    // answer.

    private var sendButton: some View {
        Button(action: fire) {
            Image(systemName: primaryIcon)
                .font(.system(size: labelSize, weight: .semibold))
                .frame(width: sendSize, height: sendSize)
                .foregroundStyle(sendLive ? .white : .secondary)
                .background(sendBackground,
                            in: RoundedRectangle(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .disabled(!sendLive)
        .help(primaryHelp)
    }

    private var primaryIcon: String {
        model.busy ? "stop.fill" : "arrow.up"
    }

    private var primaryHelp: String {
        model.busy ? "Stop" : "Send"
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

    private var sendLive: Bool {
        model.busy || model.canSend
    }

    private var sendBackground: Color {
        sendLive ? .accentColor : Color.secondary.opacity(0.18)
    }

    // The AI caveat is always shown; the Shift+Return hint only on macOS (iOS
    // has no such key) and only before the first keystroke.
    //
    // While the microphone is open this line carries the working phrase
    // instead, and that is the ONLY feedback in the gap it covers: a speaker
    // who stops talking waits out the gate's hangover and the end-of-turn
    // silence -- about two seconds -- before a turn exists to show anything
    // in. Without this the app looks asleep at exactly the moment the user is
    // wondering whether it heard them.

    private var footnote: some View {
        let quiet = model.listening || model.speech.speaking
        return Text(noteText)
            .appFont(.caption2)
            .foregroundStyle(quiet ? .secondary : .tertiary)
            .frame(maxWidth: .infinity)
            .animation(.easeInOut(duration: 0.2), value: quiet)
    }

    // The voice runs several times slower than the answer arrives, so the
    // transcript settles long before the sound does. Without saying so the
    // screen looks finished while it is still talking.

    private var noteText: String {
        let text: String
        if model.listening {
            text = listeningNote
        } else if model.speech.paused {
            text = "Paused"
        } else if model.speech.speaking {
            text = "Speaking…"
        } else {
            text = plainFootnote
        }
        return text
    }

    // The phrase says the app is awake; the seconds say the SPEECH arrived,
    // which is the part a speaker cannot otherwise tell. Nothing heard yet
    // reads as an invitation rather than a count of zero.
    private var listeningNote: String {
        model.heardSeconds > 0.05
            ? String(format: "%@…  heard %.1fs", model.thinkStatus,
                     model.heardSeconds)
            : model.thinkStatus + "…"
    }

    private var plainFootnote: String {
        model.typing || isOS
            ? Composer.caveat
            : "Shift+Return for a new line.  " + Composer.caveat
    }

}

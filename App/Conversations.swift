import CoreGraphics
import Foundation
import ImageIO
import LLM
import MD
import UniformTypeIdentifiers

// Read-only conversation history: project the live transcript to the store's
// Codable shape and back. Only the DISPLAY transcript is persisted (text +
// reasoning + tool rounds + tiny JPEG thumbnails) -- never the KV/GDN state, so
// a saved chat is KB and reopening it is view-only (continuation is a later
// opt-in). The store itself is ConversationStore.swift.

extension ChatModel {

    // Persist the live conversation when it is worth keeping: a real exchange
    // (at least one user turn + a reply) carrying more than trivial text. A
    // read-only view is already saved, so it never re-commits, and the title is
    // set once (a re-commit only refreshes the transcript + timestamp).
    // TODO(ctx-gate): gate on lastMetrics.ctx > ~150 tokens once the ctx is
    // tracked synchronously; the char count is a cheap stand-in that still skips
    // "Hi"/"Hello".
    func commitCurrent() {
        let chars = messages.reduce(0) { sum, m in sum + m.text.count }
        let worth = !readOnly && messages.count >= 2 && chars > 200
        if worth {
            let id = currentConversationId ?? UUID()
            currentConversationId = id
            let prior = ConversationStore.shared.load(id)
            let now = Date()
            let convo = ConversationStore.Convo(
                id: id,
                // The generated title overrides a stored instant one; absent
                // that, keep whatever was already saved.
                title: generatedTitle ?? prior?.title ?? conversationTitle(),
                created: prior?.created ?? now,
                updated: now,
                messages: messages.map { m in ChatModel.stored(m) },
                trace: traceEvents.map { e in ChatModel.storedTrace(e) })
            ConversationStore.shared.save(convo)
        }
    }

    // Open a saved conversation read-only: commit the live one first, then swap
    // the transcript in without touching the session (New Chat later resets the
    // engine, so the stale KV is harmless).
    // A BACKSTOP behind the sidebar's disabled rows, not the visible gate: a
    // turn in flight holds an index into `messages` and goes on appending to
    // it, so swapping the transcript here would redirect the reply into the
    // conversation just opened rather than stopping it.
    func openConversation(_ id: UUID) {
        commitCurrent()
        if !busy, let convo = ConversationStore.shared.load(id) {
            messages = convo.messages.map { s in ChatModel.restored(s) }
            traceEvents = (convo.trace ?? []).map { t in
                ChatModel.restoredTrace(t)
            }
            currentConversationId = id
            generatedTitle = nil
            readOnly = true
            statsLabel = ""
        }
    }

    // Delete a saved conversation; if it is the one on screen, start fresh.
    //
    // Forget WHICH conversation this was, and its transcript, before starting
    // that fresh one. New Chat commits the outgoing conversation on its way
    // out, and with the id still set that writes the one just deleted
    // straight back -- it survives its own deletion and returns to the
    // sidebar. Emptying the transcript first is what makes the commit decline
    // to save anything.
    func deleteConversation(_ id: UUID) {
        ConversationStore.shared.delete(id)
        if id == currentConversationId {
            currentConversationId = nil
            generatedTitle = nil
            messages = []
            traceEvents = []
            newChat()
        }
        sweepAttachments()
    }

    // A reopened (read-only) chat is now gone, so drop back to a live one; a
    // live chat forgets its saved copy (the next commit re-creates it).
    func clearAllConversations() {
        ConversationStore.shared.deleteAll()
        currentConversationId = nil
        if readOnly { newChat() }
        sweepAttachments()
    }

    // A kept document exists only so the conversation citing it can offer it
    // back, so one that nothing cites is garbage. Deleting a conversation is
    // when that becomes true, and sweeping the WHOLE directory then -- rather
    // than only the paths that conversation held -- also collects what no
    // per-delete hook could reach: a turn rolled back, a quit before the
    // commit, anything an older build left behind.
    //
    // The live transcript and the pending attachments are cited too. A file
    // kept THIS session is referenced by no saved conversation yet, and
    // deleting some other chat must not take it.
    //
    // Matched on FILENAME, not on path: the name carries a UUID and is unique
    // by construction, while the container's absolute path is not ours to
    // count on across an install.
    func sweepAttachments() {
        var cited = Set<String>()
        for convo in ConversationStore.shared.list {
            for message in convo.messages {
                for doc in message.docs ?? [] {
                    cited.insert(ChatModel.restoredURL(doc.path)
                        .lastPathComponent)
                }
            }
        }
        for message in messages {
            for doc in message.docs { cited.insert(doc.url.lastPathComponent) }
        }
        for doc in attachedDocs {
            if let url = doc.url { cited.insert(url.lastPathComponent) }
        }
        let fm = FileManager.default
        let kept = (try? fm.contentsOfDirectory(
            at: ChatModel.attachments,
            includingPropertiesForKeys: nil)) ?? []
        for file in kept where !cited.contains(file.lastPathComponent) {
            try? fm.removeItem(at: file)
        }
    }

    // The whole conversation assembled into one Markdown document for the
    // transcript tools (copy / export / share). Built through a MarkdownStream
    // so no internal parse entrypoint is needed from the app target.
    var transcriptDocument: Markdown.Document {
        let stream = MarkdownStream()
        for m in messages {
            let who = m.fromUser ? "**You**\n\n" : "**Gadeon**\n\n"
            stream.append(who + m.text + "\n\n")
        }
        return stream.finish()
    }

    // A short title for an export filename: the conversation's own title.
    var transcriptTitle: String { conversationTitle() }

    // The model-generated title once made; else the first turn that carries
    // WORDS, trimmed to a short line; else a timestamp.
    //
    // A dictated turn shows "Spoken, 1.9s" -- a stand-in for speech that
    // cannot be shown -- and an attachment-only turn shows nothing at all.
    // Naming a conversation after either says what the user DID rather than
    // what it was about, and every voice conversation ends up with the same
    // name. The reply is the better fallback there: it is about the subject
    // even when the question was spoken.
    func conversationTitle() -> String {
        var title = generatedTitle ?? ""
        if title.isEmpty {
            let spoke = messages.first { m in
                m.fromUser && !m.placeholder && !trimmed(m.text).isEmpty
            }
            let answered = messages.first { m in
                !m.fromUser && !trimmed(m.text).isEmpty
            }
            let line = trimmed((spoke ?? answered)?.text ?? "")
            title = line.count > 40
                ? String(line.prefix(40)) + "\u{2026}" : line
        }
        if title.isEmpty { title = ChatModel.timestampTitle() }
        return title
    }

    private func trimmed(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func timestampTitle() -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "MMM d, HH:mm"
        return f.string(from: Date())
    }

    // ---- Message <-> stored projection --------------------------------

    // A clip that lives INSIDE the app bundle is stored by name under a
    // marker, never by path. The bundle moves on every install and update, so
    // an absolute path into it is stale by the next build -- the video sample
    // would degrade to its own filename on a transcript from yesterday, which
    // is exactly the case the fallback exists to cover rather than to cause.
    //
    // Detected by prefix rather than by naming the sample, so any bundled
    // resource a later turn attaches travels the same way.
    private static let bundleMark = "bundle:"
    // A kept attachment is stored by NAME under its own marker for the same
    // reason: the Data container's absolute path is not ours to count on
    // across an install, and the name already carries a UUID.
    private static let storeMark = "store:"

    private static func storedPath(_ url: URL) -> String {
        var out = url.path
        if url.path.hasPrefix(Bundle.main.bundlePath) {
            out = bundleMark + url.lastPathComponent
        } else if url.deletingLastPathComponent().path
                    == ChatModel.attachments.path {
            out = storeMark + url.lastPathComponent
        }
        return out
    }

    // A bundled name that no longer resolves (the resource was dropped from a
    // later build) becomes a bare relative URL: it cannot exist, so the view
    // names it instead of playing it -- which is the honest answer, and
    // better than the row vanishing from a saved transcript.
    private static func restoredURL(_ stored: String) -> URL {
        var out = URL(fileURLWithPath: stored)
        if stored.hasPrefix(bundleMark) {
            let name = String(stored.dropFirst(bundleMark.count)) as NSString
            out = Bundle.main.url(forResource: name.deletingPathExtension,
                                  withExtension: name.pathExtension)
                ?? URL(fileURLWithPath: name as String)
        } else if stored.hasPrefix(storeMark) {
            out = ChatModel.attachments.appendingPathComponent(
                String(stored.dropFirst(storeMark.count)))
        }
        return out
    }

    private static func stored(_ m: Message) -> ConversationStore.Msg {
        ConversationStore.Msg(
            fromUser: m.fromUser, text: m.text, reasoning: m.reasoning,
            rounds: m.toolRounds.map { r in
                ConversationStore.Round(
                    emitted: r.emitted, label: r.label, symbol: r.symbol,
                    args: r.args, result: r.result)
            },
            images: m.images.compactMap { cg in jpeg(cg) },
            loopStopped: m.loopStopped,
            clips: m.clips.map { url in storedPath(url) },
            docs: m.docs.map { ref in
                ConversationStore.StoredDoc(path: storedPath(ref.url),
                                            bytes: ref.bytes)
            })
    }

    // Rebuild a display Message from its stored projection. The Markdown docs
    // are sealed from the text so a reopened turn renders exactly like a live
    // one; tool-round ids are re-indexed (the display never needs the originals).
    private static func restored(_ s: ConversationStore.Msg) -> Message {
        var m = Message(fromUser: s.fromUser, text: s.text)
        m.reasoning = s.reasoning
        m.loopStopped = s.loopStopped
        m.images = s.images.compactMap { data in decodeImage(data) }
        // The path is restored whether or not the file survived; the view
        // checks, so a clip that is gone becomes a named row rather than a
        // player that cannot play.
        m.clips = (s.clips ?? []).map { p in restoredURL(p) }
        m.docs = (s.docs ?? []).map { d in
            ChatModel.DocRef(url: restoredURL(d.path), bytes: d.bytes)
        }
        m.toolRounds = s.rounds.enumerated().map { pair in
            ToolRound(id: pair.offset, emitted: pair.element.emitted,
                      label: pair.element.label, symbol: pair.element.symbol,
                      args: pair.element.args, result: pair.element.result)
        }
        m.answerStream.append(s.text)
        m.answerDoc = m.answerStream.finish()
        if !s.reasoning.isEmpty {
            m.reasoningStream.append(s.reasoning)
            m.reasoningDoc = m.reasoningStream.finish()
        }
        return m
    }

    // A small JPEG of a transcript image for persistence (~KB, not the full
    // bitmap); mirrors traceThumbnail but from a CGImage.
    private static func jpeg(_ cg: CGImage) -> Data? {
        var out: Data? = nil
        let buf = NSMutableData()
        if let dst = CGImageDestinationCreateWithData(
            buf, UTType.jpeg.identifier as CFString, 1, nil) {
            CGImageDestinationAddImage(dst, cg, [
                kCGImageDestinationLossyCompressionQuality: 0.7,
            ] as CFDictionary)
            if CGImageDestinationFinalize(dst) { out = buf as Data }
        }
        return out
    }

    private static func decodeImage(_ data: Data) -> CGImage? {
        CGImageSourceCreateWithData(data as CFData, nil).flatMap { src in
            CGImageSourceCreateImageAtIndex(src, 0, nil)
        }
    }

    // ---- TraceEvent <-> trimmed stored projection ---------------------

    // Keep tool results / injects / diagnostics; drop the multi-KB render /
    // decode / prefill payloads. Timings + metrics survive, so the reopened
    // graph still draws; only the row payload expansion goes empty.
    private static let traceTextKinds: Set<TraceEvent.Kind> =
        [.toolCall, .toolResult, .inject, .diag]

    private static func storedTrace(_ e: TraceEvent)
        -> ConversationStore.Trace {
        let text = traceTextKinds.contains(e.kind)
            ? String(e.text.prefix(4000)) : ""
        return ConversationStore.Trace(
            kind: e.kind.rawValue, t0: e.t0, t1: e.t1, ctx: e.ctx,
            tokens: e.tokens, summary: e.summary, text: text)
    }

    private static func restoredTrace(_ t: ConversationStore.Trace)
        -> TraceEvent {
        TraceEvent(kind: TraceEvent.Kind(rawValue: t.kind) ?? .diag,
                   t0: t.t0, t1: t.t1, ctx: t.ctx, tokens: t.tokens,
                   summary: t.summary, text: t.text)
    }
}

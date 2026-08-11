import CoreGraphics
import Foundation
import ImageIO
import LLM
import MD
import UniformTypeIdentifiers

extension ChatModel {

    // TODO(ctx-gate): gate on lastMetrics.ctx instead of the char-count
    // stand-in.
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
                title: generatedTitle ?? prior?.title ?? conversationTitle(),
                created: prior?.created ?? now,
                updated: now,
                messages: messages.map { m in ChatModel.stored(m) },
                trace: traceEvents.map { e in ChatModel.storedTrace(e) })
            ConversationStore.shared.save(convo)
        }
    }

    // `!busy` guards a turn in flight: it holds an index into `messages`
    // and keeps appending, so swapping the transcript here would redirect
    // the reply into the conversation just opened.
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

    // Order matters: id/messages must clear before newChat(), or its own
    // exit-commit re-saves the conversation just deleted.
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

    func clearAllConversations() {
        ConversationStore.shared.deleteAll()
        currentConversationId = nil
        if readOnly { newChat() }
        sweepAttachments()
    }

    // A live conversation and pending attachments are cited too, so a
    // same-session file is not swept before it is saved.
    //
    // Matched on FILENAME, not path: the name carries a UUID and is
    // unique by construction; the container's path is not stable across
    // an install.
    func sweepAttachments() {
        var cited = Set<String>()
        for convo in ConversationStore.shared.list {
            for doc in convo.messages.flatMap({ m in m.docs ?? [] }) {
                if let top = ChatModel.topName(
                    ChatModel.restoredURL(doc.path)) {
                    cited.insert(top)
                }
            }
        }
        for doc in messages.flatMap({ m in m.docs }) {
            if let top = ChatModel.topName(doc.url) { cited.insert(top) }
        }
        for url in attachedDocs.compactMap({ d in d.url }) {
            if let top = ChatModel.topName(url) { cited.insert(top) }
        }
        let fm = FileManager.default
        let kept = (try? fm.contentsOfDirectory(
            at: ChatModel.attachments,
            includingPropertiesForKeys: nil)) ?? []
        for entry in kept where !cited.contains(entry.lastPathComponent) {
            try? fm.removeItem(at: entry)
        }
    }

    var transcriptDocument: Markdown.Document {
        let stream = MarkdownStream()
        for m in messages {
            let who = m.fromUser ? "**You**\n\n" : "**Gadeon**\n\n"
            stream.append(who + m.text + "\n\n")
        }
        return stream.finish()
    }

    var transcriptTitle: String { conversationTitle() }

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

    // Stored by NAME under this marker, never by path: the bundle's path
    // changes on every install/update.
    private static let bundleMark = "bundle:"
    // Stored by NAME under this marker too: the Data container's path is
    // not stable across an install either.
    private static let storeMark = "store:"

    private static var storeRoot: String {
        ChatModel.attachments.path + "/"
    }

    private static func storedPath(_ url: URL) -> String {
        var out = url.path
        if url.path.hasPrefix(Bundle.main.bundlePath) {
            out = bundleMark + url.lastPathComponent
        } else if url.path.hasPrefix(storeRoot) {
            out = storeMark + String(url.path.dropFirst(storeRoot.count))
        }
        return out
    }

    // The top-level name under `storeRoot` identifies one kept document,
    // whether it is a directory or (an older) bare file.
    private static func topName(_ url: URL) -> String? {
        var out: String? = nil
        if url.path.hasPrefix(storeRoot) {
            out = String(url.path.dropFirst(storeRoot.count))
                .split(separator: "/").first.map(String.init)
        }
        return out
    }

    // A bundled resource no longer present resolves to a URL that cannot
    // exist; the view names it rather than failing to play it.
    private static func restoredURL(_ stored: String) -> URL {
        var out = URL(fileURLWithPath: stored)
        if stored.hasPrefix(bundleMark) {
            let name = String(stored.dropFirst(bundleMark.count)) as NSString
            out = Bundle.main.url(forResource: name.deletingPathExtension,
                                  withExtension: name.pathExtension)
                ?? URL(fileURLWithPath: name as String)
        } else if stored.hasPrefix(storeMark) {
            out = URL(fileURLWithPath:
                storeRoot + String(stored.dropFirst(storeMark.count)))
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
                                            bytes: ref.bytes,
                                            short: ref.short)
            })
    }

    // Docs are sealed from the raw text so a reopened turn renders like a
    // live one; tool-round ids are re-indexed since the display never
    // needs the originals.
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
            ChatModel.DocRef(url: restoredURL(d.path), bytes: d.bytes,
                             short: d.short ?? false)
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

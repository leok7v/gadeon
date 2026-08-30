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

    func openConversation(_ id: UUID) {
        commitCurrent()
        let store = ConversationStore.shared
        if !busy, let convo = store.load(id) ?? store.loadTrashed(id) {
            messages = convo.messages.map { s in ChatModel.restored(s) }
            traceEvents = (convo.trace ?? []).map { t in
                ChatModel.restoredTrace(t)
            }
            currentConversationId = id
            generatedTitle = nil
            followupHint = ""
            readOnly = true
            statsLabel = ""
        }
    }

    func renameConversation(_ id: UUID, to title: String) {
        let name = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !name.isEmpty, var convo = ConversationStore.shared.load(id) {
            convo.title = name
            ConversationStore.shared.save(convo)
            if id == currentConversationId { generatedTitle = name }
        }
    }

    func deleteConversation(_ id: UUID) {
        ConversationStore.shared.trash(id)
        closeIfShowing(id)
        sweepAttachments()
    }

    func restoreConversation(_ id: UUID) {
        ConversationStore.shared.restore(id)
    }

    func deleteForever(_ id: UUID) {
        ConversationStore.shared.deleteForever(id)
        closeIfShowing(id)
        sweepAttachments()
    }

    func emptyTrash() {
        let gone = ConversationStore.shared.trashed.map { convo in convo.id }
        ConversationStore.shared.emptyTrash()
        for id in gone { closeIfShowing(id) }
        sweepAttachments()
    }

    // The trash can be read before it is emptied, so a destroyed
    // conversation may be the one on screen.
    private func closeIfShowing(_ id: UUID) {
        if id == currentConversationId {
            currentConversationId = nil
            generatedTitle = nil
            messages = []
            traceEvents = []
            newChat()
        }
    }

    func clearAllConversations() {
        ConversationStore.shared.trashAll()
        currentConversationId = nil
        if readOnly { newChat() }
        sweepAttachments()
    }

    func sweepAttachments() {
        var cited = Set<String>()
        let store = ConversationStore.shared
        // TRASHED conversations cite their attachments too, or a restore
        // 30 days later returns a transcript whose documents are gone.
        for convo in store.list + store.trashed {
            for doc in convo.messages.flatMap({ m in m.docs ?? [] }) {
                if let name = ChatModel.keptDocName(
                    ChatModel.restoredURL(doc.path)) {
                    cited.insert(name)
                }
            }
        }
        for doc in messages.flatMap({ m in m.docs }) {
            if let name = ChatModel.keptDocName(doc.url) { cited.insert(name) }
        }
        for url in attachedDocs.compactMap({ d in d.url }) {
            if let name = ChatModel.keptDocName(url) { cited.insert(name) }
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
            title = TopicTitle.from(messages.map { m in m.text })
        }
        if title.isEmpty { title = ChatModel.timestampTitle() }
        return title
    }

    private static func timestampTitle() -> String {
        let f = DateFormatter()
        f.locale = Locale.current
        f.dateStyle = .none
        f.timeStyle = .short
        return f.string(from: Date())
    }

    private static let bundleNameMark = "bundle:"
    private static let storeRelativeMark = "store:"

    private static var storeRoot: String {
        ChatModel.attachments.path + "/"
    }

    private static func storedPath(_ url: URL) -> String {
        var out = url.path
        if url.path.hasPrefix(Bundle.main.bundlePath) {
            out = bundleNameMark + url.lastPathComponent
        } else if url.path.hasPrefix(storeRoot) {
            out = storeRelativeMark +
                String(url.path.dropFirst(storeRoot.count))
        }
        return out
    }

    private static func keptDocName(_ url: URL) -> String? {
        var out: String? = nil
        if url.path.hasPrefix(storeRoot) {
            out = String(url.path.dropFirst(storeRoot.count))
                .split(separator: "/").first.map(String.init)
        }
        return out
    }

    private static func restoredURL(_ stored: String) -> URL {
        var out = URL(fileURLWithPath: stored)
        if stored.hasPrefix(bundleNameMark) {
            let name =
                String(stored.dropFirst(bundleNameMark.count)) as NSString
            out = Bundle.main.url(forResource: name.deletingPathExtension,
                                  withExtension: name.pathExtension)
                ?? URL(fileURLWithPath: name as String)
        } else if stored.hasPrefix(storeRelativeMark) {
            out = URL(fileURLWithPath: storeRoot +
                String(stored.dropFirst(storeRelativeMark.count)))
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

    private static func restored(_ s: ConversationStore.Msg) -> Message {
        var m = Message(fromUser: s.fromUser, text: s.text)
        m.reasoning = s.reasoning
        m.loopStopped = s.loopStopped
        m.images = s.images.compactMap { data in decodeImage(data) }
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

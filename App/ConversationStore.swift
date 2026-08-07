import Foundation

// Read-only-from-the-caller's-view conversation persistence: one JSON file
// per conversation under Application Support/conversations/<id>.json.
// `list` is the in-memory index (newest `updated` first), kept in sync with
// disk on every save/delete rather than re-scanned each time.

@MainActor @Observable final class ConversationStore {

    static let shared = ConversationStore()

    struct Round: Codable {
        let emitted: String
        let label: String
        let symbol: String
        let args: String
        let result: String?
    }

    struct Msg: Codable {
        let fromUser: Bool
        let text: String
        let reasoning: String
        let rounds: [Round]
        let images: [Data]
        let loopStopped: Bool
        // Attached video PATHS, not bytes: a conversation is kilobytes and a
        // phone clip is tens of megabytes. OPTIONAL because a synthesized
        // Codable does not fall back to a property's default for a missing
        // key -- it throws -- so every record written before this field would
        // stop decoding. Same reason `trace` is optional.
        let clips: [String]?
        // Attached DOCUMENTS: where the file was kept, and how much text came
        // out of it. Optional for the same reason `clips` is.
        let docs: [StoredDoc]?
    }

    struct StoredDoc: Codable {
        let path: String
        let bytes: Int
        // Optional for the same reason `docs` is: records written before it
        // must keep decoding.
        let short: Bool?
    }

    // Trimmed trace for debug-on-reopen: timings + metrics + tool results, but
    // NOT the multi-KB verbatim render/decode payloads (dropped in `text`).
    struct Trace: Codable {
        let kind: String
        let t0: Date
        let t1: Date
        let ctx: Int
        let tokens: Int
        let summary: String
        let text: String
    }

    struct Convo: Codable, Identifiable {
        let id: UUID
        var title: String
        let created: Date
        var updated: Date
        var messages: [Msg]
        var trace: [Trace]? = nil
    }

    private(set) var list: [Convo] = []

    // Word counts per conversation, maintained beside `list` rather than
    // stored in `Convo`: that type is Codable and one file per conversation,
    // so a new property would rewrite the on-disk format of every chat ever
    // saved. Kept here, the index costs nothing on disk and is rebuilt from
    // the same two places the in-memory list is.
    //
    // It exists so the sidebar filter scores a small dictionary per keystroke
    // instead of rescanning the full text of every conversation.
    private(set) var words: [UUID: [String: Int]] = [:]

    private init() {
        reload()
    }

    func reload() {
        let dir = conversationsDir()
        let names = (try? FileManager.default
            .contentsOfDirectory(atPath: dir.path)) ?? []
        var convos: [Convo] = []
        for name in names where name.hasSuffix(".json") {
            if let convo = decodeConvo(dir.appendingPathComponent(name)) {
                convos.append(convo)
            }
        }
        list = convos.sorted { a, b in a.updated > b.updated }
        words = [:]
        for convo in list { words[convo.id] = Self.wordCounts(convo) }
    }

    func save(_ convo: Convo) {
        if let data = try? JSONEncoder().encode(convo) {
            try? data.write(to: fileURL(convo.id))
        }
        upsert(convo)
    }

    func delete(_ id: UUID) {
        try? FileManager.default.removeItem(at: fileURL(id))
        list.removeAll { convo in convo.id == id }
        words[id] = nil
    }

    func deleteAll() {
        for convo in list {
            try? FileManager.default.removeItem(at: fileURL(convo.id))
        }
        list = []
        words = [:]
    }

    func load(_ id: UUID) -> Convo? {
        decodeConvo(fileURL(id))
    }

    private func upsert(_ convo: Convo) {
        var next = list.filter { existing in existing.id != convo.id }
        next.append(convo)
        list = next.sorted { a, b in a.updated > b.updated }
        words[convo.id] = Self.wordCounts(convo)
    }

    // The title's words count for several, because the title is what the
    // conversation is ABOUT: a word in it should outrank the same word said
    // once in passing halfway down a transcript.
    private static let titleWeight = 5

    private static func wordCounts(_ convo: Convo) -> [String: Int] {
        var counts: [String: Int] = [:]
        add(convo.title, titleWeight, &counts)
        for m in convo.messages { add(m.text, 1, &counts) }
        return counts
    }

    private static func add(_ text: String, _ weight: Int,
                            _ counts: inout [String: Int]) {
        for token in text.lowercased().split(whereSeparator: { c in
            !c.isLetter && !c.isNumber
        }) {
            counts[String(token), default: 0] += weight
        }
    }

    private func decodeConvo(_ url: URL) -> Convo? {
        (try? JSONDecoder().decode(Convo.self, from: Data(contentsOf: url)))
    }

    private func conversationsDir() -> URL {
        let fm = FileManager.default
        let base = fm.urls(for: .applicationSupportDirectory,
                           in: .userDomainMask)[0]
            .appendingPathComponent("conversations", isDirectory: true)
        try? fm.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private func fileURL(_ id: UUID) -> URL {
        conversationsDir().appendingPathComponent("\(id.uuidString).json")
    }
}

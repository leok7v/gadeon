import Foundation

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
        // A synthesized Codable throws on a missing key instead of using
        // the property default, so every added field must be Optional.
        let clips: [String]?
        let docs: [StoredDoc]?
    }

    struct StoredDoc: Codable {
        let path: String
        let bytes: Int
        let short: Bool?
    }

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

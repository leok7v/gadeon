import Foundation

// Matching for the sidebar's conversation filter: does a saved conversation
// answer this query, and if the answer is not visible in its title, what part
// of it did. The sidebar owns the field and the list; this owns only the
// question, so a filtered row stays the same row.

enum ConversationSearch {

    // Shortest query worth filtering on. One character matches most of the
    // history and reads as the list breaking rather than narrowing.
    static let minimumQuery = 2

    // Every word has to land somewhere, in any order -- the words a person
    // remembers about a conversation are rarely adjacent in it, so matching
    // the query as one literal string finds "dark matter" and misses "dark
    // energy and matter".
    //
    // Split exactly as ConversationStore indexes, on anything that is not a
    // letter or a digit. The two MUST agree: the index holds "matter", so a
    // query tokenized any other way could type "matter," and match nothing.
    static func words(_ query: String) -> [String] {
        query.lowercased()
            .split(whereSeparator: { c in !c.isLetter && !c.isNumber })
            .map(String.init)
    }

    static func active(_ query: String) -> Bool {
        let all = words(query)
        return !all.isEmpty
            && all.reduce(0) { sum, word in sum + word.count }
                >= minimumQuery
    }

    // The conversations a query answers, best first. Every query word has to
    // land somewhere -- a filter that keeps rows matching ANY word barely
    // narrows anything -- and what orders the survivors is how WELL they
    // landed: a whole word beats a word that merely starts the same way,
    // which beats one buried inside another, and a word said nine times
    // beats one said once.
    static func rank(_ list: [ConversationStore.Convo],
                     _ index: [UUID: [String: Int]],
                     _ query: String) -> [ConversationStore.Convo] {
        let wanted = words(query).map { word in word.lowercased() }
        var scored: [(convo: ConversationStore.Convo, score: Int)] = []
        scored.reserveCapacity(list.count)
        for convo in list {
            let counts = index[convo.id] ?? [:]
            var total = 0
            var landed = 0
            for want in wanted {
                let gain = score(want, counts)
                total += gain
                if gain > 0 { landed += 1 }
            }
            if landed == wanted.count { scored.append((convo, total)) }
        }
        // Recency is part of the comparator so it reports no ties, which
        // makes the order a property of the key alone.
        return scored
            .sorted { a, b in
                a.score == b.score
                    ? a.convo.updated > b.convo.updated
                    : a.score > b.score
            }
            .map { pair in pair.convo }
    }

    private static func score(_ want: String,
                              _ counts: [String: Int]) -> Int {
        var total = 0
        for (word, count) in counts {
            if word == want {
                total += 100 * count
            } else if word.hasPrefix(want) {
                total += 10 * count
            } else if word.contains(want) {
                total += count
            }
        }
        return total
    }

    // Where the match was found, for the row's second line -- but ONLY for a
    // word the title does not already show. A row whose own title carries the
    // query needs no explanation, and repeating it there costs the timestamp
    // for nothing.
    static func reason(_ convo: ConversationStore.Convo,
                       _ query: String) -> String? {
        var result: String? = nil
        for word in words(query) where result == nil {
            if convo.title.range(of: word,
                                 options: .caseInsensitive) == nil {
                result = found(word, in: convo)
            }
        }
        return result
    }

    // The text around the first occurrence of `word` in the transcript. Only
    // the messages: the caller asks this exactly when the title does NOT
    // carry the word, which is what makes the answer worth a row.
    private static func found(_ word: String,
                              in convo: ConversationStore.Convo) -> String? {
        var result: String? = nil
        for m in convo.messages where result == nil {
            if let r = m.text.range(of: word, options: .caseInsensitive) {
                result = snippet(m.text, r)
            }
        }
        return result
    }

    private static func snippet(_ text: String,
                                _ range: Range<String.Index>) -> String {
        let pad = 24
        let lo = text.index(range.lowerBound, offsetBy: -pad,
                            limitedBy: text.startIndex) ?? text.startIndex
        let hi = text.index(range.upperBound, offsetBy: pad,
                            limitedBy: text.endIndex) ?? text.endIndex
        var out = String(text[lo ..< hi])
            .replacingOccurrences(of: "\n", with: " ")
        if lo > text.startIndex { out = "\u{2026}" + out }
        if hi < text.endIndex { out += "\u{2026}" }
        return out
    }
}

import Foundation

// Markdown answer text -> the sentences a voice should actually say.
//
// STREAMING: push() takes whatever just arrived and returns only the segments
// that are COMPLETE, so speech starts on sentence one rather than at the end
// of the turn. Generation outruns speech several times over, so the queue
// behind this fills on its own; nothing here waits.
//
// Two block kinds are DESCRIBED rather than read. A fenced code block read
// aloud is unbearable, and a table read cell by cell is worse -- the geometry
// that made it a table is exactly what speech cannot carry. Lists keep their
// items and lose their bullets, which is what a person reading aloud does.

public struct SpeakableText {

    private var pending = ""      // the trailing partial line
    private var prose = ""        // speakable text awaiting a sentence end
    private var inFence = false
    private var fenceLang = ""
    private var tableRows = 0
    // Whether `pending` still begins where its line does. A sentence taken
    // out mid-line leaves the rest of that line behind, and what follows is
    // no longer a place a heading hash or a list marker can appear.
    private var atLineStart = true

    public init() {}

    // Whatever arrived; the complete segments it completed.

    public mutating func push(_ chunk: String) -> [String] {
        pending += chunk
        var out: [String] = []
        while let nl = pending.firstIndex(of: "\n") {
            let line = String(pending[pending.startIndex..<nl])
            pending = String(pending[pending.index(after: nl)...])
            take(line, into: &out)
            atLineStart = true
        }
        // A sentence can complete mid-line, and waiting for the newline would
        // hold a whole paragraph of speech back -- which is the usual case,
        // since a model streams a paragraph faster than it ends one.
        if !inFence && tableRows == 0 {
            let split = SpeakableText.splitOffSentences(pending)
            if !split.spoken.isEmpty {
                // Through the same shaping a whole line gets. Appending it
                // raw is how emphasis, links and numerals reached the voice
                // unconverted for every sentence that did not happen to end
                // exactly at a line break.
                let text = atLineStart
                    ? SpeakableText.inlineText(split.spoken)
                    : SpeakableText.inlineOnly(split.spoken)
                if !text.isEmpty {
                    if !prose.isEmpty { prose += " " }
                    prose += text
                }
                pending = split.rest
                atLineStart = false
                harvest(&out)
            }
        }
        return out
    }

    // End of turn: everything still held, complete or not.

    public mutating func finish() -> [String] {
        var out: [String] = []
        if !pending.isEmpty {
            let line = pending
            pending = ""
            take(line, into: &out)
        }
        atLineStart = true
        closeTable(&out)
        if inFence {
            out.append(SpeakableText.codeSummary(fenceLang))
            inFence = false
        }
        flushProse(&out)
        return out
    }

    // One whole line, routed by the block it belongs to.

    private mutating func take(_ line: String, into out: inout [String]) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        // The tail of a line whose opening was already spoken: a fence or a
        // table pipe can only start a line, so only the inline shaping is
        // owed here.
        if !atLineStart {
            let text = SpeakableText.inlineOnly(trimmed)
            if !text.isEmpty {
                if !prose.isEmpty { prose += " " }
                prose += text
            }
            harvest(&out)
        } else if inFence {
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                inFence = false
                out.append(SpeakableText.codeSummary(fenceLang))
            }
        } else if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
            closeTable(&out)
            flushProse(&out)
            inFence = true
            fenceLang = String(trimmed.drop(while: { c in
                c == "`" || c == "~"
            })).trimmingCharacters(in: .whitespaces)
        } else if trimmed.hasPrefix("|") {
            if tableRows == 0 { flushProse(&out) }
            tableRows += 1
        } else {
            closeTable(&out)
            if trimmed.isEmpty {
                flushProse(&out)
            } else {
                add(trimmed, into: &out)
            }
        }
    }

    // A heading or a list item is its own breath: it rarely ends in a
    // terminator, and glued to the next line it would run on.

    private mutating func add(_ trimmed: String,
                              into out: inout [String]) {
        let standalone = trimmed.hasPrefix("#")
            || SpeakableText.listMarkerLength(trimmed) > 0
        if standalone { flushProse(&out) }
        let text = SpeakableText.inlineText(trimmed)
        if !text.isEmpty {
            if !prose.isEmpty { prose += " " }
            prose += text
        }
        if standalone {
            flushProse(&out)
        } else {
            harvest(&out)
        }
    }

    private mutating func harvest(_ out: inout [String]) {
        let split = SpeakableText.splitOffSentences(prose)
        if !split.spoken.isEmpty {
            for s in SpeakableText.sentences(split.spoken) { out.append(s) }
            prose = split.rest
        }
    }

    private mutating func flushProse(_ out: inout [String]) {
        let text = prose.trimmingCharacters(in: .whitespacesAndNewlines)
        prose = ""
        if !text.isEmpty {
            for s in SpeakableText.sentences(text) { out.append(s) }
        }
    }

    private mutating func closeTable(_ out: inout [String]) {
        if tableRows > 0 {
            // The header and its dashed rule are not rows a listener counts.
            let rows = max(1, tableRows - 2)
            out.append(rows == 1 ? "A table."
                                 : "A table of \(rows) rows.")
            tableRows = 0
        }
    }

    private static func codeSummary(_ lang: String) -> String {
        let named = lang.split(separator: " ").first.map(String.init) ?? ""
        return named.isEmpty ? "A code block." : "A \(named) code block."
    }

    // ---- text shaping ---------------------------------------------------

    // How many leading characters are a list marker ("- ", "* ", "3. ").

    static func listMarkerLength(_ s: String) -> Int {
        let c = Array(s)
        var n = 0
        if c.count >= 2 && (c[0] == "-" || c[0] == "*" || c[0] == "+")
            && c[1] == " " {
            n = 2
        } else {
            var i = 0
            while i < c.count && c[i].isNumber { i += 1 }
            if i > 0 && i + 1 < c.count && (c[i] == "." || c[i] == ")")
                && c[i + 1] == " " {
                n = i + 2
            }
        }
        return n
    }

    // Markdown line -> the words in it. Emphasis, heading hashes, list
    // markers, backticks and link targets carry no sound, so they go; a
    // link's TEXT stays, because that is what a reader would say.

    static func inlineText(_ line: String) -> String {
        var body = line
        while body.hasPrefix("#") { body.removeFirst() }
        while body.hasPrefix(">") { body.removeFirst() }
        body = body.trimmingCharacters(in: .whitespaces)
        let marker = listMarkerLength(body)
        if marker > 0 { body = String(body.dropFirst(marker)) }
        return inlineOnly(body)
    }

    // The shaping that applies anywhere in a line: emphasis, links, numerals.
    // A heading hash or a list marker is NOT handled here -- those only mean
    // anything at the very start of a line, and stripping them mid-sentence
    // would eat real text.

    static func inlineOnly(_ line: String) -> String {
        let body = line
        var out = ""
        var i = body.startIndex
        while i < body.endIndex {
            let c = body[i]
            if c == "[" {
                // "[text](url)" -> "text"; a bare "[" is kept as itself.
                let link = linkText(body, from: i)
                out += link.text
                i = link.next
            } else if c == "!" && body.index(after: i) < body.endIndex
                && body[body.index(after: i)] == "[" {
                i = body.index(after: i)
            } else if c == "*" || c == "_" || c == "`" || c == "~" {
                i = body.index(after: i)
            } else {
                out.append(c)
                i = body.index(after: i)
            }
        }
        // A rule row ("---") is punctuation for the eye only.
        let bare = out.trimmingCharacters(in: .whitespaces)
        let ruleOnly = !bare.isEmpty
            && bare.allSatisfy { c in c == "-" || c == "=" || c == "|" }
        return ruleOnly ? "" : SpokenNumbers.expand(dropEnumerators(bare))
    }

    // "Split into: 1. Definitions, 2. Differences, and 3. Evidence." -- an
    // ordered list written inline. The numbers are scaffolding a listener
    // does not need, and each "N." also reads as a full stop, which chops one
    // sentence into four. Both go.
    //
    // TWO of them are required before any are dropped: one "25. " is far more
    // likely to be a sentence that ended in a number.

    static func dropEnumerators(_ line: String) -> String {
        let c = Array(line)
        var marks: [(at: Int, next: Int)] = []
        var i = 0
        while i < c.count {
            let standalone = i == 0 || !c[i - 1].isLetter && !c[i - 1].isNumber
            if standalone && c[i].isNumber {
                var j = i
                while j < c.count && c[j].isNumber { j += 1 }
                let short = j - i <= 3
                if short && j + 1 < c.count && c[j] == "."
                    && c[j + 1] == " " {
                    marks.append((i, j + 2))
                    i = j + 2
                } else {
                    i = j
                }
            } else {
                i += 1
            }
        }
        var out = line
        if marks.count >= 2 {
            out = ""
            var k = 0
            var at = 0
            while at < c.count {
                if k < marks.count && marks[k].at == at {
                    at = marks[k].next
                    k += 1
                } else {
                    out.append(c[at])
                    at += 1
                }
            }
        }
        return out
    }

    private static func linkText(_ s: String, from: String.Index)
        -> (text: String, next: String.Index) {
        var text = "["
        var next = s.index(after: from)
        if let close = s[from...].firstIndex(of: "]") {
            let inner = String(s[s.index(after: from)..<close])
            var after = s.index(after: close)
            if after < s.endIndex && s[after] == "(",
               let paren = s[after...].firstIndex(of: ")") {
                after = s.index(after: paren)
            }
            text = inner
            next = after
        }
        return (text, next)
    }

    // The prefix of `s` that ends on a sentence boundary, and the remainder.

    static func splitOffSentences(_ s: String)
        -> (spoken: String, rest: String) {
        let c = Array(s)
        var cut = 0
        var i = 0
        while i < c.count {
            if isSentenceEnd(c, i) { cut = i + 1 }
            i += 1
        }
        var spoken = ""
        var rest = s
        if cut > 0 {
            spoken = String(c[0..<cut])
            rest = String(c[cut...])
        }
        return (spoken, rest)
    }

    // A terminator that really ends a sentence: followed by space or end,
    // and neither a decimal point nor a known abbreviation's full stop.

    private static func isSentenceEnd(_ c: [Character], _ i: Int) -> Bool {
        var ends = c[i] == "." || c[i] == "!" || c[i] == "?"
        if ends {
            let next = i + 1 < c.count ? c[i + 1] : " "
            ends = next == " " || next == "\n" || next == "\"" || next == ")"
            // "3.14" -- a digit on both sides is a number, not a stop.
            if ends && c[i] == "." && i > 0 && c[i - 1].isNumber
                && i + 1 < c.count && c[i + 1].isNumber {
                ends = false
            }
            if ends && c[i] == "." { ends = !endsAbbreviation(c, i) }
        }
        return ends
    }

    // Titles and month names take a full stop mid-sentence, and a single
    // initial ("J. R. R.") does too.
    private static let abbreviations: Set<String> = [
        "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "etc",
        "eg", "ie", "approx", "fig", "no", "vol", "al",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
    ]

    private static func endsAbbreviation(_ c: [Character], _ i: Int) -> Bool {
        var start = i
        while start > 0 && (c[start - 1].isLetter || c[start - 1] == ".") {
            start -= 1
        }
        let word = String(c[start..<i]).replacingOccurrences(of: ".",
                                                             with: "")
        return word.count == 1
            || abbreviations.contains(word.lowercased())
    }

    // Split a settled run into individual sentences, so the queue holds one
    // breath per entry and a barge-in never has to cut mid-sentence.

    static func sentences(_ s: String) -> [String] {
        let c = Array(s)
        var out: [String] = []
        var start = 0
        var i = 0
        while i < c.count {
            if isSentenceEnd(c, i) {
                let piece = String(c[start...i])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !piece.isEmpty { out.append(piece) }
                start = i + 1
            }
            i += 1
        }
        if start < c.count {
            let tail = String(c[start...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !tail.isEmpty { out.append(tail) }
        }
        return out
    }
}

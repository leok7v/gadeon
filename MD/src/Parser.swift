import Foundation

// Batch parser. Adapted from md.too `src/MarkdownParser.swift`, extended
// with per-column table alignment and an internal `blocks` seam the
// streaming parser reuses so streaming and batch results cannot diverge.
// Reference link definitions ride on a task-local so the streaming parser
// can inject the definitions it has accumulated so far.

extension Markdown {

    @TaskLocal static var currentRefs: [String: URL] = [:]
    // Parse-time math switch. `$inline$` / `$$display$$` are rendered to
    // Unicode only when true. Threaded via a task-local so the streaming
    // parser (which has no style context) can set it per stream.
    @TaskLocal static var mathEnabled: Bool = true

    public static func parse(_ source: String,
                             math: Bool = true) -> Document {
        let raw = source
            .replacingOccurrences(of: "\r\n", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        let (lines, refs) = stripLinkDefinitions(raw)
        let bs = Markdown.$mathEnabled.withValue(math) {
            Markdown.$currentRefs.withValue(refs) { blocks(lines) }
        }
        var items: [Document.Item] = []
        for (i, b) in bs.enumerated() {
            items.append(Document.Item(id: i, block: b))
        }
        return Document(items: items)
    }

    // The shared block engine. Reads `currentRefs` for reference links.
    static func blocks(_ lines: [String]) -> [Block] {
        blockSpans(lines).map { pair in pair.block }
    }

    // Same engine, recording the START line of each block. The streaming
    // parser uses the spans to seal every block except the last (open)
    // one and to advance its consumed-line cursor.
    static func blockSpans(_ lines: [String])
        -> [(start: Int, block: Block)] {
        var out: [(start: Int, block: Block)] = []
        var i = 0
        while i < lines.count {
            let start = i
            let line = lines[i]
            if isFence(line) {
                out.append((start, consumeFenced(lines, &i)))
            } else if isHeading(line) {
                out.append((start, consumeHeading(lines, &i)))
            } else if isHR(line) {
                let lastRule: Bool
                if let lb = out.last?.block, case .rule = lb {
                    lastRule = true
                } else {
                    lastRule = false
                }
                if !lastRule { out.append((start, .rule)) }
                i += 1
            } else if isTableStart(lines, i) {
                out.append((start, consumeTable(lines, &i)))
            } else if isQuoteStart(line) {
                out.append((start, consumeQuote(lines, &i)))
            } else if isListStart(line) {
                out.append((start, consumeList(lines, &i)))
            } else if isIndentedCode(line) {
                out.append((start, consumeIndentedCode(lines, &i)))
            } else if line.trimmedOuter().isEmpty {
                i += 1
            } else if let img = imageBlock(line) {
                out.append((start, img))
                i += 1
            } else {
                out.append((start, consumeParagraph(lines, &i)))
            }
        }
        return out
    }

    static func stripLinkDefinitions(_ raw: [String])
        -> (lines: [String], refs: [String: URL]) {
        var refs: [String: URL] = [:]
        var out: [String] = []
        var inFence = false
        var fenceMarker = ""
        for line in raw {
            let trimmed = line.trimmedLeading()
            if inFence {
                if trimmed.hasPrefix(fenceMarker) { inFence = false }
                out.append(line)
            } else if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                inFence = true
                fenceMarker = String(trimmed.prefix(3))
                out.append(line)
            } else if let parsed = parseLinkDefinition(line) {
                refs[parsed.label] = parsed.url
            } else {
                out.append(line)
            }
        }
        return (out, refs)
    }

    static func parseLinkDefinition(_ line: String)
        -> (label: String, url: URL)? {
        var result: (String, URL)? = nil
        let t = line.trimmedLeading()
        if t.hasPrefix("["), let close = t.dropFirst().firstIndex(of: "]") {
            let rest = t.dropFirst()
            let label = String(rest[..<close]).trimmedOuter()
            let after = rest[rest.index(after: close)...]
            if !label.isEmpty, after.hasPrefix(":") {
                var rhs = String(after.dropFirst()).trimmedOuter()
                if let space = rhs.firstIndex(of: " ") {
                    rhs = String(rhs[..<space])
                }
                if rhs.hasPrefix("<"), rhs.hasSuffix(">") {
                    rhs = String(rhs.dropFirst().dropLast())
                }
                if let url = URL(string: rhs) {
                    result = (refKey(label), url)
                }
            }
        }
        return result
    }

    static func refKey(_ label: String) -> String {
        label.lowercased().split(whereSeparator: { c in
            c == " " || c == "\t" || c == "\n"
        }).joined(separator: " ")
    }

    // Inline.

    static func inline(_ raw: String) -> AttributedString {
        let withRefs = substituteRefs(raw)
        let normalized = normalizeBreaks(withRefs)
        let segs = mathEnabled ? TeX.split(normalized) : []
        let hasMath = segs.contains { seg in
            if case .math = seg { return true } else { return false }
        }
        var out = hasMath ? inlineWithMath(segs)
                          : parseInlineMarkdown(normalized)
        applyUnderlineTags(&out)
        return out
    }

    // Each math span becomes a private-use sentinel, the WHOLE line is
    // markdown-parsed once (so emphasis wrapping math -- **\(i^i\)** --
    // still pairs across the span), then the sentinels are swapped for the
    // rendered math runs, folding in any emphasis the sentinel picked up.
    private static let mathMark = "\u{F8FF}"

    static func inlineWithMath(_ segs: [TeX.Segment]) -> AttributedString {
        var text = ""
        var maths: [AttributedString] = []
        for seg in segs {
            switch seg {
                case .text(let s): text += s
                case .math(let s, let display):
                    text += mathMark + String(maths.count) + mathMark
                    maths.append(TeX.render(s, display: display))
            }
        }
        var out = parseInlineMarkdown(text)
        for (i, math) in maths.enumerated() {
            let token = mathMark + String(i) + mathMark
            if let r = out.range(of: token) {
                var m = math
                let picked = out[r].inlinePresentationIntent ?? []
                let own = m.inlinePresentationIntent ?? []
                m.inlinePresentationIntent = picked.union(own)
                out.replaceSubrange(r, with: m)
            }
        }
        return out
    }

    static func parseInlineMarkdown(_ s: String) -> AttributedString {
        let opts = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible)
        var result = AttributedString(s)
        if let parsed = try? AttributedString(markdown: s, options: opts) {
            result = parsed
        }
        return result
    }

    static func normalizeBreaks(_ s: String) -> String {
        let lines = s
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var out: [String] = []
        for (idx, line) in lines.enumerated() {
            let last = idx == lines.count - 1
            let hardBreak = line.hasSuffix("  ")
            let trimmed = hardBreak ? String(line.dropLast(2)) : line
            if hardBreak { out.append(trimmed + "\n") }
            else if last { out.append(trimmed) }
            else { out.append(trimmed + " ") }
        }
        return out.joined()
    }

    static func applyUnderlineTags(_ a: inout AttributedString) {
        var keep = true
        while keep {
            if let open = a.range(of: "<u>", options: .caseInsensitive) {
                let tail = a[open.upperBound...]
                if let close = tail.range(of: "</u>",
                                          options: .caseInsensitive) {
                    var sub = a[open.upperBound..<close.lowerBound]
                    sub.underlineStyle = .single
                    a.replaceSubrange(open.lowerBound..<close.upperBound,
                                      with: sub)
                } else {
                    a.removeSubrange(open)   // hide a partial tag streaming
                }
            } else {
                keep = false
            }
        }
    }

    static func substituteRefs(_ s: String) -> String {
        let refs = Markdown.currentRefs
        var result = s
        if !refs.isEmpty {
            result = applyRefPattern(
                result, pattern: "(!?)\\[([^\\]\\n]+)\\]\\[([^\\]\\n]*)\\]",
                hasLabelGroup: true, refs: refs)
            result = applyRefPattern(
                result, pattern: "(!?)\\[([^\\]\\n]+)\\](?![\\[\\(:])",
                hasLabelGroup: false, refs: refs)
        }
        return result
    }

    static func applyRefPattern(_ s: String, pattern: String,
                                hasLabelGroup: Bool,
                                refs: [String: URL]) -> String {
        var result = s
        if let re = try? NSRegularExpression(pattern: pattern) {
            let ns = s as NSString
            let matches = re.matches(
                in: s, range: NSRange(location: 0, length: ns.length))
            if !matches.isEmpty {
                let mutable = NSMutableString(string: s)
                for m in matches.reversed() {
                    let bang = ns.substring(with: m.range(at: 1))
                    let text = ns.substring(with: m.range(at: 2))
                    var labelSrc = text
                    if hasLabelGroup, m.numberOfRanges > 3,
                       m.range(at: 3).location != NSNotFound {
                        let g3 = ns.substring(with: m.range(at: 3))
                        if !g3.isEmpty { labelSrc = g3 }
                    }
                    if let url = refs[refKey(labelSrc)] {
                        let rep = "\(bang)[\(text)](\(url.absoluteString))"
                        mutable.replaceCharacters(in: m.range, with: rep)
                    }
                }
                result = mutable as String
            }
        }
        return result
    }

    // Headings.

    static func isHeading(_ s: String) -> Bool {
        var result = false
        let t = s.trimmedOuter()
        let n = t.prefix { c in c == "#" }.count
        if n >= 1 && n <= 6 {
            let rest = t.dropFirst(n)
            result = rest.hasPrefix(" ") || rest.isEmpty
        }
        return result
    }

    static func consumeHeading(_ lines: [String], _ i: inout Int) -> Block {
        let t = lines[i].trimmedOuter()
        let n = t.prefix { c in c == "#" }.count
        let body = String(t.dropFirst(n)).trimmedOuter()
        i += 1
        return .heading(level: n, text: inline(body))
    }

    // Rules.

    static func isHR(_ s: String) -> Bool {
        var result = false
        let t = s.trimmedOuter()
        if t.count >= 3, let c = t.first, c == "-" || c == "*" || c == "_" {
            result = t.allSatisfy { ch in ch == c || ch == " " || ch == "\t" }
        }
        return result
    }

    // Fenced and indented code.

    static func isFence(_ s: String) -> Bool {
        let t = s.trimmedLeading()
        return t.hasPrefix("```") || t.hasPrefix("~~~")
    }

    static func consumeFenced(_ lines: [String], _ i: inout Int) -> Block {
        let raw = lines[i]
        let t = raw.trimmedLeading()
        let fence = String(t.prefix(3))
        let lang = String(t.dropFirst(3)).trimmedOuter()
        let indent = raw.count - t.count
        let pad = String(repeating: " ", count: indent)
        i += 1
        var body: [String] = []
        var done = false
        while i < lines.count, !done {
            let line = lines[i]
            let trimmed = line.trimmedLeading()
            if trimmed.hasPrefix(fence) {
                done = true
            } else if indent > 0, line.hasPrefix(pad) {
                body.append(String(line.dropFirst(indent)))
            } else {
                body.append(line)
            }
            i += 1
        }
        let language = lang.isEmpty ? nil : lang
        return .code(language: language, text: body.joined(separator: "\n"))
    }

    static func isIndentedCode(_ s: String) -> Bool {
        var result = false
        if !s.trimmedOuter().isEmpty {
            result = s.hasPrefix("    ") || s.hasPrefix("\t")
        }
        return result
    }

    static func consumeIndentedCode(_ lines: [String],
                                    _ i: inout Int) -> Block {
        var body: [String] = []
        var done = false
        while i < lines.count, !done {
            let line = lines[i]
            if line.trimmedOuter().isEmpty {
                body.append("")
                i += 1
            } else if line.hasPrefix("    ") {
                body.append(String(line.dropFirst(4)))
                i += 1
            } else if line.hasPrefix("\t") {
                body.append(String(line.dropFirst(1)))
                i += 1
            } else {
                done = true
            }
        }
        while let last = body.last, last.isEmpty { body.removeLast() }
        return .code(language: nil, text: body.joined(separator: "\n"))
    }

    // Blockquotes.

    static func isQuoteStart(_ s: String) -> Bool {
        leadingSpaces(s) <= 3 && s.trimmedLeading().hasPrefix(">")
    }

    static func consumeQuote(_ lines: [String], _ i: inout Int) -> Block {
        var inner: [String] = []
        var collecting = true
        while i < lines.count, collecting {
            let line = lines[i]
            if isQuoteStart(line) {
                var t = line.trimmedLeading()
                t = String(t.dropFirst())
                if t.hasPrefix(" ") { t = String(t.dropFirst()) }
                inner.append(t)
                i += 1
            } else if !line.trimmedOuter().isEmpty,
                      isLazyContinuation(line) {
                inner.append(line.trimmedLeading())
                i += 1
            } else {
                collecting = false
            }
        }
        return .quote(blocks(inner))
    }

    // Lists.

    static func isListStart(_ s: String) -> Bool { listMarker(s) != nil }

    static func listMarker(_ line: String)
        -> (label: String, sig: Character, offset: Int, rest: String)? {
        var result: (String, Character, Int, String)? = nil
        let leading = line.prefix { c in c == " " }.count
        if leading <= 3 {
            let afterIndent = line.dropFirst(leading)
            if let first = afterIndent.first,
               first == "-" || first == "*" || first == "+" {
                result = afterMarker(
                    afterIndent.dropFirst(), leading: leading,
                    markerWidth: 1, label: "\u{2022}", sig: first)
            } else {
                let digits = afterIndent.prefix { c in c.isNumber }
                let afterDigits = afterIndent.dropFirst(digits.count)
                if !digits.isEmpty, digits.count <= 9,
                   let delim = afterDigits.first,
                   delim == "." || delim == ")" {
                    result = afterMarker(
                        afterDigits.dropFirst(), leading: leading,
                        markerWidth: digits.count + 1,
                        label: String(digits) + ".", sig: delim)
                }
            }
        }
        return result
    }

    static func afterMarker(_ tail: Substring, leading: Int,
                            markerWidth: Int, label: String,
                            sig: Character)
        -> (label: String, sig: Character, offset: Int, rest: String)? {
        var result: (String, Character, Int, String)? = nil
        let spaces = tail.prefix { c in c == " " }.count
        let blankRest = tail.allSatisfy { c in c == " " }
        if blankRest {
            result = (label, sig, leading + markerWidth + 1, "")
        } else if spaces >= 1 {
            let n = spaces >= 5 ? 1 : spaces
            result = (label, sig, leading + markerWidth + n,
                      String(tail.dropFirst(n)))
        }
        return result
    }

    static func consumeList(_ lines: [String], _ i: inout Int) -> Block {
        var items: [ListItem] = []
        var tight = true
        var sig: Character? = nil
        var done = false
        while i < lines.count, !done {
            if let m = listMarker(lines[i]), sig == nil || m.sig == sig {
                sig = m.sig
                var body: [String] = []
                let (checked, rest) = stripTaskMarker(m.rest)
                body.append(rest)
                i += 1
                if collectItemBody(lines, &i, m.offset, &body) {
                    tight = false
                }
                items.append(ListItem(marker: m.label, checked: checked,
                                      blocks: blocks(body)))
                let gap = interItemGap(lines, &i, sig: m.sig)
                if gap.loose { tight = false }
                if gap.ended { done = true }
            } else {
                done = true
            }
        }
        return .list(items: items, tight: tight)
    }

    static func stripTaskMarker(_ s: String)
        -> (checked: Bool?, rest: String) {
        var result: (Bool?, String) = (nil, s)
        if s.hasPrefix("[ ] ") {
            result = (false, String(s.dropFirst(4)))
        } else if s.hasPrefix("[x] ") || s.hasPrefix("[X] ") {
            result = (true, String(s.dropFirst(4)))
        }
        return result
    }

    static func collectItemBody(_ lines: [String], _ i: inout Int,
                                _ offset: Int,
                                _ body: inout [String]) -> Bool {
        var loose = false
        var lastWasBlank = false
        var collecting = true
        while i < lines.count, collecting {
            let line = lines[i]
            if line.trimmedOuter().isEmpty {
                var j = i
                while j < lines.count, lines[j].trimmedOuter().isEmpty {
                    j += 1
                }
                if j < lines.count, leadingSpaces(lines[j]) >= offset {
                    var k = i
                    while k < j { body.append(""); k += 1 }
                    i = j
                    loose = true
                    lastWasBlank = true
                } else {
                    collecting = false
                }
            } else if leadingSpaces(line) >= offset {
                body.append(dropIndent(line, offset))
                i += 1
                lastWasBlank = false
            } else if !lastWasBlank, isLazyContinuation(line) {
                body.append(line.trimmedLeading())
                i += 1
                lastWasBlank = false
            } else {
                collecting = false
            }
        }
        return loose
    }

    static func interItemGap(_ lines: [String], _ i: inout Int,
                             sig: Character) -> (loose: Bool, ended: Bool) {
        var result: (loose: Bool, ended: Bool) = (false, false)
        let before = i
        while i < lines.count, lines[i].trimmedOuter().isEmpty { i += 1 }
        if i > before {
            if i < lines.count, let n = listMarker(lines[i]), n.sig == sig {
                result = (true, false)
            } else {
                i = before
                result = (false, true)
            }
        }
        return result
    }

    static func isLazyContinuation(_ line: String) -> Bool {
        !(isHeading(line) || isHR(line) || isFence(line) ||
          isQuoteStart(line) || isListStart(line))
    }

    static func leadingSpaces(_ s: String) -> Int {
        var n = 0
        var done = false
        for c in s {
            if !done {
                if c == " " { n += 1 }
                else if c == "\t" { n += 4 - (n % 4) }
                else { done = true }
            }
        }
        return n
    }

    static func dropIndent(_ s: String, _ n: Int) -> String {
        var dropped = 0
        var idx = s.startIndex
        var done = false
        while idx < s.endIndex, !done {
            let c = s[idx]
            if c == " ", dropped < n {
                dropped += 1
                idx = s.index(after: idx)
            } else if c == "\t", dropped < n {
                dropped += 4 - (dropped % 4)
                idx = s.index(after: idx)
            } else {
                done = true
            }
        }
        return String(s[idx...])
    }

    // Tables (GFM) with per-column alignment.

    static func isTableRow(_ s: String) -> Bool {
        let t = s.trimmedOuter()
        return t.contains("|") && !t.isEmpty
    }

    static func isTableSeparator(_ s: String) -> Bool {
        var result = false
        let t = s.trimmedOuter()
        if t.contains("|"), t.contains("-") {
            result = t.allSatisfy { ch in "-:| \t".contains(ch) }
        }
        return result
    }

    static func isTableStart(_ lines: [String], _ i: Int) -> Bool {
        var result = false
        if i + 1 < lines.count {
            result = isTableRow(lines[i]) && isTableSeparator(lines[i + 1])
        }
        return result
    }

    static func consumeTable(_ lines: [String], _ i: inout Int) -> Block {
        var headers: [String] = []
        var rows: [[String]] = []
        var alignments: [Alignment] = []
        if i < lines.count, isTableRow(lines[i]) {
            headers = parseRow(lines[i])
            i += 1
        }
        if i < lines.count, isTableSeparator(lines[i]) {
            alignments = parseAlignments(lines[i])
            i += 1
        }
        while i < lines.count, isTableRow(lines[i]) {
            rows.append(parseRow(lines[i]))
            i += 1
        }
        return .table(headers: headers, rows: rows,
                      alignments: alignments)
    }

    static func parseRow(_ s: String) -> [String] {
        let pipes = CharacterSet(charactersIn: "|")
        let t = s.trimmedOuter().trimmingCharacters(in: pipes)
        return t.split(separator: "|", omittingEmptySubsequences: false)
                .map { p in p.trimmingCharacters(in: .whitespaces) }
    }

    static func parseAlignments(_ s: String) -> [Alignment] {
        parseRow(s).map { cell in
            let t = cell.trimmingCharacters(in: .whitespaces)
            let left = t.hasPrefix(":")
            let right = t.hasSuffix(":")
            let a: Alignment
            if left && right { a = .center }
            else if right { a = .right }
            else if left { a = .left }
            else { a = .none }
            return a
        }
    }

    // Images.

    static let imagePattern =
        #"^!\[([^\]]*)\]\(([^\s\)]+)(?:\s+"[^"]*")?\)"#
        + #"\s*(?:\{([^}]*)\})?\s*$"#

    static let imageLineRegex: NSRegularExpression? =
        try? NSRegularExpression(pattern: imagePattern)

    static func imageBlock(_ line: String) -> Block? {
        var result: Block? = nil
        if let re = imageLineRegex {
            let trimmed = line.trimmedOuter()
            let ns = trimmed as NSString
            let range = NSRange(location: 0, length: ns.length)
            if let m = re.firstMatch(in: trimmed, options: [],
                                     range: range) {
                let alt = ns.substring(with: m.range(at: 1))
                let raw = ns.substring(with: m.range(at: 2))
                if let url = URL(string: raw) {
                    var width: CGFloat?
                    var height: CGFloat?
                    if m.numberOfRanges >= 4,
                       m.range(at: 3).location != NSNotFound {
                        let attrs = ns.substring(with: m.range(at: 3))
                        (width, height) = parseDimensions(attrs)
                    }
                    result = .image(alt: alt, url: url,
                                    width: width, height: height)
                }
            }
        }
        return result
    }

    static func parseDimensions(_ attrs: String) -> (CGFloat?, CGFloat?) {
        var width: CGFloat?
        var height: CGFloat?
        let pat = #"(width|height)\s*=\s*(\d+(?:\.\d+)?)(?:px)?"#
        if let re = try? NSRegularExpression(pattern: pat,
                                             options: .caseInsensitive) {
            let ns = attrs as NSString
            let full = NSRange(location: 0, length: ns.length)
            re.enumerateMatches(in: attrs, options: [],
                                range: full) { m, _, _ in
                if let m, m.numberOfRanges == 3 {
                    let key = ns.substring(with: m.range(at: 1)).lowercased()
                    let val = ns.substring(with: m.range(at: 2))
                    if let n = Double(val) {
                        if key == "width" { width = CGFloat(n) }
                        else if key == "height" { height = CGFloat(n) }
                    }
                }
            }
        }
        return (width, height)
    }

    // Paragraphs.

    static func consumeParagraph(_ lines: [String],
                                 _ i: inout Int) -> Block {
        var body: [String] = []
        var done = false
        while i < lines.count, !done {
            let line = lines[i]
            let blank = line.trimmedOuter().isEmpty
            let other = isHeading(line) || isHR(line) || isFence(line) ||
                        isTableStart(lines, i) || isQuoteStart(line) ||
                        isListStart(line) || isIndentedCode(line) ||
                        imageBlock(line) != nil
            if blank || other { done = true }
            else { body.append(line); i += 1 }
        }
        return .paragraph(inline(body.joined(separator: "\n")))
    }
}

extension String {

    func trimmedOuter() -> String {
        trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func trimmedLeading() -> String {
        var i = startIndex
        while i < endIndex, self[i] == " " || self[i] == "\t" {
            i = index(after: i)
        }
        return String(self[i...])
    }
}

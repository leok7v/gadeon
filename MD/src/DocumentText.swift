import Foundation

// Flattens a document into one NSAttributedString for the single selectable
// surface. Adapted from md.too `src/DocumentText.swift`. Code / table / image
// runs carry atomic attributes so selection snaps around them. Tables are
// built per platform (NSTextTable on macOS, tab stops on iOS).
//
// Main actor because what it builds is platform text objects: an AppKit
// attachment cell is main-actor isolated, and a laid-out formula holds the
// shared math font, which is not safe to hand to a second thread.

@MainActor enum DocumentText {

    static func attributed(from document: Markdown.Document,
                           style: MarkdownStyle,
                           images: [URL: PlatformImage] = [:])
        -> NSAttributedString {
        let m = NSMutableAttributedString()
        for item in document.items {
            m.append(render(item.block, style: style, images: images))
        }
        return m
    }

    static func render(_ block: Markdown.Block, style: MarkdownStyle,
                       images: [URL: PlatformImage]) -> NSAttributedString {
        let result: NSAttributedString
        switch block {
            case .paragraph(let attr):
                result = paragraph(attr, style: style)
            case .heading(let level, let attr):
                result = heading(level: level, text: attr, style: style)
            case .code(let lang, let text):
                result = code(language: lang, text: text, style: style)
            case .quote(let inner):
                result = quote(inner, style: style, images: images)
            case .list(let items, let tight):
                result = list(items: items, tight: tight, depth: 0,
                              style: style, images: images)
            case .table(let headers, let rows, let aligns):
                result = table(headers: headers, rows: rows,
                               alignments: aligns, style: style,
                               images: images)
            case .math(let tex):
                result = math(tex, style: style)
            case .rule:
                result = rule(style: style)
            case .image(let alt, let url, let w, let h):
                result = image(alt: alt, url: url, width: w, height: h,
                               style: style, images: images)
        }
        return result
    }

    static func bodyFont(_ style: MarkdownStyle) -> PlatformFont {
        FontRole.body(style.bodySize).platformFont
    }

    // The narrowest this document can be drawn before a table or a formula
    // is asked for less room than its content can occupy. One text view holds
    // the whole document, so there is no per-block escape here the way the
    // block renderer has: the answer is a single width for everything, and
    // the caller scrolls sideways when the viewport is smaller. Zero for a
    // document with neither, which is the common case and leaves the text
    // width-aligned to its container.

    static func minimumWidth(of document: Markdown.Document,
                             style: MarkdownStyle) -> CGFloat {
        minimumWidth(of: document.items.map { item in item.block },
                     style: style)
    }

    static func minimumWidth(of blocks: [Markdown.Block],
                             style: MarkdownStyle) -> CGFloat {
        var widest: CGFloat = 0
        for block in blocks {
            let w = minimumWidth(ofBlock: block, style: style)
            if w > widest { widest = w }
        }
        return widest
    }

    private static func minimumWidth(ofBlock block: Markdown.Block,
                                     style: MarkdownStyle) -> CGFloat {
        var result: CGFloat = 0
        switch block {
            case .table(let headers, let rows, _):
                result = tableMinimumWidth(headers: headers, rows: rows,
                                           style: style)
            case .math(let tex):
                result = mathMinimumWidth(tex, style: style)
            case .quote(let inner):
                result = indented(minimumWidth(of: inner, style: style),
                                  by: 18)
            case .list(let items, _):
                for item in items {
                    let w = indented(minimumWidth(of: item.blocks,
                                                  style: style), by: 20)
                    if w > result { result = w }
                }
            default:
                result = 0
        }
        return result
    }

    // A formula has no line breaks to give and the single surface has no way
    // to scroll one on its own, so the surface has to be wide enough to hold
    // it whole -- the same bargain the tables strike.
    //
    // Wide enough for the copy button too. The paragraph is centred, so the
    // slack is split between the two margins and a gutter on the right costs
    // the same on the left; without it, a formula that exactly fills the
    // surface leaves the button sitting on top of it.

    private static func mathMinimumWidth(_ tex: String,
                                         style: MarkdownStyle) -> CGFloat {
        let size = TeX.displaySize(body: bodyFont(style).pointSize)
        var result: CGFloat = 0
        if let layout = TeX.layout(tex, size: size) {
            result = ceil(layout.width) + 8 + copyButtonGutter * 2
        }
        return result
    }

    // An indent only widens a document that had something to widen it; a
    // quote full of prose still asks for nothing.

    private static func indented(_ inner: CGFloat,
                                 by amount: CGFloat) -> CGFloat {
        inner > 0 ? inner + amount : 0
    }

    // Widest token a column must be able to hold, measured on the text that
    // will be DRAWN rather than the markdown that was typed: a link shows its
    // label, not its href, and a cell holding an image shows no words at all.
    // Measuring the source instead turns one image URL into a demand for two
    // thousand points.
    //
    // Measured against the WIDEST face the cell could end up in, not the one
    // it probably will. A run marked as code becomes monospaced and one marked
    // strong becomes bold, either of which outgrows the plain body face, and a
    // token that outgrows its column is exactly what this number prevents.

    static func longestWordWidth(_ cell: String,
                                 font: PlatformFont) -> CGFloat {
        var widest: CGFloat = 0
        let faces = [platformBoldItalicFont(of: font, bold: true,
                                            italic: true),
                     monoFont(at: font.pointSize)]
        for word in renderedText(of: cell).split(separator: " ") {
            let ns = String(word) as NSString
            for face in faces {
                let w = ns.size(withAttributes: [.font: face]).width
                if w > widest { widest = w }
            }
        }
        return widest
    }

    private static func renderedText(of cell: String) -> String {
        var result = TeX.scriptsToUnicode(cell)
        if let first = Markdown.parse(cell).items.first {
            switch first.block {
                case .paragraph(let a): result = String(a.characters)
                case .image: result = ""
                default: break
            }
        }
        return result
    }

    static func columnMinimums(headers: [String], rows: [[String]],
                               cols: Int,
                               style: MarkdownStyle) -> [CGFloat] {
        let body = bodyFont(style)
        let bold = boldFont(of: body)
        var out = [CGFloat](repeating: 0, count: cols)
        for c in 0..<cols {
            var widest: CGFloat = 0
            if c < headers.count {
                let w = longestWordWidth(headers[c], font: bold)
                if w > widest { widest = w }
            }
            for row in rows where c < row.count {
                let w = longestWordWidth(row[c], font: body)
                if w > widest { widest = w }
            }
            out[c] = ceil(widest)
        }
        return out
    }

    static func tableCell(_ text: String, base: PlatformFont,
                          style: MarkdownStyle,
                          images: [URL: PlatformImage]) -> NSAttributedString {
        let parsed = Markdown.parse(text)
        let m = NSMutableAttributedString()
        if let first = parsed.items.first {
            fillCell(first.block, text: text, base: base, style: style,
                     images: images, into: m)
        }
        return m
    }

    private static func fillCell(_ block: Markdown.Block, text: String,
                                 base: PlatformFont, style: MarkdownStyle,
                                 images: [URL: PlatformImage],
                                 into m: NSMutableAttributedString) {
        switch block {
            case .image(let alt, let url, let w, let h):
                appendImage(alt: alt, url: url, width: w, height: h,
                            base: base, images: images, into: m)
            case .paragraph(let attr):
                translateInline(attr, base: base, style: style, into: m)
            default:
                m.append(NSAttributedString(
                    string: text,
                    attributes: [.font: base,
                                 .foregroundColor: platformDefaultTextColor]))
        }
    }

    private static func appendImage(alt: String, url: URL, width: CGFloat?,
                                    height: CGFloat?, base: PlatformFont,
                                    images: [URL: PlatformImage],
                                    into m: NSMutableAttributedString) {
        if let img = images[url] {
            let attachment = NSTextAttachment()
            attachment.image = img
            attachment.bounds = imageBounds(img, width: width, height: height)
            m.append(NSAttributedString(attachment: attachment))
        } else {
            let label = alt.isEmpty ? url.absoluteString : alt
            m.append(NSAttributedString(
                string: "[Image: \(label)]",
                attributes: [.font: base,
                             .foregroundColor: platformSecondaryColor]))
        }
    }

    private static func quote(_ blocks: [Markdown.Block],
                              style: MarkdownStyle,
                              images: [URL: PlatformImage])
        -> NSAttributedString {
        let m = NSMutableAttributedString()
        for inner in blocks {
            m.append(render(inner, style: style, images: images))
        }
        let full = NSRange(location: 0, length: m.length)
        m.enumerateAttribute(.paragraphStyle, in: full,
                             options: []) { value, range, _ in
            let merged = NSMutableParagraphStyle()
            if let existing = value as? NSParagraphStyle {
                merged.setParagraphStyle(existing)
            }
            merged.headIndent += 18
            merged.firstLineHeadIndent += 18
            m.addAttribute(.paragraphStyle, value: merged, range: range)
        }
        m.addAttribute(.backgroundColor,
                       value: platformWhite(0.5, alpha: 0.06), range: full)
        return m
    }

    private static func list(items: [Markdown.ListItem], tight: Bool,
                             depth: Int, style: MarkdownStyle,
                             images: [URL: PlatformImage])
        -> NSAttributedString {
        let m = NSMutableAttributedString()
        let indent = CGFloat(depth + 1) * 20
        let para = NSMutableParagraphStyle()
        para.headIndent = indent
        para.firstLineHeadIndent = indent - 20
        para.tabStops = [NSTextTab(textAlignment: .left, location: indent)]
        para.paragraphSpacing = tight ? 2 : 8
        para.paragraphSpacingBefore = tight ? 2 : 4
        for item in items {
            m.append(listItem(item, para: para, depth: depth,
                              style: style, images: images))
        }
        return m
    }

    private static func listItem(_ item: Markdown.ListItem,
                                 para: NSParagraphStyle, depth: Int,
                                 style: MarkdownStyle,
                                 images: [URL: PlatformImage])
        -> NSAttributedString {
        let marker = item.checked.map { c in
            c ? "\u{2611}" : "\u{2610}"
        } ?? item.marker
        let base = bodyFont(style)
        let line = NSMutableAttributedString(
            string: "\(marker)\t",
            attributes: [.font: base,
                         .foregroundColor: platformSecondaryColor,
                         .paragraphStyle: para])
        var headHandled = false
        if let first = item.blocks.first, case .paragraph(let attr) = first {
            let body = NSMutableAttributedString()
            translateInline(attr, base: base, style: style, into: body)
            body.addAttribute(.paragraphStyle, value: para,
                              range: NSRange(location: 0, length: body.length))
            line.append(body)
            headHandled = true
        }
        if !headHandled, let first = item.blocks.first {
            line.append(render(first, style: style, images: images))
        }
        line.append(NSAttributedString(string: "\n"))
        for rest in item.blocks.dropFirst() {
            line.append(render(rest, style: style, images: images))
        }
        return line
    }

    private static func image(alt: String, url: URL, width: CGFloat?,
                              height: CGFloat?, style: MarkdownStyle,
                              images: [URL: PlatformImage])
        -> NSAttributedString {
        let m = NSMutableAttributedString()
        appendImage(alt: alt, url: url, width: width, height: height,
                    base: bodyFont(style), images: images, into: m)
        let full = NSRange(location: 0, length: m.length)
        m.addAttribute(atomicKindKey, value: AtomicKind.image.rawValue,
                       range: full)
        m.addAttribute(atomicIdKey, value: UUID().uuidString, range: full)
        m.append(NSAttributedString(string: "\n\n"))
        return m
    }

    private static func imageBounds(_ img: PlatformImage, width: CGFloat?,
                                    height: CGFloat?) -> CGRect {
        let fit = aspectFit(intrinsicWidth: img.size.width,
                            intrinsicHeight: img.size.height,
                            explicitWidth: width, explicitHeight: height,
                            maxWidth: 320)
        return CGRect(x: 0, y: 0, width: fit.width, height: fit.height)
    }

    private static func code(language: String?, text: String,
                             style: MarkdownStyle) -> NSAttributedString {
        let font = FontRole.mono(style.codeSize).platformFont
        let highlighted = style.highlightCode
            ? Highlight.attribute(text, language: language, baseFont: font)
            : NSAttributedString(string: text, attributes: [.font: font])
        let m = NSMutableAttributedString(attributedString: highlighted)
        // A trailing newline INSIDE the tinted run so the LAST code line's
        // background paints: NSTextView draws no line-fragment background for a
        // run's final line when it abuts a plain paragraph break, so a code
        // block whose text lacks a trailing newline (e.g. ending "print(df)")
        // renders that line outside the block. One separator newline follows.
        if !text.hasSuffix("\n") {
            m.append(NSAttributedString(string: "\n", attributes: [.font: font]))
        }
        let full = NSRange(location: 0, length: m.length)
        m.addAttribute(.backgroundColor,
                       value: platformWhite(0.5, alpha: 0.10), range: full)
        m.addAttribute(atomicKindKey, value: AtomicKind.code.rawValue,
                       range: full)
        m.addAttribute(atomicIdKey, value: UUID().uuidString, range: full)
        m.addAttribute(atomicCopyKey, value: text, range: full)
        m.append(NSAttributedString(string: "\n"))
        return m
    }

    // A display sits in its own centred paragraph, carrying the TeX it came
    // from on atomicCopyKey so Copy yields the formula rather than the
    // object-replacement character an attachment would otherwise hand over.
    // Same contract as a code fence or a table, so the copy overlay needs
    // nothing new.
    //
    // A layout the engine refuses falls back to the Unicode spelling, which
    // is readable where an empty box is not.

    private static func math(_ tex: String,
                             style: MarkdownStyle) -> NSAttributedString {
        let base = bodyFont(style)
        let m = NSMutableAttributedString()
        if let layout = TeX.layout(tex,
                                   size: TeX.displaySize(body: base.pointSize)) {
            m.append(NSAttributedString(attachment: mathAttachment(layout)))
        } else {
            translateInline(TeX.render(tex, display: true), base: base,
                            style: style, into: m)
        }
        let content = NSRange(location: 0, length: m.length)
        m.addAttribute(atomicKindKey, value: AtomicKind.math.rawValue,
                       range: content)
        m.addAttribute(atomicIdKey, value: UUID().uuidString, range: content)
        m.addAttribute(atomicCopyKey, value: tex, range: content)
        m.append(NSAttributedString(string: "\n\n"))
        let para = NSMutableParagraphStyle()
        para.alignment = .center
        para.paragraphSpacing = style.blockSpacing
        para.paragraphSpacingBefore = style.blockSpacing
        m.addAttribute(.paragraphStyle, value: para,
                       range: NSRange(location: 0, length: m.length))
        return m
    }

    private static func paragraph(_ attr: AttributedString,
                                  style: MarkdownStyle) -> NSAttributedString {
        let m = NSMutableAttributedString()
        translateInline(attr, base: bodyFont(style), style: style, into: m)
        m.append(NSAttributedString(string: "\n\n"))
        return m
    }

    private static func heading(level: Int, text: AttributedString,
                                style: MarkdownStyle) -> NSAttributedString {
        let font = FontRole.heading(
            level: level, size: style.headingSize(level)).platformFont
        let m = NSMutableAttributedString()
        translateInline(text, base: font, style: style, into: m)
        // Space the heading off from whatever precedes it. A tight list ends
        // with a single newline and no blank line, so without a spacing-before
        // a heading right after a list butts against it.
        let para = NSMutableParagraphStyle()
        para.paragraphSpacingBefore = style.blockSpacing
        m.addAttribute(.paragraphStyle, value: para,
                       range: NSRange(location: 0, length: m.length))
        m.append(NSAttributedString(string: "\n\n"))
        return m
    }

    private static func rule(style: MarkdownStyle) -> NSAttributedString {
        NSAttributedString(
            string: String(repeating: "\u{2500}", count: 8) + "\n\n",
            attributes: [.font: bodyFont(style),
                         .foregroundColor: platformSecondaryColor])
    }

    static func translateInline(_ attr: AttributedString, base: PlatformFont,
                                style: MarkdownStyle,
                                into m: NSMutableAttributedString) {
        for run in attr.runs {
            let segment = String(attr[run.range].characters)
            let intent = run.inlinePresentationIntent ?? []
            var runFont = styledRunFont(intent: intent, base: base)
            var attrs: [NSAttributedString.Key: Any] = [
                .foregroundColor: platformDefaultTextColor,
            ]
            if let level = run[ScriptAttribute.self] {
                let script = scriptRunFont(level, base: runFont)
                runFont = script.font
                attrs[.baselineOffset] = script.offset
            }
            attrs[.font] = runFont
            if intent.contains(.strikethrough) {
                attrs[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
            }
            if let url = run.link { attrs[.link] = url }
            m.append(NSAttributedString(string: segment, attributes: attrs))
        }
    }
}

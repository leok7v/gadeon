import SwiftUI

// Per-block SwiftUI renderer. Identity-driven: each block carries a stable
// id, so feeding a changed document (streaming snapshot) re-renders only the
// block whose value changed. Serves both static text and streaming.

public struct MarkdownView: View {

    let document: Markdown.Document
    let style: MarkdownStyle

    public init(_ document: Markdown.Document,
                style: MarkdownStyle = .default) {
        self.document = document
        self.style = style
    }

    public init(_ source: String, style: MarkdownStyle = .default) {
        self.document = Markdown.parse(source, math: style.renderMath)
        self.style = style
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: style.blockSpacing) {
            ForEach(document.items) { item in
                BlockView(block: item.block, style: style)
            }
        }
    }
}

struct BlockView: View {

    let block: Markdown.Block
    let style: MarkdownStyle

    var body: some View {
        switch block {
            case .heading(let level, let text):
                SelectableText(text, font: headingFont(level),
                               selectable: style.selectable, bold: true)
                    .padding(.top, level <= 2 ? 6 : 2)
            case .paragraph(let text):
                SelectableText(text, font: bodyFont,
                               selectable: style.selectable)
            case .code(let language, let text):
                CodeBlock(text: text, language: language, style: style)
            case .quote(let blocks):
                QuoteBlock(blocks: blocks, style: style)
            case .list(let items, let tight):
                ListBlock(items: items, tight: tight, style: style)
            case .table(let headers, let rows, let alignments):
                TableBlock(headers: headers, rows: rows,
                           alignments: alignments, style: style)
            case .rule:
                Rectangle().fill(style.ruleColor)
                    .frame(height: 1).padding(.vertical, 4)
            case .image(let alt, let url, let width, let height):
                ImageBlock(alt: alt, url: url, width: width,
                           height: height, style: style)
        }
    }

    private var bodyFont: PlatformFont {
        FontRole.body(style.bodySize).platformFont
    }

    private func headingFont(_ level: Int) -> PlatformFont {
        FontRole.heading(level: level,
                         size: style.headingSize(level)).platformFont
    }
}

private struct QuoteBlock: View {
    let blocks: [Markdown.Block]
    let style: MarkdownStyle

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Rectangle().fill(style.quoteBar).frame(width: 3)
            VStack(alignment: .leading, spacing: style.paragraphSpacing) {
                ForEach(Array(blocks.enumerated()), id: \.offset) { pair in
                    BlockView(block: pair.element, style: style)
                }
            }
        }
    }
}

private struct ListBlock: View {
    let items: [Markdown.ListItem]
    let tight: Bool
    let style: MarkdownStyle

    var body: some View {
        let gap: CGFloat = tight ? 3 : style.paragraphSpacing + 4
        VStack(alignment: .leading, spacing: gap) {
            ForEach(Array(items.enumerated()), id: \.offset) { pair in
                row(pair.element, gap: gap)
            }
        }
    }

    private func row(_ item: Markdown.ListItem, gap: CGFloat) -> some View {
        HStack(alignment: .top, spacing: 6) {
            marker(item).frame(width: gutter, alignment: .trailing)
            VStack(alignment: .leading, spacing: gap) {
                ForEach(Array(item.blocks.enumerated()),
                        id: \.offset) { pair in
                    BlockView(block: pair.element, style: style)
                }
            }
        }
    }

    private var gutter: CGFloat {
        let widest = items.map { item in
            item.checked == nil ? item.marker.count : 1
        }.max() ?? 1
        return CGFloat(widest) * 10 + 8
    }

    @ViewBuilder
    private func marker(_ item: Markdown.ListItem) -> some View {
        if let checked = item.checked {
            Image(systemName: checked ? "checkmark.square.fill" : "square")
                .foregroundStyle(checked ? Color.accentColor
                                         : style.secondaryColor)
        } else {
            Text(item.marker).foregroundStyle(style.secondaryColor)
        }
    }
}

private struct CodeBlock: View {
    let text: String
    let language: String?
    let style: MarkdownStyle

    var body: some View {
        let font = FontRole.mono(style.codeSize).platformFont
        let ns = style.highlightCode
            ? Highlight.attribute(text, language: language, baseFont: font)
            : NSAttributedString(string: text, attributes: [.font: font])
        ScrollView(.horizontal, showsIndicators: false) {
            SelectableText(ns: ns, font: font, nowrap: true,
                           selectable: style.selectable)
                .padding(style.codePadding)
        }
        .background(RoundedRectangle(cornerRadius: style.cornerRadius)
            .fill(style.codeBackground))
        .overlay(alignment: .topTrailing) {
            CopyButton(text: text, style: style).padding(6)
        }
    }
}

private struct ImageBlock: View {
    let alt: String
    let url: URL
    let width: CGFloat?
    let height: CGFloat?
    let style: MarkdownStyle
    @State private var image: Image?
    @State private var failed = false

    var body: some View {
        Group {
            if let image {
                sized(image)
            } else {
                placeholder(failed
                    ? (alt.isEmpty ? "image unavailable" : alt)
                    : "loading\u{2026}")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .accessibilityLabel(alt)
        .task(id: url) { await load() }
    }

    @ViewBuilder
    private func sized(_ image: Image) -> some View {
        let scaled = image.resizable().scaledToFit()
        if let w = width, let h = height {
            scaled.frame(width: w, height: h, alignment: .leading)
        } else if let w = width {
            scaled.frame(maxWidth: w, alignment: .leading)
        } else if let h = height {
            scaled.frame(maxHeight: h, alignment: .leading)
        } else {
            scaled.frame(maxWidth: 320, alignment: .leading)
        }
    }

    private func placeholder(_ text: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "photo").foregroundStyle(style.secondaryColor)
            Text(text).foregroundStyle(style.secondaryColor).italic()
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: style.cornerRadius)
            .fill(style.codeBackground))
    }

    private func load() async {
        var req = URLRequest(url: url)
        req.setValue(ImagePrefetch.userAgent, forHTTPHeaderField: "User-Agent")
        let decoded = try? await URLSession.shared.data(for: req).0
        if let decoded, let img = platformDecodeImage(decoded) {
            image = img
        } else {
            failed = true
        }
    }
}

struct CopyButton: View {
    let text: String
    let style: MarkdownStyle
    @State private var copied = false

    var body: some View {
        Button(action: copy) {
            Image(systemName: copied ? "checkmark" : "doc.on.doc")
                .font(.caption)
                .foregroundStyle(style.secondaryColor)
                .padding(4)
                .background(Circle().fill(style.codeBackground))
        }
        .buttonStyle(.plain)
        .help("Copy")
    }

    private func copy() {
        platformSetClipboardString(text)
        copied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            copied = false
        }
    }
}

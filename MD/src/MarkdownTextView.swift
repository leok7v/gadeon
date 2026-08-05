import SwiftUI

// The whole document as one selectable native text surface. Drag-selection
// snaps around atomic code / table / image units (see Bridges-macOS). Images
// are prefetched and decoded, then embedded as text attachments. Pass a
// MarkdownFindController to enable Find / Find Next (force-selects matches).

public struct MarkdownTextView: View {

    let document: Markdown.Document
    let style: MarkdownStyle
    let find: MarkdownFindController?
    // The message id this surface registers under, so Find spans every bubble.
    let findId: UUID?
    // scrolls: fill the frame and scroll internally (the whole-document
    // reading / Find surface); false self-sizes so it can sit inside a chat
    // bubble the transcript scrolls, while keeping the single-surface's
    // cross-paragraph selection (drag snaps around whole tables / code).
    let scrolls: Bool
    // The sentence being read aloud right now, tinted where it appears.
    let speaking: String?
    @State private var images: [URL: PlatformImage] = [:]

    public init(_ document: Markdown.Document,
                style: MarkdownStyle = .default,
                find: MarkdownFindController? = nil,
                findId: UUID? = nil,
                scrolls: Bool = true,
                speaking: String? = nil) {
        self.document = document
        self.style = style
        self.find = find
        self.findId = findId
        self.scrolls = scrolls
        self.speaking = speaking
    }

    public init(_ source: String, style: MarkdownStyle = .default,
                find: MarkdownFindController? = nil,
                findId: UUID? = nil,
                scrolls: Bool = true,
                speaking: String? = nil) {
        self.document = Markdown.parse(source, math: style.renderMath)
        self.style = style
        self.find = find
        self.findId = findId
        self.scrolls = scrolls
        self.speaking = speaking
    }

    public var body: some View {
        SelectableText(
            ns: DocumentText.attributed(from: document, style: style,
                                        images: images),
            font: FontRole.body(style.bodySize).platformFont,
            selectable: style.selectable, scrolls: scrolls, find: find,
            findId: findId, speaking: speaking)
            // Key the fetch on the image URLs, not the whole document: a
            // streaming answer changes `document` on every appended token, and
            // keying on it would cancel and restart the fetch each time (never
            // completing for a slow URL). The URL set is stable until an image
            // actually appears or changes.
            .task(id: ImagePrefetch.collectURLs(in: document)) {
                images = await ImagePrefetch.fetchAndDecode(
                    in: document, decode: { data in
                        platformDocumentImage(data)
                    })
            }
    }
}

// The raw text on the SAME findable / selectable single surface as
// MarkdownTextView, but WITHOUT parsing -- markers like ** stay literal. Used
// when Markdown rendering is off, so Find and cross-paragraph selection keep
// working with plain text.

public struct PlainTextView: View {

    let text: String
    let style: MarkdownStyle
    let find: MarkdownFindController?
    let findId: UUID?
    let scrolls: Bool
    let speaking: String?

    public init(_ text: String, style: MarkdownStyle = .default,
                find: MarkdownFindController? = nil, findId: UUID? = nil,
                scrolls: Bool = false, speaking: String? = nil) {
        self.text = text
        self.style = style
        self.find = find
        self.findId = findId
        self.scrolls = scrolls
        self.speaking = speaking
    }

    public var body: some View {
        let font = FontRole.body(style.bodySize).platformFont
        return SelectableText(
            ns: NSAttributedString(string: text,
                attributes: [.font: font,
                             .foregroundColor: platformDefaultTextColor]),
            font: font, selectable: style.selectable,
            scrolls: scrolls, find: find, findId: findId,
            speaking: speaking)
    }
}

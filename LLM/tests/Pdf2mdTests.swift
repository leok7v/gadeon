import CoreGraphics
import CoreText
import Foundation
import Testing
@testable import LLM

// The vendored PDF converter (LLM/src/Shared/Pdf2md.swift -- see
// LLM/fixtures/pdf2md/ORIGIN.md) against pages GENERATED here, so every
// assertion has ground truth we placed rather than a score against a corpus.
//
// Generated is right HERE and wrong for quality. These pages carry the three
// layouts the app depends on -- prose, a ruled table, two columns -- and say
// whether a re-import still handles them. Whether the converter is GOOD on
// real documents is a different question, measured upstream against its own
// benchmark, and this suite deliberately does not try to answer it.
//
// The pages carry a real text layer, so `.geometry` reads their own word
// boxes and nothing is recognized. That is what makes these runs fast and
// identical on every machine, which an OCR pass would not be -- and
// `spansMatchTheTextLayer` is what keeps it honest, since a silent fall back
// to recognition would change the span count.
struct Pdf2mdTests {
    private static let width: CGFloat = 612
    private static let height: CGFloat = 792

    // PDFDocument reads a URL, so a generated page has to reach the disk.
    private static func page(_ draw: (CGContext) -> Void) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".pdf")
        var box = CGRect(x: 0, y: 0, width: width, height: height)
        let ctx = try #require(CGContext(url as CFURL, mediaBox: &box, nil))
        ctx.beginPDFPage(nil)
        draw(ctx)
        ctx.endPDFPage()
        ctx.closePDF()
        return url
    }

    // CoreText rather than a string draw: it emits real text-showing
    // operators with the font embedded, which is what puts the words in the
    // page's text layer where `.geometry` can find them.
    private static func put(_ ctx: CGContext, _ text: String,
                            _ x: CGFloat, _ y: CGFloat,
                            _ size: CGFloat = 11) {
        let font = CTFontCreateWithName("Helvetica" as CFString, size, nil)
        let attributed = NSAttributedString(
            string: text,
            attributes: [kCTFontAttributeName as NSAttributedString.Key: font])
        let line = CTLineCreateWithAttributedString(attributed)
        ctx.textPosition = CGPoint(x: x, y: y)
        CTLineDraw(line, ctx)
    }

    private static func rule(_ ctx: CGContext, _ y: CGFloat,
                             _ from: CGFloat, _ to: CGFloat) {
        ctx.setStrokeColor(gray: 0, alpha: 1)
        ctx.setLineWidth(1)
        ctx.move(to: CGPoint(x: from, y: y))
        ctx.addLine(to: CGPoint(x: to, y: y))
        ctx.strokePath()
    }

    private static func words(_ lines: [String]) -> Int {
        lines.reduce(0) { total, line in
            total + line.split(whereSeparator: { c in c.isWhitespace }).count
        }
    }

    // The converter as the app configures it, with the per-page audit kept.
    private static func convert(_ url: URL) async throws
        -> (markdown: String, audits: [Audit]) {
        var converter = Converter()
        converter.mode = .geometry
        var audits: [Audit] = []
        converter.onAudit = { _, audit in audits.append(audit) }
        let markdown = try await converter.markdown(of: url)
        return (markdown, audits)
    }

    // ---- the three layouts ------------------------------------------------

    // Two paragraphs of several lines each, not two lines and a gap: the
    // paragraph break is judged against the MEDIAN line gap, so a page has to
    // carry enough ordinary lines for that median to mean anything.
    private static let opening = [
        "Quiet harbours keep their boats in careful rows,",
        "and the tide returns them to the same moorings",
        "each evening without any argument from the crews",
        "who have long since stopped counting the seasons.",
    ]
    private static let closing = [
        "Winter closes the channel for weeks together,",
        "and the pilots take their charts home to study",
        "until the ice gives way somewhere near March.",
    ]
    private static let prose = opening + closing

    private static func prosePage() throws -> URL {
        try page { ctx in
            for (index, line) in opening.enumerated() {
                put(ctx, line, 72, 700 - CGFloat(index) * 16)
            }
            // Twice the line spacing, which is what ends a paragraph: same
            // column, new block.
            for (index, line) in closing.enumerated() {
                put(ctx, line, 72, 604 - CGFloat(index) * 16)
            }
        }
    }

    private static let header = ["Region", "Units", "Share"]
    private static let north = ["North", "1240", "18%"]
    private static let south = ["South", "980", "14%"]
    private static let columnX: [CGFloat] = [80, 260, 420]
    // A full-width line above the table, and it is LOAD-BEARING rather than
    // decoration. A bare table alone on a page is genuinely a two-column
    // layout: the gap between its second and third columns sits near the
    // middle with a real share of the words beyond it, which is exactly what
    // a page gutter looks like. Prose crossing that gap is what tells them
    // apart, and every real document has some.
    private static let caption =
        "Table 1 reports the regional distribution of units shipped and "
        + "the relative share each market took over the quarter."

    private static func tablePage() throws -> URL {
        try page { ctx in
            put(ctx, caption, 72, 730)
            rule(ctx, 700, 72, 540)
            rule(ctx, 676, 72, 540)
            rule(ctx, 628, 72, 540)
            for (index, cell) in header.enumerated() {
                put(ctx, cell, columnX[index], 682)
            }
            for (index, cell) in north.enumerated() {
                put(ctx, cell, columnX[index], 658)
            }
            for (index, cell) in south.enumerated() {
                put(ctx, cell, columnX[index], 634)
            }
        }
    }

    private static let left = [
        "Morning fog settles over the quiet valley",
        "and the road climbs slowly past the orchard",
        "before it turns and runs away westward",
    ]
    private static let right = [
        "Meridian bells ring across the lower town",
        "where the ferries wait for the evening crowd",
        "and lanterns burn along the crooked pier",
    ]

    // A gutter 50pt wide sits at x 0.498 of the page, well past pageGap and
    // inside pageMiddle, with the words split evenly either side -- which is
    // what tells a page gutter from the space between two words.
    private static func columnsPage() throws -> URL {
        try page { ctx in
            for (index, line) in left.enumerated() {
                put(ctx, line, 72, 700 - CGFloat(index) * 16, 9)
            }
            for (index, line) in right.enumerated() {
                put(ctx, line, 330, 700 - CGFloat(index) * 16, 9)
            }
        }
    }

    // ---- what a re-import must not break ----------------------------------

    // Every recognized span reaches the output exactly once. This is the
    // assertion that needs no ground truth and survives upstream tuning, and
    // it catches the quiet loss: a span inside a table's rows but outside its
    // columns simply vanishes, and nothing else would notice.
    @Test func everyLayoutAccountsForAllItsText() async throws {
        for url in [try Pdf2mdTests.prosePage(),
                    try Pdf2mdTests.tablePage(),
                    try Pdf2mdTests.columnsPage()] {
            let (_, audits) = try await Pdf2mdTests.convert(url)
            #expect(audits.count == 1)
            for audit in audits {
                #expect(audit.lost == 0,
                        "\(audit.lost) span(s) lost in \(url.lastPathComponent)")
                #expect(audit.unaccounted == 0,
                        "\(audit.unaccounted) unaccounted")
            }
            try? FileManager.default.removeItem(at: url)
        }
    }

    // The text layer is what was read, not a recognition of the raster: the
    // span count is exactly the words drawn. A fall back to Vision would move
    // this number, so it is also the tripwire on the `.geometry` path.
    @Test func spansMatchTheTextLayer() async throws {
        let url = try Pdf2mdTests.prosePage()
        let (_, audits) = try await Pdf2mdTests.convert(url)
        let audit = try #require(audits.first)
        #expect(audit.spans == Pdf2mdTests.words(Pdf2mdTests.prose))
        try? FileManager.default.removeItem(at: url)
    }

    @Test func proseKeepsItsWordsAndBreaksItsParagraph() async throws {
        let url = try Pdf2mdTests.prosePage()
        let (markdown, _) = try await Pdf2mdTests.convert(url)
        #expect(markdown.contains("careful rows"))
        #expect(markdown.contains("counting the seasons"))
        #expect(markdown.contains("near March"))
        // The wide gap is a paragraph end, so the last line is its own block.
        #expect(markdown.contains("\n\nWinter closes"))
        try? FileManager.default.removeItem(at: url)
    }

    // The rules say where the table is and where its header ends, so the
    // cells must land in their own columns rather than collapsing into prose.
    @Test func ruledTableKeepsItsCells() async throws {
        let url = try Pdf2mdTests.tablePage()
        let (markdown, _) = try await Pdf2mdTests.convert(url)
        #expect(markdown.contains("| Region | Units | Share |"),
                "header row missing:\n\(markdown)")
        #expect(markdown.contains("| North | 1240 | 18% |"),
                "first data row missing:\n\(markdown)")
        #expect(markdown.contains("| South | 980 | 14% |"),
                "second data row missing:\n\(markdown)")
        try? FileManager.default.removeItem(at: url)
    }

    // A two-column page reads one column at a time: the whole left column
    // precedes the whole right one. Read in raster order the lines would
    // interleave, which is the failure this pins.
    @Test func twoColumnsReadOneAtATime() async throws {
        let url = try Pdf2mdTests.columnsPage()
        let (markdown, _) = try await Pdf2mdTests.convert(url)
        let last = try #require(markdown.range(of: "westward"))
        let first = try #require(markdown.range(of: "Meridian"))
        #expect(last.lowerBound < first.lowerBound,
                "columns interleaved:\n\(markdown)")
        try? FileManager.default.removeItem(at: url)
    }
}

import CoreGraphics
import Foundation
import PDFKit
import Vision

enum Mode: String, CaseIterable {
    case text
    case vision
    case geometry
    case docling
}

enum ConversionError: Error, CustomStringConvertible {
    case unreadable(URL)
    case rasterFailed(Int)
    case notImplemented(Mode)

    var description: String {
        var result = ""
        switch self {
        case .unreadable(let url):
            result = "not a readable PDF with pages: \(url.path)"
        case .rasterFailed(let page):
            result = "cannot rasterize page \(page)"
        case .notImplemented(let mode):
            result = "mode '\(mode.rawValue)' is not implemented yet"
        }
        return result
    }
}

// Coordinates throughout are Vision-normalized: 0...1 across the page,
// origin bottom-left, y growing upward. Every extraction mode feeds the
// same Element stream so one emitter renders whatever they produce.
struct Fragment {
    let index: Int
    let text: String
    let frame: CGRect
}

enum Element {
    case paragraph(String)
    case table([[String]])
}

struct Page {
    let number: Int
    let elements: [Element]
}

// The result of the exactly-once check, per page. Every recognized span
// must reach the output exactly once: `lost` counts text that never
// arrived, `unaccounted` text that arrived more often than it was
// recognized. Needs no ground truth, so a consumer with no benchmark
// can still assert lost == 0 && unaccounted == 0 over any corpus.
struct Audit {
    let spans: Int
    let lost: Int
    let unaccounted: Int
}

// A line of drawn ink: `position` is y for a horizontal rule and x for
// an upright one, `span` its extent along the other axis. Rules carry
// what text geometry cannot - where a table starts and stops, and, when
// the table draws them, exactly where its rows and columns divide.
struct Ruling {
    let position: CGFloat
    let span: ClosedRange<CGFloat>
}

struct Grid {
    let rows: [ClosedRange<CGFloat>]
    let columns: [ClosedRange<CGFloat>]
    let header: Int
}

struct Converter {
    var mode: Mode = .vision
    /// Raster scale; 8pt type needs about 3x to recognize.
    var scale: CGFloat = 3
    /// Share of a median line height two fragments may differ by
    /// vertically and still count as one text band.
    var bandTolerance: CGFloat = 0.6
    /// Normalized whitespace width that separates two columns, measured
    /// on the ink profile of all rows at once. It sits near the profile
    /// resolution on purpose: a gap between columns can be little wider
    /// than a gap between words, and it is the union over rows - not the
    /// width alone - that tells them apart.
    var columnGap: CGFloat = 0.004
    /// The same width measured on text-layer boxes, which are tight to
    /// their glyphs where a recognized box overshoots them by about a
    /// seventh of its width. The whitespace between two words therefore
    /// SURVIVES in the text layer that recognition had already
    /// swallowed, and a threshold set against swallowed gaps reads
    /// ordinary word spacing as a column edge. Swept over the whole
    /// bench: 0.004/0.006/0.008/0.010/0.012/0.015 give cell F1
    /// 0.461/0.494/0.512/0.524/0.531/0.521.
    var writtenGap: CGFloat = 0.012
    /// Whitespace that separates the page's own text columns. A folio
    /// or a centred caption parked in the gutter leaves only a sliver
    /// of the corridor open, and that sliver is narrower than the gap
    /// between two words - so width alone cannot tell a gutter from a
    /// word space. What can is position and balance.
    var pageGap: CGFloat = 0.004
    /// Where across the page width a gutter may sit.
    var pageMiddle: ClosedRange<CGFloat> = 0.3...0.7
    /// Least share of the page's words on each side of a gutter.
    var pageShare: CGFloat = 0.15
    /// Ink gap that divides two table columns, measured on the page
    /// raster instead of on the recognized boxes. A word box is not its
    /// ink - it overshoots one end and clips the other - and that error
    /// is the size of the gap being judged. Zero profiles the boxes.
    var cellGap: CGFloat = 0.006
    /// Resolution of the ink profile across the page width.
    var buckets = 1024
    /// Ink darker than this counts as drawn; 250 keeps antialiasing.
    var inkLevel: UInt8 = 250
    /// Ink this dark is a glyph rather than a tint. Only glyph ink is
    /// asked to fall inside a text-layer word box: a shaded table fills
    /// whole cells with light grey that no word covers, and judged at
    /// `inkLevel` such a page looks as unexplained as a scan.
    var glyphLevel: UInt8 = 128
    /// Least share of a page's glyph ink that must fall inside a
    /// text-layer word box before `.geometry` trusts the layer. Rules
    /// and figure strokes are dark ink no word covers, so a fully
    /// described page still scores well under 1; a scan scores 0.
    var inkShare: CGFloat = 0.35
    /// Share of the page a line of ink must cover to be a rule.
    var rulingWidth: CGFloat = 0.1
    var rulingHeight: CGFloat = 0.03
    /// Rules closer than this share of the page are one boundary.
    var rulingMerge: CGFloat = 0.005
    /// How much of the shorter span two extents must share to agree.
    var spanAgreement: CGFloat = 0.5
    /// How much of the table's width a rule must cover to divide ROWS.
    /// A rule under one header group spans less and is not a boundary.
    var fullWidth: CGFloat = 0.8
    /// How many consecutive lines must share the same columns before
    /// unruled text counts as a table. Two lines align by accident far
    /// too often; three rarely do.
    var alignedRows = 3
    /// A candidate unruled table is rejected when more than this share
    /// of its filled cells run longer than `proseCellLength` characters:
    /// those are sentences, and sentences mean paragraph, not table.
    var proseCellShare: CGFloat = 0.3
    var proseCellLength = 30
    /// Share of a table's inked rows that must reach into a bucket
    /// before it stops being a column wall. Guards the corridors against
    /// a single spanning header, which crosses every column it heads.
    /// Swept over the whole bench: 0.05/0.10/0.15/0.30 give cell F1
    /// 0.405/0.420/0.417/0.375, and 0.10 dominates on every metric.
    var cellVote: CGFloat = 0.10
    /// A window holding a single text band no taller than this many
    /// line heights is a spanning header, not the end of the table. A
    /// table can never START on one otherwise: the band between the top
    /// rule and the next resolves into one column and looks like prose.
    /// Admitted only at a region's FIRST window, which is the whole of
    /// the defect: a table cannot START on a spanning header. Allowing
    /// it anywhere would let a lone caption between two stacked tables
    /// fuse them, and the height bound alone did not stop that - it cost
    /// 25 false positives and 4 points of precision.
    var spanWindow: CGFloat = 2.2
    /// Multiple of the median band gap that ends a paragraph.
    var rowGapRatio: CGFloat = 1.35
    /// Report what the page detector found, on standard error.
    var trace = false
    /// Called once per page with its span audit, independent of `trace`.
    /// The counts are computed on every page regardless, so surfacing
    /// them costs nothing; this exists because the audit is the only
    /// correctness assertion available to a caller that has no ground
    /// truth to score against.
    var onAudit: ((Int, Audit) -> Void)?

    private var wordGap: CGFloat {
        mode == .geometry ? writtenGap : columnGap
    }

    // TODO: a second output shape, for clients that run their own ViT
    // tower and text transformer rather than reading Markdown.
    //
    // Markdown flattens away exactly what such a client needs: it keeps
    // the words and discards where they sat. Emit instead, on request:
    //
    //   - the text spans as an indexed list - one entry per recognized
    //     span, carrying its string and its normalized frame - so the
    //     client can address any span by index;
    //   - the layout as structure over those indices: which spans form
    //     a cell, a row, a column, a header, a paragraph, referenced by
    //     span index rather than by copying the text;
    //   - optionally a very low resolution image of the BOX STRUCTURE
    //     alone - no glyphs, just the numbered rectangles - so a vision
    //     tower sees the geometry as an image while the language model
    //     reads the spans, and the numbering is what joins the two.
    //     Render filled rectangles rather than the page: at ~47 DPI a
    //     filled box survives where real ink turns to mush, and the
    //     boxes ARE the layout signal. Note what this implies about the
    //     question to ask - having the boxes already, the model is not
    //     being asked to find reading order, only to name roles.
    //
    // The point is that the client fuses layout and text itself; we
    // supply both halves already aligned by index, and never force the
    // geometry through a Markdown table it does not fit.

    func markdown(of url: URL) async throws -> String {
        let document = try Converter.require(PDFDocument(url: url),
                                             .unreadable(url))
        let pages = try await self.pages(of: document, url)
        return Converter.render(pages)
    }

    private func pages(of document: PDFDocument,
                       _ url: URL) async throws -> [Page] {
        var result: [Page] = []
        if document.pageCount == 0 {
            throw ConversionError.unreadable(url)
        }
        for index in 0..<document.pageCount {
            if let page = document.page(at: index) {
                let elements = try await self.elements(of: page, index + 1)
                result.append(Page(number: index + 1, elements: elements))
            }
        }
        return result
    }

    private func elements(of page: PDFPage,
                          _ number: Int) async throws -> [Element] {
        var result: [Element] = []
        switch mode {
        case .text:
            result = Converter.paragraphs(of: page.string ?? "")
        case .vision, .geometry:
            result = try await rebuilt(page, number)
        case .docling:
            throw ConversionError.notImplemented(mode)
        }
        return result
    }

    // MODE text: PDFKit's own extraction. Blank-line separated groups
    // become paragraphs; everything columnar collapses into prose.

    private static func paragraphs(of text: String) -> [Element] {
        var result: [Element] = []
        var lines: [String] = []
        for line in text.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty {
                if !lines.isEmpty {
                    result.append(.paragraph(lines.joined(separator: " ")))
                    lines = []
                }
            } else {
                lines.append(trimmed)
            }
        }
        if !lines.isEmpty {
            result.append(.paragraph(lines.joined(separator: " ")))
        }
        return result
    }

    // MODE vision: rasterize, recognize, rebuild geometry.

    private static func raster(_ page: PDFPage, _ scale: CGFloat,
                               _ number: Int) throws -> CGImage {
        let bounds = page.bounds(for: .mediaBox)
        let width = Int((bounds.width * scale).rounded())
        let height = Int((bounds.height * scale).rounded())
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue)
        if let context {
            context.setFillColor(gray: 1, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: CGFloat(width),
                                height: CGFloat(height)))
            context.scaleBy(x: scale, y: scale)
            context.translateBy(x: -bounds.origin.x, y: -bounds.origin.y)
            page.draw(with: .mediaBox, to: context)
        }
        return try Converter.require(context?.makeImage(),
                                     .rasterFailed(number))
    }

    private static func recognize(_ image: CGImage) async throws
        -> [Fragment] {
        var request = RecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        let observations = try await request.perform(on: image)
        let spans = observations.flatMap { observation -> [Fragment] in
            var result: [Fragment] = []
            if let candidate = observation.topCandidates(1).first {
                result = Converter.words(of: candidate)
            }
            return result
        }
        return spans.enumerated().map { position, span in
            Fragment(index: position, text: span.text, frame: span.frame)
        }
    }

    // MODE geometry: a digital-born PDF already states every word and
    // where it sits, so recognition only re-derives what the file says.
    // The raster is still drawn, because rules and ink corridors are
    // drawn rather than written and no text layer carries them.
    //
    // A page whose layer is missing or partial - a scan, a table pasted
    // in as an image - must not come out empty, so the words are checked
    // against the ink before they are believed. A page that fails that
    // check is then read as `.vision` reads it, THRESHOLDS INCLUDED:
    // recognized boxes overshoot their glyphs where written ones are
    // tight, and every width the layout measures is measured against the
    // boxes it will judge. Carrying the mode on the reader rather than
    // on the converter is what keeps those two in step.

    private func rebuilt(_ page: PDFPage,
                         _ number: Int) async throws -> [Element] {
        let image = try Converter.raster(page, scale, number)
        let written = mode == .geometry ? Converter.written(of: page) : []
        let share = written.isEmpty ? 0
            : Converter.explained(written, image, glyphLevel)
        let enough = !written.isEmpty && share >= inkShare
        if mode == .geometry {
            report(String(format: "page %d: text layer %d words, "
                          + "covers %.3f of the ink%@", number,
                          written.count, Double(share),
                          enough ? "" : ", recognizing instead"))
        }
        var reader = self
        var fragments = written
        if !enough {
            reader.mode = .vision
            fragments = try await Converter.recognize(image)
        }
        return reader.layout(fragments, image, number)
    }

    // `page.string` and `page.selection(for:)` share one index space,
    // and `characterBounds(at:)` does not: PDFKit synthesizes a space or
    // a newline wherever the content stream jumps, and a synthesized
    // character owns no glyph. So a character walk drifts by one per
    // synthesized separator - silently, and only on the pages that have
    // them. Selecting the range a word occupies in the string asks
    // PDFKit for the box in its own terms and never has to guess which
    // separators were real.

    private static func written(of page: PDFPage) -> [Fragment] {
        let bounds = page.bounds(for: .mediaBox)
        let units = Array((page.string ?? "").utf16)
        var result: [Fragment] = []
        var from = -1
        for index in 0...units.count {
            let blank = index == units.count
                || Converter.blank(units[index])
            if blank {
                let word = from < 0 ? nil
                    : Converter.word(page, units, from..<index, bounds,
                                     result.count)
                if let word { result.append(word) }
                from = -1
            } else if from < 0 {
                from = index
            }
        }
        return result
    }

    private static func blank(_ unit: UInt16) -> Bool {
        Unicode.Scalar(unit).map { scalar in
            CharacterSet.whitespacesAndNewlines.contains(scalar)
        } ?? false
    }

    private static func word(_ page: PDFPage, _ units: [UInt16],
                             _ range: Range<Int>, _ bounds: CGRect,
                             _ index: Int) -> Fragment? {
        let selection = page.selection(
            for: NSRange(location: range.lowerBound, length: range.count))
        let box = selection?.bounds(for: page) ?? .null
        var result: Fragment? = nil
        if !box.isNull && !box.isEmpty {
            result = Fragment(
                index: index,
                text: String(decoding: units[range], as: UTF16.self),
                frame: Converter.normalized(box, bounds))
        }
        return result
    }

    private static func normalized(_ box: CGRect,
                                   _ bounds: CGRect) -> CGRect {
        CGRect(x: (box.minX - bounds.minX) / bounds.width,
               y: (box.minY - bounds.minY) / bounds.height,
               width: box.width / bounds.width,
               height: box.height / bounds.height)
    }

    // The share of a page's glyph ink that falls inside a word box.
    // Drawn into a mask rather than tested per pixel against every box,
    // so the cost is one page-sized fill instead of pixels times words.

    private static func explained(_ fragments: [Fragment],
                                  _ image: CGImage,
                                  _ ink: UInt8) -> CGFloat {
        var result: CGFloat = 0
        let cover = Converter.mask(fragments, image)
        let drawn = image.dataProvider?.data
        let inside = cover?.dataProvider?.data
        if let cover, let drawn, let inside, image.bitsPerPixel == 8,
           cover.width == image.width, cover.height == image.height {
            let bytes = CFDataGetBytePtr(drawn)
            let over = CFDataGetBytePtr(inside)
            if let bytes, let over {
                result = withExtendedLifetime((drawn, inside)) {
                    Converter.covered(bytes, image, over, cover, ink)
                }
            }
        }
        return result
    }

    // Ink no darker than `ink` anywhere on the page leaves nothing for a
    // word to explain, so the words explain all of it - the caller has
    // already established that there ARE words.

    private static func covered(_ bytes: UnsafePointer<UInt8>,
                                _ image: CGImage,
                                _ over: UnsafePointer<UInt8>,
                                _ cover: CGImage,
                                _ ink: UInt8) -> CGFloat {
        var inside = 0
        var total = 0
        for row in 0..<image.height {
            for column in 0..<image.width
            where bytes[row * image.bytesPerRow + column] < ink {
                total += 1
                if over[row * cover.bytesPerRow + column] < 128 {
                    inside += 1
                }
            }
        }
        return total > 0 ? CGFloat(inside) / CGFloat(total) : 1
    }

    private static func mask(_ fragments: [Fragment],
                             _ image: CGImage) -> CGImage? {
        let width = CGFloat(image.width)
        let height = CGFloat(image.height)
        let context = CGContext(
            data: nil, width: image.width, height: image.height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue)
        if let context {
            context.setFillColor(gray: 1, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.setFillColor(gray: 0, alpha: 1)
            for fragment in fragments {
                context.fill(CGRect(x: fragment.frame.minX * width,
                                    y: fragment.frame.minY * height,
                                    width: fragment.frame.width * width,
                                    height: fragment.frame.height * height))
            }
        }
        return context?.makeImage()
    }

    // Recognition returns whole lines, and a line crosses every column
    // it touches - so a line box cannot tell two columns apart. Words
    // can: the whitespace between two columns is the only gap no word
    // ever spans.

    private static func words(of text: RecognizedText) -> [Fragment] {
        let string = text.string
        var result: [Fragment] = []
        for word in string.split(whereSeparator: { character in
            character.isWhitespace
        }) {
            let range = word.startIndex..<word.endIndex
            if let box = text.boundingBox(for: range) {
                result.append(Fragment(index: 0, text: String(word),
                                       frame: box.boundingBox.cgRect))
            }
        }
        return result
    }

    // A page is read one text column at a time: on a two-column layout
    // the whole left column precedes the whole right one, and no table
    // straddles the two.

    private func layout(_ fragments: [Fragment], _ image: CGImage,
                        _ number: Int) -> [Element] {
        let level = Converter.scan(image, inkLevel, rulingWidth,
                                   level: true)
        let uprights = Converter.scan(image, inkLevel, rulingHeight,
                                      level: false)
        let sides = self.sides(of: fragments)
        report("page \(number): \(fragments.count) words, "
               + "\(level.count) rules, \(uprights.count) uprights, "
               + "\(sides.count) page columns")
        var result: [Element] = []
        for side in sides {
            let inside = fragments.filter { fragment in
                side.contains(fragment.frame.midX)
            }
            result += self.elements(inside, level, uprights,
                                    side, image)
        }
        let audit = Converter.audit(fragments, result)
        onAudit?(number, audit)
        report("    audit: \(audit.spans) spans, \(audit.lost) lost, "
               + "\(audit.unaccounted) unaccounted")
        return result
    }

    // A page reads as two columns only where a corridor divides the
    // TEXT: near the middle, with a real share of the words on either
    // hand. Everything else that survives the ink profile - the outer
    // margin, a table's own column gap, the slivers a centred page
    // number leaves on both sides of itself - is not a gutter.

    private func sides(of fragments: [Fragment])
        -> [ClosedRange<CGFloat>] {
        let spans = columns(of: fragments, pageGap)
        let least = Int(CGFloat(fragments.count) * pageShare)
        var cut = CGFloat.infinity
        for (left, right) in zip(spans, spans.dropFirst()) {
            let middle = (left.upperBound + right.lowerBound) / 2
            let beyond = fragments.filter { fragment in
                fragment.frame.midX > middle
            }.count
            if pageMiddle.contains(middle) && beyond >= least
                && fragments.count - beyond >= least {
                cut = middle
            }
        }
        return Converter.joined(spans, cut)
    }

    private static func joined(_ spans: [ClosedRange<CGFloat>],
                               _ cut: CGFloat)
        -> [ClosedRange<CGFloat>] {
        var result: [ClosedRange<CGFloat>] = []
        for span in spans {
            let last = result.last
            let together = last.map { last in
                (last.upperBound < cut) == (span.lowerBound < cut)
            } ?? false
            if let last, together {
                result[result.count - 1] = last.lowerBound...span.upperBound
            } else {
                result.append(span)
            }
        }
        return result
    }

    // Every recognized span must reach the output exactly once. The
    // check needs no ground truth: a multiset difference names dropped
    // text, duplicated text, and text that was never recognized at all,
    // on any page. Losses hide easily - a span inside a table's rows
    // but outside its columns simply vanishes - and nothing else in the
    // pipeline would notice.

    private static func audit(_ fragments: [Fragment],
                              _ elements: [Element]) -> Audit {
        var want: [String: Int] = [:]
        for fragment in fragments {
            want[fragment.text, default: 0] += 1
        }
        var have: [String: Int] = [:]
        for element in elements {
            for word in Converter.spoken(element)
                .split(whereSeparator: { character in
                    character.isWhitespace
                }) {
                have[String(word), default: 0] += 1
            }
        }
        var lost = 0
        var extra = 0
        for (word, count) in want {
            lost += max(count - (have[word] ?? 0), 0)
        }
        for (word, count) in have {
            extra += max(count - (want[word] ?? 0), 0)
        }
        return Audit(spans: fragments.count, lost: lost,
                     unaccounted: extra)
    }

    private func elements(_ fragments: [Fragment], _ level: [Ruling],
                          _ uprights: [Ruling],
                          _ side: ClosedRange<CGFloat>,
                          _ image: CGImage) -> [Element] {
        var placed = Set<Int>()
        var found: [(CGFloat, Element)] = []
        for region in self.regions(level, fragments, side) {
            let inside = within(fragments, region, side)
            report(String(format: "    table %.3f...%.3f",
                          Double(region.lowerBound),
                          Double(region.upperBound)))
            found.append((region.upperBound,
                          .table(cells(of: region, fragments, level,
                                       uprights, side, image))))
            for fragment in inside { placed.insert(fragment.index) }
        }
        let rest = fragments.filter { fragment in
            !placed.contains(fragment.index)
        }
        return (found + prose(rest)).sorted { left, right in
            left.0 > right.0
        }.map { pair in pair.1 }
    }

    // The guard asks whether the STRIP just added still resolves into
    // columns - `line...bottom`, bottom being the last rule accepted -
    // not whether the whole region does. Testing the whole region fails
    // in both directions at once: once any table is inside it the union
    // has columns forever and the region swallows captions, prose and
    // the next table; while one wide row anywhere collapses that same
    // union to one column and cuts the region short.
    //
    // A table region runs from its first rule to its last. Rules alone
    // would also join two tables with a paragraph between them, so a
    // region grows only while the text it covers still resolves into
    // columns - prose spans the full width and resolves into one.

    private func regions(_ level: [Ruling], _ fragments: [Fragment],
                         _ side: ClosedRange<CGFloat>)
        -> [ClosedRange<CGFloat>] {
        let mine = level.filter { rule in
            Converter.agrees(rule.span, side, spanAgreement)
        }.sorted { left, right in left.position > right.position }
        let lines = Converter.merged(mine, rulingMerge)
        var result: [ClosedRange<CGFloat>] = []
        var top = CGFloat.infinity
        var bottom = CGFloat.infinity
        for line in lines {
            let window = within(fragments, line...bottom, side)
            let holds = top < .infinity
                && columns(of: window, wordGap).count >= 2
            let heights = window.map { fragment in fragment.frame.height }
            let lone = top < .infinity && bottom == top
                && !window.isEmpty && bands(window).count == 1
                && top - line <= Converter.median(heights) * spanWindow
            if holds || lone {
                bottom = line
            } else {
                if top < .infinity && bottom < top {
                    result.append(bottom...top)
                }
                top = line
                bottom = line
            }
        }
        if top < .infinity && bottom < top { result.append(bottom...top) }
        return result
    }

    private func within(_ fragments: [Fragment],
                        _ region: ClosedRange<CGFloat>,
                        _ side: ClosedRange<CGFloat>) -> [Fragment] {
        fragments.filter { fragment in
            region.contains(fragment.frame.midY)
                && side.contains(fragment.frame.midX)
        }
    }

    // Rows come from the rules when a table draws one per row, and from
    // the text lines themselves when it draws only the three rules of a
    // book-style table. Columns likewise: drawn uprights when present,
    // otherwise the whitespace corridors that no row crosses.

    private func cells(of region: ClosedRange<CGFloat>,
                       _ fragments: [Fragment], _ level: [Ruling],
                       _ uprights: [Ruling],
                       _ side: ClosedRange<CGFloat>,
                       _ image: CGImage) -> [[String]] {
        let inside = within(fragments, region, side)
        let lines = bands(inside).map { band in Converter.extent(band) }
        let width = Converter.spread(inside)
        let inner = Converter.merged(level.filter { rule in
            rule.position > region.lowerBound + rulingMerge
                && rule.position < region.upperBound - rulingMerge
                && Converter.covers(rule.span, width, fullWidth)
        }.sorted { left, right in left.position > right.position },
                                     rulingMerge)
        let ruled = inner.count + 1 >= lines.count && lines.count > 1
        let rows = ruled ? Converter.slice(region, inner)
            : Converter.separated(lines, region)
        let corridors = cellGap > 0
            ? self.corridors(region, width, image)
            : columns(of: inside, wordGap)
        let posts = Converter.merged(uprights.filter { post in
            Converter.agrees(post.span, region, spanAgreement)
                && side.contains(post.position)
        }.sorted { left, right in left.position > right.position },
                                     rulingMerge).reversed()
        let bars = Converter.slices(width,
                                    Converter.cuts(Array(posts),
                                                   corridors, wordGap))
        let header = inner.isEmpty ? 1
            : max(rows.filter { row in
                row.lowerBound >= inner[0] - rulingMerge
            }.count, 1)
        report(String(format: "      rows %d (ruled %@) inner %d "
                      + "columns %d header %d", rows.count,
                      ruled ? "yes" : "no", inner.count, bars.count,
                      header))
        return Converter.laid(Grid(rows: rows, columns: bars,
                                   header: header), inside)
    }

    // Place what is certain first, and let it decide the rest. A span
    // lying wholly inside one column can only belong to that column, so
    // those spans - and only those - are trusted to report where the
    // column really is. A corridor merely says where the whitespace
    // was; the settled extent says where the text actually sits, which
    // is what an ambiguous span, straddling a boundary or right-aligned
    // away from its neighbours, must be judged against.

    private static func settled(_ columns: [ClosedRange<CGFloat>],
                                _ fragments: [Fragment])
        -> [ClosedRange<CGFloat>] {
        var lower = columns.map { column in column.upperBound }
        var upper = columns.map { column in column.lowerBound }
        var sure = [Bool](repeating: false, count: columns.count)
        for fragment in fragments {
            let held = columns.indices.filter { index in
                columns[index].lowerBound <= fragment.frame.midX
                    && fragment.frame.midX <= columns[index].upperBound
                    && columns[index].lowerBound <= fragment.frame.minX
                    && fragment.frame.maxX <= columns[index].upperBound
            }
            if held.count == 1 {
                let index = held[0]
                lower[index] = min(lower[index], fragment.frame.minX)
                upper[index] = max(upper[index], fragment.frame.maxX)
                sure[index] = true
            }
        }
        return columns.indices.map { index in
            sure[index] ? lower[index]...upper[index] : columns[index]
        }
    }

    // Order a row by x alone. Sorting by midY first and using x only to
    // break ties reads as reading order but is not: midY comes back from
    // recognition as a float and two words on one line are never exactly
    // equal, so the tiebreak never fires and the row comes out ordered by
    // OCR jitter. The band already established these fragments are one
    // row; x is the only axis left that means anything.

    private static func laid(_ grid: Grid,
                             _ fragments: [Fragment]) -> [[String]] {
        let bars = Converter.settled(grid.columns, fragments)
        var result: [[String]] = []
        for row in grid.rows {
            var cells = [String](repeating: "", count: bars.count)
            let line = fragments.filter { fragment in
                row.contains(fragment.frame.midY)
            }.sorted { left, right in
                left.frame.minX < right.frame.minX
            }
            for fragment in line {
                let column = Converter.column(of: fragment, bars)
                cells[column] = cells[column].isEmpty ? fragment.text
                    : cells[column] + " " + fragment.text
            }
            if cells.contains(where: { cell in !cell.isEmpty }) {
                result.append(cells)
            }
        }
        return Converter.folded(Converter.occupied(result), grid.header)
    }

    // A column empty on every row is not a column. Drawn rules and ink
    // corridors both propose more edges than a table has - a rule at the
    // table's own border, a corridor inside a wide gutter between two
    // value columns - and each surplus edge shows up as a column of
    // nothing.

    private static func occupied(_ rows: [[String]]) -> [[String]] {
        let width = rows.reduce(0) { widest, row in
            max(widest, row.count)
        }
        var used: [Bool] = Array(repeating: false, count: width)
        for row in rows {
            for (index, cell) in row.enumerated() where !cell.isEmpty {
                used[index] = true
            }
        }
        var result: [[String]] = []
        for row in rows {
            var kept: [String] = []
            for index in 0..<width {
                if used[index] {
                    kept.append(index < row.count ? row[index] : "")
                }
            }
            result.append(kept)
        }
        return result
    }

    // Markdown has one header row and no spanning cells, so a stacked
    // header - what LaTeX writes with multicolumn - folds into that one
    // row, each column keeping every level of its own label.

    private static func folded(_ rows: [[String]],
                               _ header: Int) -> [[String]] {
        var result = rows
        if header > 1 && rows.count > header {
            var merged = [String](repeating: "", count: rows[0].count)
            for level in rows.prefix(header) {
                for (index, cell) in level.enumerated()
                    where !cell.isEmpty && index < merged.count {
                    merged[index] = merged[index].isEmpty ? cell
                        : merged[index] + " " + cell
                }
            }
            result = [merged] + rows.dropFirst(header)
        }
        return result
    }

    // Text the rules did not claim. Some of it is still a table: a
    // renderer may draw none at all. Consecutive lines that keep
    // resolving into the same columns are one - prose collapses to a
    // single column as soon as a second line joins it, because no two
    // prose lines break their words in the same places.

    private func prose(_ fragments: [Fragment])
        -> [(CGFloat, Element)] {
        let bands = self.bands(fragments)
        let gap = Converter.medianGap(bands) * rowGapRatio
        var result: [(CGFloat, Element)] = []
        var index = 0
        while index < bands.count {
            let top = Converter.extent(bands[index]).upperBound
            let shared = aligned(bands, index)
            let candidate = shared - index >= alignedRows - 1
                ? speculative(Array(bands[index...shared])) : []
            if Converter.tabular(candidate, proseCellShare, proseCellLength) {
                result.append((top, .table(candidate)))
                index = shared + 1
            } else {
                var stop = index
                var next = index + 1
                while next < bands.count
                      && Converter.distance(bands[next - 1],
                                            bands[next]) <= gap {
                    stop = next
                    next += 1
                }
                let text = bands[index...stop].map { band in
                    Converter.text(of: band)
                }
                result.append((top,
                               .paragraph(text.joined(separator: " "))))
                index = stop + 1
            }
        }
        return result
    }

    private func speculative(_ run: [[Fragment]]) -> [[String]] {
        let words = run.flatMap { band in band }
        let rows = run.map { band in Converter.extent(band) }
        let floor = rows[rows.count - 1].lowerBound
        let region = floor...rows[0].upperBound
        return Converter.laid(
            Grid(rows: Converter.separated(rows, region),
                 columns: columns(of: words, wordGap),
                 header: 1), words)
    }

    // Aligned lines are not proof of a table: a paragraph whose lines
    // happen to break alike passes the column test too. What separates
    // them is what the cells CONTAIN - a table cell is short and
    // label-like, a prose cell is a sentence. Taken from MarkItDown's
    // pdfplumber path, which rejects a candidate when too many of its
    // cells read as prose.

    private static func tabular(_ rows: [[String]], _ share: CGFloat,
                                _ length: Int) -> Bool {
        var filled = 0
        var wordy = 0
        for row in rows {
            for cell in row where !cell.isEmpty {
                filled += 1
                if cell.count > length { wordy += 1 }
            }
        }
        return filled > 0 && CGFloat(wordy) <= CGFloat(filled) * share
    }

    private func aligned(_ bands: [[Fragment]], _ start: Int) -> Int {
        var end = start
        var words = bands[start]
        var index = start + 1
        while index < bands.count
              && columns(of: words + bands[index], wordGap).count >= 2 {
            words += bands[index]
            end = index
            index += 1
        }
        return end
    }

    private func bands(_ fragments: [Fragment]) -> [[Fragment]] {
        let sorted = fragments.sorted { left, right in
            left.frame.midY > right.frame.midY
        }
        let heights = sorted.map { fragment in fragment.frame.height }
        let tolerance = Converter.median(heights) * bandTolerance
        var result: [[Fragment]] = []
        for fragment in sorted {
            let anchor = result.last?.first
            let near = abs((anchor?.frame.midY ?? -1)
                           - fragment.frame.midY) <= tolerance
            if anchor != nil && near {
                result[result.count - 1].append(fragment)
            } else {
                result.append([fragment])
            }
        }
        return result.map { band in
            band.sorted { left, right in
                left.frame.minX < right.frame.minX
            }
        }
    }

    // Column corridors. A gap between two columns is no wider than the
    // gap between two words - the difference is that the column gap is
    // empty on EVERY row. So accumulate ink from all rows into one
    // profile first; only gaps that survive that union are corridors.

    private func columns(of fragments: [Fragment], _ gap: CGFloat,
                         _ vote: CGFloat = 0) -> [ClosedRange<CGFloat>] {
        let lines = vote > 0 ? bands(fragments).map { band in band }
            : [fragments]
        var result: [ClosedRange<CGFloat>] = []
        for span in Converter.profile(lines, buckets, vote) {
            let last = result.last
            let joined = span.lowerBound - (last?.upperBound ?? -1) < gap
            if let last, joined {
                let upper = max(last.upperBound, span.upperBound)
                result[result.count - 1] = last.lowerBound...upper
            } else {
                result.append(span)
            }
        }
        return result
    }

    // A boolean union asks "did ANY row cross here", and one spanning
    // header answers yes for every corridor beneath it - which is why a
    // multicolumn head used to collapse a table to a single column. Count
    // the crossing ROWS instead: a bucket is a wall while few enough rows
    // cross it, so a head spanning three of twelve rows can no longer
    // erase a corridor the other nine respect.

    private static func profile(_ lines: [[Fragment]], _ buckets: Int,
                                _ vote: CGFloat)
        -> [ClosedRange<CGFloat>] {
        var crossings = [Int](repeating: 0, count: buckets)
        let last = buckets - 1
        for line in lines {
            var touched = [Bool](repeating: false, count: buckets)
            for fragment in line {
                let from = Converter.bucket(fragment.frame.minX, last)
                let upto = Converter.bucket(fragment.frame.maxX, last)
                for index in min(from, upto)...max(from, upto) {
                    touched[index] = true
                }
            }
            for index in 0..<buckets where touched[index] {
                crossings[index] += 1
            }
        }
        let wall = Int((CGFloat(lines.count) * vote).rounded(.down))
        var result: [ClosedRange<CGFloat>] = []
        var start = -1
        for index in 0...buckets {
            let ink = index < buckets && crossings[index] > wall
            if ink && start < 0 { start = index }
            if !ink && start >= 0 {
                let lower = CGFloat(start) / CGFloat(buckets)
                let upper = CGFloat(index) / CGFloat(buckets)
                result.append(lower...upper)
                start = -1
            }
        }
        return result
    }

    private static func bucket(_ x: CGFloat, _ last: Int) -> Int {
        min(max(Int(x * CGFloat(last + 1)), 0), last)
    }

    // Judge by how much of the span a column holds, not by where its
    // midpoint lands: a cell spanning two columns has its midpoint over
    // the gap between them, and a midpoint test sends it to whichever
    // side rounds better.

    private static func column(of fragment: Fragment,
                               _ columns: [ClosedRange<CGFloat>]) -> Int {
        var best = 0
        var strongest = -CGFloat.infinity
        for (index, column) in columns.enumerated() {
            let shared = min(column.upperBound, fragment.frame.maxX)
                - max(column.lowerBound, fragment.frame.minX)
            let claim = shared > 0 ? shared
                : -min(abs(column.lowerBound - fragment.frame.midX),
                       abs(column.upperBound - fragment.frame.midX))
            if claim > strongest {
                strongest = claim
                best = index
            }
        }
        return best
    }

    private static func extent(_ band: [Fragment]) -> ClosedRange<CGFloat> {
        let lower = band.map { fragment in fragment.frame.minY }.min() ?? 0
        let upper = band.map { fragment in fragment.frame.maxY }.max() ?? 0
        return lower...max(upper, lower)
    }

    // Cut a range at descending interior positions, top piece first, so
    // rows and columns come out in reading order.

    private static func slice(_ whole: ClosedRange<CGFloat>,
                              _ cuts: [CGFloat]) -> [ClosedRange<CGFloat>] {
        var result: [ClosedRange<CGFloat>] = []
        var upper = whole.upperBound
        for cut in cuts where cut > whole.lowerBound && cut < upper {
            result.append(cut...upper)
            upper = cut
        }
        result.append(whole.lowerBound...upper)
        return result
    }

    // Drawn uprights and whitespace corridors are both evidence of a
    // column edge, and a table often supplies one where it lacks the
    // other: a single upright beside three corridors is three columns
    // plus one, not a choice between them.

    private static func cuts(_ posts: [CGFloat],
                             _ corridors: [ClosedRange<CGFloat>],
                             _ apart: CGFloat) -> [CGFloat] {
        var found = posts
        for (left, right) in zip(corridors, corridors.dropFirst()) {
            found.append((left.upperBound + right.lowerBound) / 2)
        }
        var result: [CGFloat] = []
        for cut in found.sorted() {
            if cut - (result.last ?? -1) > apart { result.append(cut) }
        }
        return result
    }

    // Text-band extents overlap where one line's descender reaches
    // past the next line's ascender, and an overlapping row would claim
    // the same word twice. Cut instead halfway between the bands.

    private static func separated(_ lines: [ClosedRange<CGFloat>],
                                  _ region: ClosedRange<CGFloat>)
        -> [ClosedRange<CGFloat>] {
        var cuts: [CGFloat] = []
        for (upper, lower) in zip(lines, lines.dropFirst()) {
            cuts.append((upper.lowerBound + lower.upperBound) / 2)
        }
        return Converter.slice(region, cuts)
    }

    // Cut a range at ascending interior positions, left piece first.

    private static func slices(_ whole: ClosedRange<CGFloat>,
                               _ cuts: [CGFloat]) -> [ClosedRange<CGFloat>] {
        var result: [ClosedRange<CGFloat>] = []
        var lower = whole.lowerBound
        for cut in cuts where cut > lower && cut < whole.upperBound {
            result.append(lower...cut)
            lower = cut
        }
        result.append(lower...whole.upperBound)
        return result
    }

    private static func spread(_ fragments: [Fragment])
        -> ClosedRange<CGFloat> {
        let lower = fragments.map { f in f.frame.minX }.min() ?? 0
        let upper = fragments.map { f in f.frame.maxX }.max() ?? 1
        return lower...max(upper, lower)
    }

    private static func distance(_ upper: [Fragment],
                                 _ lower: [Fragment]) -> CGFloat {
        let bottom = upper.map { f in f.frame.minY }.min() ?? 0
        let top = lower.map { f in f.frame.maxY }.max() ?? 0
        return bottom - top
    }

    private static func medianGap(_ bands: [[Fragment]]) -> CGFloat {
        var gaps: [CGFloat] = []
        for (upper, lower) in zip(bands, bands.dropFirst()) {
            gaps.append(Converter.distance(upper, lower))
        }
        return max(Converter.median(gaps), 0)
    }

    private static func median(_ values: [CGFloat]) -> CGFloat {
        let sorted = values.sorted()
        var result: CGFloat = 0
        if !sorted.isEmpty { result = sorted[sorted.count / 2] }
        return result
    }

    private static func text(of band: [Fragment]) -> String {
        band.map { fragment in fragment.text }.joined(separator: " ")
    }

    // Column corridors read off the raster rather than off the boxes.
    // Rows that a drawn rule crosses are left out: a rule covers every
    // corridor under it, and one such row would erase them all. The rule
    // is measured against the TABLE's width, which is what it spans -
    // against the page column's it would look far too short to matter.

    // Ink unioned over every row asks "did ANYTHING cross here", and one
    // spanning header answers yes for every corridor beneath it - which
    // is how a multicolumn head collapses a table to a single column.
    // Count crossing ROWS instead: a bucket stays a wall while fewer
    // than `cellVote` of the region's inked rows reach into it, so a
    // head spanning one line of twelve can no longer erase a corridor
    // the other eleven respect.

    private func corridors(_ region: ClosedRange<CGFloat>,
                           _ width: ClosedRange<CGFloat>,
                           _ image: CGImage) -> [ClosedRange<CGFloat>] {
        var crossings = [Int](repeating: 0, count: buckets)
        var inked = 0
        let data = image.dataProvider?.data
        if let data, image.bitsPerPixel == 8 {
            let bytes = CFDataGetBytePtr(data)
            if let bytes {
                let top = Converter.row(region.upperBound, image.height)
                let floor = Converter.row(region.lowerBound, image.height)
                let least = Int(CGFloat(image.width) * fullWidth
                                * (width.upperBound - width.lowerBound))
                let from = Converter.bucket(width.lowerBound,
                                            image.width - 1)
                let upto = Converter.bucket(width.upperBound,
                                            image.width - 1)
                for row in top...floor where Converter.inked(
                    bytes, image, row, inkLevel, least, true) == nil {
                    var touched = [Bool](repeating: false, count: buckets)
                    for column in from...upto
                    where bytes[row * image.bytesPerRow + column]
                        < inkLevel {
                        touched[Converter.bucket(
                            CGFloat(column) / CGFloat(image.width),
                            buckets - 1)] = true
                    }
                    var any = false
                    for index in 0..<buckets where touched[index] {
                        crossings[index] += 1
                        any = true
                    }
                    if any { inked += 1 }
                }
            }
        }
        let wall = Int((CGFloat(inked) * cellVote).rounded(.down))
        var covered = [Bool](repeating: false, count: buckets)
        for index in 0..<buckets {
            covered[index] = crossings[index] > wall
        }
        return Converter.runs(covered, cellGap)
    }

    private static func row(_ y: CGFloat, _ height: Int) -> Int {
        min(max(Int((1 - y) * CGFloat(height)), 0), height - 1)
    }

    private static func runs(_ covered: [Bool],
                             _ gap: CGFloat) -> [ClosedRange<CGFloat>] {
        var result: [ClosedRange<CGFloat>] = []
        var start = -1
        for index in 0...covered.count {
            let ink = index < covered.count && covered[index]
            if ink && start < 0 { start = index }
            if !ink && start >= 0 {
                let lower = CGFloat(start) / CGFloat(covered.count)
                let upper = CGFloat(index) / CGFloat(covered.count)
                let last = result.last
                if let last, lower - last.upperBound < gap {
                    result[result.count - 1] = last.lowerBound...upper
                } else {
                    result.append(lower...upper)
                }
                start = -1
            }
        }
        return result
    }

    // Rule detection, in normalized page space like everything else.
    // The same scan finds horizontal rules and vertical ones; only the
    // axis it walks differs.

    private static func scan(_ image: CGImage, _ ink: UInt8,
                             _ share: CGFloat, level: Bool) -> [Ruling] {
        var result: [Ruling] = []
        let data = image.dataProvider?.data
        if let data, image.bitsPerPixel == 8 {
            let bytes = CFDataGetBytePtr(data)
            if let bytes {
                let across = level ? image.width : image.height
                let along = level ? image.height : image.width
                let least = Int(CGFloat(across) * share)
                var start = -1
                var span = 0...0
                for index in 0...along {
                    let line = index < along
                        ? Converter.inked(bytes, image, index, ink, least,
                                          level)
                        : nil
                    let extends = line.map { found in
                        start >= 0 && Converter.overlaps(found, span)
                    } ?? false
                    if extends, let line {
                        let lower = min(span.lowerBound, line.lowerBound)
                        let upper = max(span.upperBound, line.upperBound)
                        span = lower...upper
                    } else {
                        if start >= 0 {
                            result.append(Converter.ruling(start, span,
                                                           image, level))
                            result.append(Converter.ruling(index - 1, span,
                                                           image, level))
                        }
                        start = line == nil ? -1 : index
                        span = line ?? 0...0
                    }
                }
            }
        }
        return result
    }

    // A rule is a long CONTIGUOUS run of ink. Measuring instead from
    // the first inked pixel to the last would work for a horizontal
    // rule on an otherwise empty row, but never for a vertical one:
    // any text elsewhere in that pixel column stretches the extent and
    // the rule dissolves into it.

    private static func inked(_ bytes: UnsafePointer<UInt8>,
                              _ image: CGImage, _ index: Int,
                              _ ink: UInt8, _ least: Int,
                              _ level: Bool) -> ClosedRange<Int>? {
        let across = level ? image.width : image.height
        var start = -1
        var last = -1
        var gap = 0
        var longest = -1
        var from = 0
        var upto = -1
        for step in 0..<across {
            let row = level ? index : step
            let column = level ? step : index
            let dark = bytes[row * image.bytesPerRow + column] < ink
            if dark {
                if start < 0 { start = step }
                last = step
                gap = 0
            } else if start >= 0 {
                gap += 1
                if gap > 2 {
                    if last - start > longest {
                        longest = last - start
                        from = start
                        upto = last
                    }
                    start = -1
                }
            }
        }
        if start >= 0 && last - start > longest {
            longest = last - start
            from = start
            upto = last
        }
        var result: ClosedRange<Int>? = nil
        if longest + 1 >= least { result = from...upto }
        return result
    }

    private static func ruling(_ index: Int, _ span: ClosedRange<Int>,
                               _ image: CGImage, _ level: Bool) -> Ruling {
        let across = CGFloat(level ? image.width : image.height)
        let along = CGFloat(level ? image.height : image.width)
        let lower = CGFloat(span.lowerBound) / across
        let upper = CGFloat(span.upperBound) / across
        let offset = CGFloat(index) / along
        var position = offset
        var range = lower...upper
        if level {
            position = 1 - offset
        } else {
            range = (1 - upper)...(1 - lower)
        }
        return Ruling(position: position, span: range)
    }

    private static func merged(_ rulings: [Ruling],
                               _ tolerance: CGFloat) -> [CGFloat] {
        var result: [CGFloat] = []
        for ruling in rulings {
            if (result.last ?? .infinity) - ruling.position > tolerance {
                result.append(ruling.position)
            }
        }
        return result
    }

    // `agrees` asks whether two extents overlap substantially, scaled
    // by the SHORTER one - so a short rule always agrees with the wide
    // table containing it. `covers` asks the different question a row
    // boundary needs: does this rule cross the whole table.

    private static func covers(_ span: ClosedRange<CGFloat>,
                               _ whole: ClosedRange<CGFloat>,
                               _ share: CGFloat) -> Bool {
        let lower = max(span.lowerBound, whole.lowerBound)
        let upper = min(span.upperBound, whole.upperBound)
        let shared = max(upper - lower, 0)
        let width = whole.upperBound - whole.lowerBound
        return width > 0 && shared >= width * share
    }

    private static func agrees(_ span: ClosedRange<CGFloat>,
                               _ other: ClosedRange<CGFloat>,
                               _ share: CGFloat) -> Bool {
        let lower = max(span.lowerBound, other.lowerBound)
        let upper = min(span.upperBound, other.upperBound)
        let shared = max(upper - lower, 0)
        let shorter = min(span.upperBound - span.lowerBound,
                          other.upperBound - other.lowerBound)
        return shorter > 0 && shared >= shorter * share
    }

    private static func overlaps<T: Comparable>(
        _ left: ClosedRange<T>, _ right: ClosedRange<T>) -> Bool {
        max(left.lowerBound, right.lowerBound)
            <= min(left.upperBound, right.upperBound)
    }

    private func report(_ message: String) {
        if trace {
            FileHandle.standardError.write(Data((message + "\n").utf8))
        }
    }

    // Markdown emission, shared by every mode.

    private static func render(_ pages: [Page]) -> String {
        var blocks: [String] = []
        for page in pages {
            for element in page.elements {
                blocks.append(Converter.render(element))
            }
        }
        return blocks.joined(separator: "\n\n") + "\n"
    }

    // The words an element carries, without the Markdown that frames
    // them - pipes and rules are not recognized text and must not count.

    private static func spoken(_ element: Element) -> String {
        var result = ""
        switch element {
        case .paragraph(let text):
            result = text
        case .table(let rows):
            result = rows.map { row in
                row.joined(separator: " ")
            }.joined(separator: " ")
        }
        return result
    }

    private static func render(_ element: Element) -> String {
        var result = ""
        switch element {
        case .paragraph(let text):
            result = text
        case .table(let rows):
            result = Converter.table(rows)
        }
        return result
    }

    private static func table(_ rows: [[String]]) -> String {
        let width = rows.reduce(0) { widest, row in
            max(widest, row.count)
        }
        var lines: [String] = []
        for (index, row) in rows.enumerated() {
            var cells = row.map { cell in
                cell.replacingOccurrences(of: "|", with: "\\|")
            }
            while cells.count < width { cells.append("") }
            lines.append("| " + cells.joined(separator: " | ") + " |")
            if index == 0 {
                let rule = Array(repeating: "---", count: width)
                lines.append("| " + rule.joined(separator: " | ") + " |")
            }
        }
        return lines.joined(separator: "\n")
    }

    private static func require<T>(_ value: T?,
                                   _ failure: ConversionError) throws -> T {
        switch value {
        case .some(let unwrapped): return unwrapped
        case .none:                throw failure
        }
    }
}

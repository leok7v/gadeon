#if os(macOS)
import AppKit

// A cell rather than an image, so the formula stays vector and picks up the
// text colour at DRAW time. An attachment holding a rasterized formula bakes
// one theme's ink into the document and has to be rebuilt when the theme
// flips; this one just redraws.

final class MathAttachmentCell: NSTextAttachmentCell {

    // The metrics are held apart from the layout because the two sizing
    // overrides below are nonisolated, and a laid-out formula holds the
    // shared math font, which is not safe to read off this actor. Drawing IS
    // on it, so that one reads the layout directly.
    private nonisolated let extent: CGSize
    private nonisolated let descent: CGFloat
    private let layout: MathLayout
    private nonisolated static let inset: CGFloat = 4

    init(layout: MathLayout) {
        self.layout = layout
        self.extent = CGSize(width: layout.width + Self.inset * 2,
                             height: layout.height)
        self.descent = layout.descent
        super.init()
    }

    // Never archived: the attachment is built fresh from the markdown every
    // time the document is laid out.
    required init(coder: NSCoder) {
        fatalError("MathAttachmentCell is not decodable")
    }

    override func cellSize() -> NSSize { extent }

    // The formula sits on the text baseline like a very tall glyph, so its
    // descent is what hangs below.
    override func cellBaselineOffset() -> NSPoint {
        NSPoint(x: 0, y: -descent)
    }

    override func draw(withFrame cellFrame: NSRect, in controlView: NSView?) {
        if let ctx = NSGraphicsContext.current?.cgContext {
            let origin = CGPoint(x: cellFrame.minX + Self.inset,
                                 y: cellFrame.minY)
            layout.draw(in: ctx, at: origin,
                        color: NSColor.textColor.cgColor,
                        flipped: controlView?.isFlipped ?? true)
        }
    }
}

extension DocumentText {

    static func mathAttachment(_ layout: MathLayout) -> NSTextAttachment {
        let attachment = NSTextAttachment()
        attachment.attachmentCell = MathAttachmentCell(layout: layout)
        return attachment
    }

    // Horizontal cell padding, as a percentage of the table width so it can
    // be subtracted from the column-share budget in the same unit.
    private static var cellPad: CGFloat { 0.6 }

    private static func contentBudget(cols: Int) -> CGFloat {
        max(100 - CGFloat(cols) * cellPad * 2, 50)
    }

    // Solved against the SHARES the table will actually be built with, not
    // against the bare sum of the minimums. A column's share is a fixed
    // fraction of the table width, so the width at which column c finally
    // holds its widest token is min[c] / share[c], and the table needs the
    // largest of those. Summing the minimums instead answers a question
    // nobody asked: the shares are weighted by character count, so the sum
    // can be reached with a column still starved.

    static func tableMinimumWidth(headers: [String], rows: [[String]],
                                  style: MarkdownStyle) -> CGFloat {
        var result: CGFloat = 0
        let cols = max(headers.count, rows.map { r in r.count }.max() ?? 0)
        if cols > 0 {
            let mins = columnMinimums(headers: headers, rows: rows,
                                      cols: cols, style: style)
            let fractions = TableMetrics.pointWidths(
                headers: headers, rows: rows,
                available: contentBudget(cols: cols) / 100)
            for c in 0..<cols where c < fractions.count && fractions[c] > 0 {
                let need = mins[c] / fractions[c]
                if need > result { result = need }
            }
            result = ceil(result)
        }
        return result
    }

    static func table(headers: [String], rows: [[String]],
                      alignments: [Markdown.Alignment], style: MarkdownStyle,
                      images: [URL: PlatformImage]) -> NSAttributedString {
        let m = NSMutableAttributedString()
        let cols = max(headers.count, rows.map { r in r.count }.max() ?? 0)
        if cols > 0 {
            let atomicId = UUID().uuidString
            let textTable = NSTextTable()
            textTable.numberOfColumns = cols
            // Automatic, not fixed: fixed layout is CSS table-layout: fixed,
            // where a cell whose content outgrows its declared width spills
            // OVER the next column instead of widening, so headers print on
            // top of each other. The shares below are weighted by character
            // count and cannot know the rendered size, so any font they were
            // not computed for overflowed them.
            textTable.layoutAlgorithm = .automaticLayoutAlgorithm
            // The shares have to leave room for the padding, or the table
            // demands 100% plus the padding and the last columns are squeezed
            // off the edge. Percentage padding keeps that in one unit.
            let shares = TableMetrics.pointWidths(
                headers: headers, rows: rows,
                available: contentBudget(cols: cols))
            var rowIdx = 0
            if !headers.isEmpty {
                m.append(tableRow(headers, table: textTable, rowIdx: rowIdx,
                                  cols: cols, shares: shares,
                                  alignments: alignments, bold: true,
                                  tint: platformWhite(0.5, alpha: 0.14),
                                  atomicId: atomicId, style: style,
                                  images: images))
                rowIdx += 1
            }
            for (idx, row) in rows.enumerated() {
                let tint: PlatformColor = idx % 2 == 1
                    ? platformWhite(0.5, alpha: 0.07) : platformClearColor
                m.append(tableRow(row, table: textTable, rowIdx: rowIdx,
                                  cols: cols, shares: shares,
                                  alignments: alignments, bold: false,
                                  tint: tint, atomicId: atomicId,
                                  style: style, images: images))
                rowIdx += 1
            }
            // One CONTIGUOUS atomic id AND kind over the whole table content
            // (cells + the inter-row separators, which carry neither per-cell)
            // so the copy-overlay grouping yields ONE block -- one Copy button
            // at the table corner -- and selection-snap sees one atomic unit.
            // Stamped BEFORE the trailing block newline is appended, so a drag
            // ending just past the table does not grab the blank line and the
            // copy overlay hugs the table.
            let content = NSRange(location: 0, length: m.length)
            m.addAttribute(atomicIdKey, value: atomicId, range: content)
            m.addAttribute(atomicKindKey,
                           value: AtomicKind.table.rawValue, range: content)
            m.addAttribute(atomicCopyKey,
                           value: TableMetrics.serializeMonospaced(
                               headers: headers, rows: rows),
                           range: content)
            m.append(NSAttributedString(string: "\n"))
        }
        return m
    }

    private static func tableRow(_ cells: [String], table: NSTextTable,
                                 rowIdx: Int, cols: Int, shares: [CGFloat],
                                 alignments: [Markdown.Alignment], bold: Bool,
                                 tint: PlatformColor, atomicId: String,
                                 style: MarkdownStyle,
                                 images: [URL: PlatformImage])
        -> NSAttributedString {
        let m = NSMutableAttributedString()
        let base = bold ? boldFont(of: bodyFont(style)) : bodyFont(style)
        for col in 0..<cols {
            let cellText = col < cells.count ? cells[col] : ""
            let block = NSTextTableBlock(table: table, startingRow: rowIdx,
                                         rowSpan: 1, startingColumn: col,
                                         columnSpan: 1)
            if col < shares.count {
                block.setValue(shares[col], type: .percentageValueType,
                               for: .width)
            }
            block.setWidth(cellPad, type: .percentageValueType,
                           for: .padding, edge: .minX)
            block.setWidth(cellPad, type: .percentageValueType,
                           for: .padding, edge: .maxX)
            block.setWidth(3, type: .absoluteValueType,
                           for: .padding, edge: .minY)
            block.setWidth(3, type: .absoluteValueType,
                           for: .padding, edge: .maxY)
            block.backgroundColor = tint
            let para = NSMutableParagraphStyle()
            // Word wrapping is safe here only because the surface refuses to
            // be narrower than tableMinimumWidth: NSTextTable cannot lay out a
            // row holding a token wider than its column -- it widens that
            // column, gives up on the rest, and stacks every remaining cell at
            // the widened column's origin, so the row reads as overlapping
            // glyphs. No column-width spelling avoids it; only never posing
            // the question does.
            para.lineBreakMode = .byWordWrapping
            para.textBlocks = [block]
            para.alignment = nsAlignment(col, alignments)
            let cellAttr = NSMutableAttributedString(
                attributedString: tableCell(cellText, base: base,
                                            style: style, images: images))
            if cellAttr.length == 0 {
                cellAttr.append(NSAttributedString(
                    string: "\u{00A0}",
                    attributes: [.font: base,
                                 .foregroundColor: platformDefaultTextColor]))
            }
            let full = NSRange(location: 0, length: cellAttr.length)
            cellAttr.addAttribute(.paragraphStyle, value: para, range: full)
            cellAttr.addAttribute(atomicKindKey,
                                  value: AtomicKind.table.rawValue,
                                  range: full)
            cellAttr.addAttribute(atomicIdKey, value: atomicId, range: full)
            m.append(cellAttr)
            m.append(NSAttributedString(string: "\n"))
        }
        return m
    }

    private static func nsAlignment(_ col: Int,
                                    _ aligns: [Markdown.Alignment])
        -> NSTextAlignment {
        let a = col < aligns.count ? aligns[col] : .none
        let result: NSTextAlignment
        switch a {
            case .center: result = .center
            case .right: result = .right
            default: result = .left
        }
        return result
    }
}
#endif

#if os(macOS)
import AppKit

extension DocumentText {

    static func table(headers: [String], rows: [[String]],
                      alignments: [Markdown.Alignment], style: MarkdownStyle,
                      images: [URL: PlatformImage]) -> NSAttributedString {
        let m = NSMutableAttributedString()
        let cols = max(headers.count, rows.map { r in r.count }.max() ?? 0)
        if cols > 0 {
            let atomicId = UUID().uuidString
            let textTable = NSTextTable()
            textTable.numberOfColumns = cols
            textTable.layoutAlgorithm = .fixedLayoutAlgorithm
            let shares = TableMetrics.pointWidths(headers: headers, rows: rows,
                                                  available: 100)
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
            block.setWidth(6, type: .absoluteValueType, for: .padding)
            block.backgroundColor = tint
            let para = NSMutableParagraphStyle()
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

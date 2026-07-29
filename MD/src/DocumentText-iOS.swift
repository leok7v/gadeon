#if os(iOS)
import UIKit

extension DocumentText {

    static func table(headers: [String], rows: [[String]],
                      alignments: [Markdown.Alignment], style: MarkdownStyle,
                      images: [URL: PlatformImage]) -> NSAttributedString {
        let m = NSMutableAttributedString()
        let cols = max(headers.count, rows.map { r in r.count }.max() ?? 0)
        if cols > 0 {
            let atomicId = UUID().uuidString
            let widths = TableMetrics.pointWidths(headers: headers, rows: rows,
                                                  available: 320)
            var stops: [NSTextTab] = []
            var x: CGFloat = 0
            for (col, w) in widths.enumerated() {
                x += w
                stops.append(NSTextTab(textAlignment: tabAlignment(col,
                             alignments), location: x))
            }
            if !headers.isEmpty {
                m.append(tableRow(headers, stops: stops, bold: true,
                                  tint: platformWhite(0.5, alpha: 0.14),
                                  atomicId: atomicId, style: style,
                                  images: images))
            }
            for (idx, row) in rows.enumerated() {
                let tint: PlatformColor = idx % 2 == 1
                    ? platformWhite(0.5, alpha: 0.07) : platformClearColor
                m.append(tableRow(row, stops: stops, bold: false, tint: tint,
                                  atomicId: atomicId, style: style,
                                  images: images))
            }
            // One CONTIGUOUS atomic id over the whole table content so the copy
            // grouping is ONE block. Stamped BEFORE the trailing block newline
            // is appended, so a drag ending just past the table does not grab
            // the blank line and the copy overlay hugs the table. (Kind is
            // already contiguous: each row stamps it over its full range.)
            let content = NSRange(location: 0, length: m.length)
            m.addAttribute(atomicIdKey, value: atomicId, range: content)
            m.addAttribute(atomicCopyKey,
                           value: TableMetrics.serializeMonospaced(
                               headers: headers, rows: rows),
                           range: content)
            m.append(NSAttributedString(string: "\n"))
        }
        return m
    }

    private static func tableRow(_ cells: [String], stops: [NSTextTab],
                                 bold: Bool, tint: PlatformColor,
                                 atomicId: String, style: MarkdownStyle,
                                 images: [URL: PlatformImage])
        -> NSAttributedString {
        let para = NSMutableParagraphStyle()
        para.tabStops = stops
        para.lineBreakMode = .byTruncatingTail
        let base = bold ? boldFont(of: bodyFont(style)) : bodyFont(style)
        let m = NSMutableAttributedString()
        for (i, cell) in cells.enumerated() {
            if i > 0 {
                m.append(NSAttributedString(string: "\t",
                                            attributes: [.font: base]))
            }
            m.append(tableCell(cell, base: base, style: style,
                               images: images))
        }
        m.append(NSAttributedString(string: "\n", attributes: [.font: base]))
        let full = NSRange(location: 0, length: m.length)
        m.addAttribute(.paragraphStyle, value: para, range: full)
        m.addAttribute(.backgroundColor, value: tint, range: full)
        m.addAttribute(atomicKindKey, value: AtomicKind.table.rawValue,
                       range: full)
        m.addAttribute(atomicIdKey, value: atomicId, range: full)
        return m
    }

    private static func tabAlignment(_ col: Int,
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

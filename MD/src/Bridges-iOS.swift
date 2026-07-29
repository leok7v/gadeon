#if os(iOS)
import SwiftUI
import UIKit

// The background color a run carried BEFORE a find tint replaced it, stashed in
// the text storage so clearing a find restores code / table tints. NSNull marks
// a run that had no background. Lives in the storage (not a parallel array) so
// its ranges shift with edits and can never dangle.
private let findBaseBgKey = NSAttributedString.Key("MD.find.baseBg")

// iOS backing for NativeText. Self-sizing, non-scrolling, selectable.
extension NativeText: UIViewRepresentable {

    func makeUIView(context: Context) -> ResizingTextView {
        let v = ResizingTextView(usingTextLayoutManager: false)
        v.isEditable = false
        v.isSelectable = selectable
        // A whole-document surface scrolls internally (fills its frame, so Find
        // can scrollRangeToVisible); a per-block chat surface self-sizes and
        // the transcript scrolls it.
        v.scrolls = scrolls
        v.isScrollEnabled = scrolls
        v.backgroundColor = .clear
        v.textContainerInset = .zero
        v.textContainer.lineFragmentPadding = 0
        v.adjustsFontForContentSizeCategory = true
        v.linkTextAttributes = [
            .foregroundColor: UIColor.link,
            .underlineStyle: NSUnderlineStyle.single.rawValue,
        ]
        v.setContentCompressionResistancePriority(.defaultLow,
                                                  for: .horizontal)
        v.nowrap = nowrap
        if nowrap {
            v.textContainer.widthTracksTextView = false
            v.textContainer.size = CGSize(
                width: CGFloat.greatestFiniteMagnitude,
                height: CGFloat.greatestFiniteMagnitude)
        }
        // The OS find navigator (hardware Cmd+F) rides alongside the
        // controller-driven find; either presents and selects matches.
        if find != nil { v.isFindInteractionEnabled = true }
        v.findId = findId
        v.findController = find
        if let find, let findId { find.register(findId, v) }
        return v
    }

    static func dismantleUIView(_ v: ResizingTextView, coordinator: ()) {
        if let id = v.findId { v.findController?.unregister(id, v) }
    }

    func updateUIView(_ v: ResizingTextView, context: Context) {
        v.nowrap = nowrap
        v.isSelectable = selectable
        let next = MarkdownDiag.timed("resolve") { resolved() }
        MarkdownDiag.timed("setText len=\(next.length)") {
            v.applyResolved(next)
        }
    }

    // Height computed for the PROPOSED width, not left to the intrinsic-
    // size dance: inside a LazyVStack (the chat transcript) the
    // invalidation after the width settles is ignored and every block
    // keeps its first unconstrained one-line measurement -- each
    // paragraph rendered as a single clipped line. nowrap (code inside a
    // horizontal scroller) keeps the intrinsic natural width.
    func sizeThatFits(_ proposal: ProposedViewSize,
                      uiView v: ResizingTextView,
                      context: Context) -> CGSize? {
        var result: CGSize? = nil
        // A scrolling surface accepts the proposed size (fills its frame); only
        // the self-sizing surface computes a height.
        if !nowrap, !scrolls, let w = proposal.width, w > 0, w.isFinite {
            let fit = MarkdownDiag.timed("size len=\(v.textStorage.length)") {
                v.sizeThatFits(
                    CGSize(width: w, height: .greatestFiniteMagnitude))
            }
            result = CGSize(width: w, height: ceil(fit.height))
        }
        return result
    }

    final class ResizingTextView: UITextView, FindableTextView {

        var nowrap: Bool = false
        var scrolls: Bool = false
        var findId: UUID?
        weak var findController: MarkdownFindController?
        private var lastWidth: CGFloat = 0
        private var findMatches: [NSRange] = []
        private var activeIndex: Int? = nil
        private var findQuery = ""
        private var findCaseSensitive = false

        var liveFindCount: Int { findMatches.count }

        // The active match's vertical center as a fraction of the laid-out
        // height, so the transcript can scroll the exact line into view.
        func activeMatchFraction() -> CGFloat? {
            var result: CGFloat? = nil
            if let i = activeIndex, i >= 0, i < findMatches.count,
               NSMaxRange(findMatches[i]) <= textStorage.length {
                let gr = layoutManager.glyphRange(
                    forCharacterRange: findMatches[i],
                    actualCharacterRange: nil)
                let rect = layoutManager.boundingRect(forGlyphRange: gr,
                                                      in: textContainer)
                let h = layoutManager.usedRect(for: textContainer).height
                if h > 0 { result = min(1, max(0, rect.midY / h)) }
            }
            return result
        }

        // Re-render to `next` while a Find may be active. Find tints are REAL
        // .backgroundColor in the storage (UIKit's NSLayoutManager has no
        // temporary attributes), so they are stripped BEFORE the diff:
        // otherwise applyIncremental reads the tints as changed attributes and
        // re-splices from the first match to the end (losing its O(delta)
        // contract), AND restoreBackgrounds would later write the saved colors
        // at ranges the splice has since shifted (smearing code/table tints).
        // Strip at valid pre-edit ranges, splice clean, then recompute and
        // re-tint from the new text.
        func applyResolved(_ next: NSAttributedString) {
            let active = !findQuery.isEmpty
            if active { clearHighlights() }
            if !textStorage.isEqual(to: next) {
                textStorage.beginEditing()
                applyIncremental(textStorage, next)
                textStorage.endEditing()
                invalidateIntrinsicContentSize()
            }
            if active { reapplyFind() }
        }

        // Concrete (never a dynamic system color, which resolves to nil off a
        // trait environment and aborts the attribute set).
        private var findTint: UIColor {
            UIColor(red: 1.0, green: 0.84, blue: 0.2, alpha: 0.35)
        }
        private var activeTint: UIColor {
            UIColor(red: 1.0, green: 0.6, blue: 0.0, alpha: 0.6)
        }

        // Highlight EVERY match; the controller activates one later. No
        // selection here, so other bubbles do not each grab a selection.
        func findAll(_ query: String, caseSensitive: Bool) -> Int {
            findQuery = query
            findCaseSensitive = caseSensitive
            findMatches = markdownFindRanges(in: text, query: query,
                                             caseSensitive: caseSensitive)
            activeIndex = nil
            highlightAll()
            return findMatches.count
        }

        func setActive(_ localIndex: Int?) {
            activeIndex = localIndex
            highlightAll()
            if let i = localIndex, i >= 0, i < findMatches.count,
               NSMaxRange(findMatches[i]) <= textStorage.length {
                selectedRange = findMatches[i]
                scrollRangeToVisible(findMatches[i])
            } else {
                selectedRange = NSRange(location: 0, length: 0)
            }
        }

        func clearFind() {
            findQuery = ""
            findMatches = []
            activeIndex = nil
            clearHighlights()
        }

        func reapplyFind() {
            if !findQuery.isEmpty {
                findMatches = markdownFindRanges(
                    in: text, query: findQuery,
                    caseSensitive: findCaseSensitive)
                if let a = activeIndex, a >= findMatches.count {
                    activeIndex = nil
                }
                highlightAll()
            }
        }

        // UIKit's NSLayoutManager has no temporary attributes, so find tints are
        // real .backgroundColor. Each tinted run's prior background is stashed
        // IN the storage under findBaseBgKey (NSNull marks "no background"), so
        // clearing restores the code / table tints. Keeping it in the storage --
        // not a parallel array whose offsets go stale after a splice and could
        // be raced into a bad buffer -- is what makes this safe. Ranges are
        // collected into a LOCAL array before mutating, so the enumeration is
        // never modified underneath itself.
        private func clearHighlights() {
            let full = NSRange(location: 0, length: textStorage.length)
            var restores: [(range: NSRange, base: Any)] = []
            textStorage.enumerateAttribute(findBaseBgKey, in: full,
                                           options: []) { val, r, _ in
                if let val { restores.append((r, val)) }
            }
            for e in restores {
                if let color = e.base as? UIColor {
                    textStorage.addAttribute(.backgroundColor, value: color,
                                             range: e.range)
                } else {
                    textStorage.removeAttribute(.backgroundColor,
                                                range: e.range)
                }
                textStorage.removeAttribute(findBaseBgKey, range: e.range)
            }
        }

        private func highlightAll() {
            clearHighlights()
            let len = textStorage.length
            for (i, r) in findMatches.enumerated() where NSMaxRange(r) <= len {
                let tint = i == activeIndex ? activeTint : findTint
                var bases: [(range: NSRange, base: Any)] = []
                textStorage.enumerateAttribute(.backgroundColor, in: r,
                                               options: []) { val, sub, _ in
                    bases.append((sub, val ?? NSNull()))
                }
                for b in bases {
                    textStorage.addAttribute(findBaseBgKey, value: b.base,
                                             range: b.range)
                }
                textStorage.addAttribute(.backgroundColor, value: tint,
                                         range: r)
            }
        }

        override var intrinsicContentSize: CGSize {
            // A scrolling surface fills whatever frame it is given; only the
            // self-sizing per-block surface reports its content height.
            if scrolls {
                return CGSize(width: UIView.noIntrinsicMetric,
                              height: UIView.noIntrinsicMetric)
            }
            layoutManager.ensureLayout(for: textContainer)
            let r = layoutManager.usedRect(for: textContainer)
            let h = r.height + textContainerInset.top +
                    textContainerInset.bottom
            let w = nowrap
                ? r.width + textContainerInset.left + textContainerInset.right
                : UIView.noIntrinsicMetric
            return CGSize(width: w, height: h)
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            if bounds.size.width != lastWidth {
                lastWidth = bounds.size.width
                invalidateIntrinsicContentSize()
            }
        }
    }
}
#endif

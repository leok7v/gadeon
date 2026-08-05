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
        let v = ResizingTextView.textKit1()
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
        v.setSpoken(speaking)
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

        // A TextKit 1 view, built through UITextView's DESIGNATED initializer.
        //
        // NOT `init(usingTextLayoutManager: false)`, which asks for the same
        // TextKit 1 stack and looks equivalent. Constructing a Swift subclass
        // through THAT one never runs the subclass's stored-property
        // initializers: every property below keeps the zeroed memory ObjC
        // calloc'd it with, so `findMatches` holds a NULL buffer where an
        // empty Array must point at the shared empty singleton -- and reading
        // its count faults at address 0x10, which is the crash this replaced.
        //
        // MEASURED on iOS 18, the two side by side: usingTextLayoutManager
        // gives buffer 0x0 and activeIndex .some(0) where the declared
        // defaults are [] and nil; frame:textContainer: gives both correctly.
        // Both leave textLayoutManager nil and the container tracking width,
        // so this is TextKit 1 either way and nothing about layout moves.
        // AppKit's NSTextView does NOT share the defect, which is why the
        // macOS twin never crashed and cannot stand in for a test of this.
        static func textKit1() -> ResizingTextView {
            let storage = NSTextStorage()
            let layout = NSLayoutManager()
            let container = NSTextContainer(
                size: CGSize(width: 0,
                             height: CGFloat.greatestFiniteMagnitude))
            storage.addLayoutManager(layout)
            layout.addTextContainer(container)
            return ResizingTextView(frame: .zero, textContainer: container)
        }

        var nowrap: Bool = false
        var scrolls: Bool = false
        var findId: UUID?
        weak var findController: MarkdownFindController?
        private var lastWidth: CGFloat = 0
        private var findMatches: [NSRange] = []
        private var activeIndex: Int? = nil
        private var findQuery = ""
        private var findCaseSensitive = false
        private var spokenText: String?
        private var spokenRange: NSRange?

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
        // The spoken tint is a real background too, so it counts as "active"
        // here for exactly the same reason a find tint does: left in place it
        // would be read as a changed attribute and re-splice the whole tail.
        func applyResolved(_ next: NSAttributedString) {
            let active = !findQuery.isEmpty || spokenRange != nil
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
        // Cool against find's warm pair, so the two never read as the same
        // thing when a search happens to be open while a reply is spoken.
        private var spokenTint: UIColor {
            UIColor(red: 0.25, green: 0.55, blue: 1.0, alpha: 0.28)
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
            highlightAll()
        }

        func reapplyFind() {
            if !findQuery.isEmpty {
                findMatches = markdownFindRanges(
                    in: text, query: findQuery,
                    caseSensitive: findCaseSensitive)
                if let a = activeIndex, a >= findMatches.count {
                    activeIndex = nil
                }
            }
            locateSpoken()
            highlightAll()
        }

        func setSpoken(_ text: String?) {
            if text != spokenText {
                spokenText = text
                locateSpoken()
                highlightAll()
            }
        }

        // A literal search, so a sentence the speech layer reshaped simply
        // does not tint rather than tinting the wrong one.
        private func locateSpoken() {
            let ns = text as NSString
            var found: NSRange? = nil
            if let want = spokenText, !want.isEmpty {
                let r = ns.range(of: want)
                if r.location != NSNotFound { found = r }
            }
            spokenRange = found
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

        // The ONE writer of tinted backgrounds. The spoken sentence is painted
        // first and find over it, so a search stays visible through a reply
        // being read aloud; both go through the same base-colour stash, which
        // is what lets clearHighlights put the code / table tints back.
        //
        // RE-ENTRANCY, not threading, is the hazard -- everything here is
        // @MainActor. Writing the storage invalidates layout, which can bring
        // SwiftUI straight back through updateUIView -> applyResolved ->
        // reapplyFind, and that REASSIGNS findMatches. A loop reading the
        // property across those writes was then left holding a released
        // buffer: the crash landed inside the iterator rather than the body.
        //
        // Two things stop it. The tint set is computed from LOCAL copies
        // before anything is written, so no live enumeration spans a write;
        // and the writes are grouped, so layout is invalidated once at
        // endEditing rather than after every attribute.
        private func highlightAll() {
            let matches = findMatches
            let active = activeIndex
            let spoken = spokenRange
            textStorage.beginEditing()
            clearHighlights()
            let len = textStorage.length
            var tinted: [(range: NSRange, tint: UIColor)] = []
            if let r = spoken, NSMaxRange(r) <= len {
                tinted.append((r, spokenTint))
            }
            for (i, r) in matches.enumerated() where NSMaxRange(r) <= len {
                tinted.append((r, i == active ? activeTint : findTint))
            }
            for entry in tinted {
                var bases: [(range: NSRange, base: Any)] = []
                textStorage.enumerateAttribute(.backgroundColor,
                                               in: entry.range,
                                               options: []) { val, sub, _ in
                    bases.append((sub, val ?? NSNull()))
                }
                for b in bases where
                    textStorage.attribute(findBaseBgKey, at: b.range.location,
                                          effectiveRange: nil) == nil {
                    textStorage.addAttribute(findBaseBgKey, value: b.base,
                                             range: b.range)
                }
                textStorage.addAttribute(.backgroundColor, value: entry.tint,
                                         range: entry.range)
            }
            textStorage.endEditing()
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

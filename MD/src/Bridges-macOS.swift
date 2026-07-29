#if os(macOS)
import SwiftUI
import AppKit

// macOS backing for NativeText. Selection expands to whole atomic units
// (code / table / image) so a drag never cuts a highlighted block in half.
extension NativeText: NSViewRepresentable {

    final class Coordinator: NSObject, NSTextViewDelegate {

        private var anchorScope: NSRange? = nil
        var findId: UUID?
        weak var findController: MarkdownFindController?

        func textView(_ tv: NSTextView, clickedOnLink link: Any,
                      at: Int) -> Bool {
            var url: URL? = nil
            if let u = link as? URL { url = u }
            else if let s = link as? String { url = URL(string: s) }
            var handled = false
            if let url {
                NSWorkspace.shared.open(url)
                handled = true
            }
            return handled
        }

        func textView(_ textView: NSTextView,
                      willChangeSelectionFromCharacterRange old: NSRange,
                      toCharacterRange new: NSRange) -> NSRange {
            var result = new
            if let storage = textView.textStorage {
                if new.length == 0 {
                    anchorScope = atomicScope(at: new.location, in: storage)
                } else if let scope = anchorScope {
                    result = extend(new, scope: scope, in: storage)
                } else {
                    result = expand(new, in: storage)
                }
            }
            return result
        }

        private func extend(_ new: NSRange, scope: NSRange,
                            in storage: NSTextStorage) -> NSRange {
            let endLo = new.location
            let endHi = new.location + new.length
            let scopeLo = scope.location
            let scopeHi = scope.location + scope.length
            let inside = endLo >= scopeLo && endHi <= scopeHi
            var result = new
            if !inside {
                let lo = min(endLo, scopeLo)
                let hi = max(endHi, scopeHi)
                result = expand(NSRange(location: lo, length: hi - lo),
                                in: storage)
            }
            return result
        }

        private func atomicScope(at pos: Int,
                                 in storage: NSTextStorage) -> NSRange? {
            var result: NSRange? = nil
            if pos >= 0, pos < storage.length {
                var effective = NSRange(location: 0, length: 0)
                let value = storage.attribute(atomicKindKey, at: pos,
                                              effectiveRange: &effective)
                if value != nil { result = effective }
            }
            return result
        }

        private func expand(_ range: NSRange,
                            in storage: NSTextStorage) -> NSRange {
            var lo = range.location
            var hi = range.location + range.length
            storage.enumerateAttribute(atomicKindKey, in: range,
                                       options: []) { value, r, _ in
                if value != nil {
                    if r.location < lo { lo = r.location }
                    let end = r.location + r.length
                    if end > hi { hi = end }
                }
            }
            return NSRange(location: lo, length: hi - lo)
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> ResizingTextView {
        let v = ResizingTextView()
        v.delegate = context.coordinator
        v.isEditable = false
        v.isSelectable = selectable
        v.drawsBackground = false
        v.backgroundColor = .clear
        v.textContainerInset = .zero
        v.textContainer?.lineFragmentPadding = 0
        v.textContainer?.widthTracksTextView = !nowrap
        if nowrap {
            v.textContainer?.containerSize = NSSize(
                width: CGFloat.greatestFiniteMagnitude,
                height: CGFloat.greatestFiniteMagnitude)
        }
        v.isVerticallyResizable = true
        v.isHorizontallyResizable = nowrap
        v.setContentCompressionResistancePriority(.defaultLow,
                                                  for: .horizontal)
        v.linkTextAttributes = [
            .foregroundColor: NSColor.linkColor,
            .underlineStyle: NSUnderlineStyle.single.rawValue,
            .cursor: NSCursor.pointingHand,
        ]
        context.coordinator.findId = findId
        context.coordinator.findController = find
        if let find, let findId { find.register(findId, v) }
        return v
    }

    static func dismantleNSView(_ v: ResizingTextView,
                                coordinator: Coordinator) {
        if let id = coordinator.findId {
            coordinator.findController?.unregister(id, v)
        }
    }

    func updateNSView(_ v: ResizingTextView, context: Context) {
        v.nowrap = nowrap
        v.isSelectable = selectable
        let next = resolved()
        if let ts = v.textStorage, !ts.isEqual(to: next) {
            ts.beginEditing()
            applyIncremental(ts, next)
            ts.endEditing()
            v.invalidateIntrinsicContentSize()
            v.reapplyFind()
        }
    }

    final class ResizingTextView: NSTextView, FindableTextView {

        var nowrap: Bool = false
        private var lastBounds: NSSize = .zero
        // Copy buttons keyed by the block's atomic id, and each block's current
        // corner rect for click hit-testing. Keeping the buttons across layouts
        // (rather than tearing down) preserves a copy-confirmation checkmark
        // through a streaming reflow.
        private var copyButtons: [String: CopyRunButton] = [:]
        private var copyRects: [(id: String, rect: NSRect)] = []
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
               let lm = layoutManager, let tc = textContainer,
               NSMaxRange(findMatches[i]) <= (textStorage?.length ?? 0) {
                let gr = lm.glyphRange(forCharacterRange: findMatches[i],
                                       actualCharacterRange: nil)
                let rect = lm.boundingRect(forGlyphRange: gr, in: tc)
                let h = lm.usedRect(for: tc).height
                if h > 0 { result = min(1, max(0, rect.midY / h)) }
            }
            return result
        }
        // Find highlight tints. Concrete sRGB (never a dynamic system color,
        // which resolves to nil off a trait environment) so highlighting can
        // never abort on a nil value.
        private var findTint: NSColor {
            NSColor(srgbRed: 1.0, green: 0.84, blue: 0.2, alpha: 0.35)
        }
        private var activeTint: NSColor {
            NSColor(srgbRed: 1.0, green: 0.6, blue: 0.0, alpha: 0.6)
        }

        // Highlight EVERY match; the controller activates one later. No
        // selection here, so the other bubbles do not each grab a selection.
        func findAll(_ query: String, caseSensitive: Bool) -> Int {
            findQuery = query
            findCaseSensitive = caseSensitive
            findMatches = markdownFindRanges(in: string, query: query,
                                             caseSensitive: caseSensitive)
            activeIndex = nil
            highlightAll()
            return findMatches.count
        }

        // Select + scroll one match into view (the active one), or clear the
        // active selection with nil while keeping the highlights.
        func setActive(_ localIndex: Int?) {
            activeIndex = localIndex
            highlightAll()
            let len = textStorage?.length ?? 0
            if let i = localIndex, i >= 0, i < findMatches.count,
               NSMaxRange(findMatches[i]) <= len {
                setSelectedRange(findMatches[i])
                scrollRangeToVisible(findMatches[i])
            } else {
                setSelectedRange(NSRange(location: 0, length: 0))
            }
        }

        func clearFind() {
            findQuery = ""
            findMatches = []
            activeIndex = nil
            if let lm = layoutManager, let ts = textStorage {
                lm.removeTemporaryAttribute(.backgroundColor,
                    forCharacterRange: NSRange(location: 0, length: ts.length))
            }
        }

        func reapplyFind() {
            if !findQuery.isEmpty {
                findMatches = markdownFindRanges(
                    in: string, query: findQuery,
                    caseSensitive: findCaseSensitive)
                if let a = activeIndex, a >= findMatches.count {
                    activeIndex = nil
                }
                highlightAll()
            }
        }

        // TEMPORARY attributes, not real .backgroundColor: they layer over the
        // text without mutating the storage, so code / table backgrounds
        // survive a find AND applyIncremental's attribute diff is undisturbed.
        private func highlightAll() {
            if let lm = layoutManager, let ts = textStorage {
                let full = NSRange(location: 0, length: ts.length)
                lm.removeTemporaryAttribute(.backgroundColor,
                                            forCharacterRange: full)
                for (i, r) in findMatches.enumerated()
                where NSMaxRange(r) <= ts.length {
                    lm.setTemporaryAttributes(
                        [.backgroundColor: i == activeIndex
                            ? activeTint : findTint],
                        forCharacterRange: r)
                }
            }
        }

        override var intrinsicContentSize: NSSize {
            var result = super.intrinsicContentSize
            if let lm = layoutManager, let tc = textContainer {
                lm.ensureLayout(for: tc)
                let r = lm.usedRect(for: tc)
                let inset = textContainerInset
                let w = nowrap ? r.width + inset.width * 2
                               : NSView.noIntrinsicMetric
                result = NSSize(width: w, height: r.height + inset.height * 2)
            }
            return result
        }

        override func layout() {
            super.layout()
            if bounds.size != lastBounds {
                lastBounds = bounds.size
                invalidateIntrinsicContentSize()
            }
            rebuildCopyOverlays()
        }

        // A corner Copy button pinned to each atomic code / table block, so a
        // reader copies the WHOLE block's source (atomicCopyKey) in one click
        // -- the affordance the per-block SwiftUI code / table used to carry.
        // Buttons are REUSED across layouts, keyed by the block's atomic id, so
        // a copy checkmark survives a streaming reflow. The button is
        // decorative (it never intercepts the mouse, see CopyRunButton.hitTest)
        // so a text-selection drag can start anywhere over the block; this view
        // detects a corner click in mouseDown and drives the copy, reading the
        // block's CURRENT source by id so a still-streaming block copies its
        // latest text. Images carry an atomic id but no copy source, so no
        // button.
        func rebuildCopyOverlays() {
            copyRects = []
            var live: Set<String> = []
            if let lm = layoutManager, let tc = textContainer,
               let ts = textStorage {
                lm.ensureLayout(for: tc)
                let origin = textContainerOrigin
                ts.enumerateAttribute(
                    atomicIdKey,
                    in: NSRange(location: 0, length: ts.length),
                    options: []) { value, range, _ in
                    let id = value as? String
                    let hasCopy = id != nil && ts.attribute(
                        atomicCopyKey, at: range.location,
                        effectiveRange: nil) != nil
                    if let id, hasCopy {
                        let gr = lm.glyphRange(forCharacterRange: range,
                                               actualCharacterRange: nil)
                        let block = lm.boundingRect(forGlyphRange: gr, in: tc)
                        // Center the button on the FIRST line fragment, not the
                        // block's overall top: the symbol is centered in its
                        // frame, so anchoring to the block top made it read as
                        // sitting on the first line's baseline (worse for
                        // tables, whose first row sits below cell padding).
                        let line = lm.lineFragmentUsedRect(
                            forGlyphAt: gr.location, effectiveRange: nil)
                        let x = block.maxX + origin.x - 26
                        let y = line.minY + origin.y + (line.height - 22) / 2
                        let frame = NSRect(x: x, y: y, width: 22, height: 22)
                        let btn = copyButtons[id] ?? makeCopyButton(id)
                        btn.frame = frame
                        copyButtons[id] = btn
                        live.insert(id)
                        copyRects.append((id, frame))
                    }
                }
            }
            for (id, btn) in copyButtons where !live.contains(id) {
                btn.removeFromSuperview()
                copyButtons[id] = nil
            }
        }

        private func makeCopyButton(_ id: String) -> CopyRunButton {
            let btn = CopyRunButton()
            btn.atomicId = id
            btn.autoresizingMask = [.minXMargin]
            addSubview(btn)
            return btn
        }

        // A click landing in a block's corner button copies that block; a drag
        // (selection) leaves a non-empty selection, so it is never a copy. The
        // button is decorative, so the down-point is tested here.
        override func mouseDown(with event: NSEvent) {
            let p = convert(event.locationInWindow, from: nil)
            let hit = copyRects.first { rc in NSPointInRect(p, rc.rect) }?.id
            super.mouseDown(with: event)
            if let hit, selectedRange().length == 0 { copyBlock(hit) }
        }

        // Read the block's CURRENT source by locating its atomic id in the live
        // storage (ranges shift as tokens stream, but the id is stable), so the
        // copy is never stale.
        private func copyBlock(_ id: String) {
            var copy: String? = nil
            if let ts = textStorage {
                ts.enumerateAttribute(
                    atomicIdKey,
                    in: NSRange(location: 0, length: ts.length),
                    options: []) { value, range, stop in
                    if value as? String == id {
                        copy = ts.attribute(atomicCopyKey, at: range.location,
                                            effectiveRange: nil) as? String
                        stop.pointee = true
                    }
                }
            }
            if let copy {
                platformSetClipboardString(copy)
                copyButtons[id]?.flashDone()
            }
        }
    }
}

// A small Copy button overlaid at a code / table block's corner. Decorative:
// it never intercepts the mouse (hitTest returns nil) so a text-selection drag
// can begin anywhere over the block; the hosting text view detects a corner
// click and drives the copy, then calls flashDone for the brief checkmark.
private final class CopyRunButton: NSButton {
    var atomicId = ""

    init() {
        super.init(frame: .zero)
        isBordered = false
        bezelStyle = .regularSquare
        imagePosition = .imageOnly
        imageScaling = .scaleProportionallyDown
        contentTintColor = .secondaryLabelColor
        toolTip = "Copy"
        setSymbol("doc.on.doc")
    }

    required init?(coder: NSCoder) { nil }

    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    private func setSymbol(_ name: String) {
        image = NSImage(systemSymbolName: name,
                        accessibilityDescription: "Copy")
    }

    func flashDone() {
        setSymbol("checkmark")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            [weak self] in self?.setSymbol("doc.on.doc")
        }
    }
}
#endif

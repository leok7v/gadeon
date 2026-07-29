import SwiftUI
import UIKit

// iOS counterpart of the macOS prompt editor (parallel symbol, no #if os()). A
// UITextView inserts at the caret, scrolls smoothly, and grows
// minLines..maxLines via sizeThatFits. The soft keyboard has no Shift+Return,
// so Return inserts a newline and Send submits; onSubmit is kept for API
// parity.

struct PromptEditor: UIViewRepresentable {
    @Binding var text: String
    @Binding var focused: Bool
    @Binding var caret: Int
    var disabled: Bool
    var minLines: Int
    var maxLines: Int
    var onSubmit: () -> Void
    // Symbol parity with macOS (the shared Composer passes it). Unused on iOS:
    // the field refuses drops (willBecomeEditableForDrop -> .no) and iPad
    // in-field drag-drop is a won't-do; the chat handler owns [+]/Photos.
    var onDropFiles: ([URL]) -> Void = { _ in }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UITextView {
        let tv = UITextView()
        tv.delegate = context.coordinator
        // Refuse drops on the text view so the chat's drop handler owns them
        // (an @reference at the caret) rather than the field inserting content.
        tv.textDropDelegate = context.coordinator
        tv.font = UIFont.preferredFont(forTextStyle: .body)
        tv.backgroundColor = .clear
        tv.textContainerInset = UIEdgeInsets(
            top: 2, left: 0, bottom: 2, right: 0)
        tv.textContainer.lineFragmentPadding = 0
        tv.isScrollEnabled = true
        tv.text = text
        return tv
    }

    func updateUIView(_ tv: UITextView, context: Context) {
        context.coordinator.parent = self
        // On an external change (send/clear, or a dropped reference) put the
        // caret where the model wants it; the compare keeps local typing from
        // resetting it each keystroke.
        if tv.text != text {
            tv.text = text
            let len = (text as NSString).length
            let loc = max(0, min(caret, len))
            if let pos = tv.position(from: tv.beginningOfDocument,
                                     offset: loc) {
                tv.selectedTextRange = tv.textRange(from: pos, to: pos)
            }
        }
        // Any first-responder-affecting UIKit call HERE re-enters SwiftUI's
        // responder graph mid-update -- an AttributeGraph cycle:
        // setEditable(false) resigns, becomeFirstResponder() acquires, both
        // walk canBecomeFirstResponder -> responderNode while this update
        // runs. So apply both on the next tick, off the update pass. Focus is
        // skipped while disabled so it does not fight the resign disabling
        // triggers.
        let editable = !disabled
        let wantFocus = focused && !disabled
        if tv.isEditable != editable
            || (wantFocus && !tv.isFirstResponder) {
            Task { @MainActor in
                if tv.isEditable != editable { tv.isEditable = editable }
                if wantFocus, !tv.isFirstResponder {
                    tv.becomeFirstResponder()
                }
            }
        }
    }

    // Grow with the text, clamped to [min, max] lines. Returning the size here
    // (not feeding a measured height back through a resizing binding) is what
    // keeps SwiftUI's AttributeGraph from cycling.

    func sizeThatFits(_ proposal: ProposedViewSize, uiView: UITextView,
                      context: Context) -> CGSize? {
        let font = uiView.font ?? UIFont.preferredFont(forTextStyle: .body)
        let width = proposal.width ?? 300
        let inset = uiView.textContainerInset.top +
            uiView.textContainerInset.bottom
        let line = font.lineHeight
        let fit = uiView.sizeThatFits(
            CGSize(width: width, height: .greatestFiniteMagnitude)).height
        let lo = ceil(CGFloat(minLines) * line) + inset
        let hi = ceil(CGFloat(maxLines) * line) + inset
        return CGSize(width: width, height: min(max(fit, lo), hi))
    }

    @MainActor final class Coordinator: NSObject, UITextViewDelegate,
                                        UITextDropDelegate {
        var parent: PromptEditor

        init(_ parent: PromptEditor) { self.parent = parent }

        func textViewDidChange(_ tv: UITextView) {
            if parent.text != tv.text { parent.text = tv.text }
        }

        func textViewDidChangeSelection(_ tv: UITextView) {
            if let range = tv.selectedTextRange {
                let loc = tv.offset(from: tv.beginningOfDocument,
                                    to: range.start)
                if parent.caret != loc { parent.caret = loc }
            }
        }

        func textViewDidBeginEditing(_ tv: UITextView) {
            if !parent.focused { parent.focused = true }
        }

        func textViewDidEndEditing(_ tv: UITextView) {
            if parent.focused { parent.focused = false }
        }

        func textView(_ textView: UITextView,
                      willBecomeEditableForDrop drop: UITextDroppable)
            -> UITextDropEditability {
            .no
        }
    }

}

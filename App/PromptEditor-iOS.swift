import SwiftUI
import UIKit

struct PromptEditor: UIViewRepresentable {
    @Binding var text: String
    @Binding var focused: Bool
    @Binding var caret: Int
    var disabled: Bool
    var minLines: Int
    var maxLines: Int
    // `scale` already carries the device's Dynamic Type ratio; `font`
    // below must stay the UNSCALED body size, or the category is
    // counted twice.
    var scale: CGFloat = 1
    var onSubmit: () -> Void
    // Unused on iOS (kept for symbol parity with the macOS twin); the
    // field refuses drops via willBecomeEditableForDrop.
    var onDropFiles: ([URL]) -> Void = { _ in }

    // The body point size at a given scale, shared with the SwiftUI
    // placeholder drawn over the field so the two cannot drift apart.
    static func points(_ scale: CGFloat) -> CGFloat {
        UIFont.labelFontSize * scale
    }

    private var font: UIFont {
        UIFont.systemFont(ofSize: PromptEditor.points(scale))
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UITextView {
        let tv = UITextView()
        tv.delegate = context.coordinator
        // Refuse drops on the text view so the chat's drop handler owns them
        // (an @reference at the caret) rather than the field inserting content.
        tv.textDropDelegate = context.coordinator
        tv.font = font
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
        if tv.font != font { tv.font = font }
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
        // A first-responder-affecting UIKit call here re-enters SwiftUI's
        // responder graph mid-update (an AttributeGraph cycle), so both
        // calls are deferred to the next tick.
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
        let font = uiView.font ?? self.font
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

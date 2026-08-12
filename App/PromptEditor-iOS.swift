import SwiftUI
import UIKit

struct PromptEditor: UIViewRepresentable {

    @Binding var text: String
    @Binding var focused: Bool
    @Binding var caret: Int
    var disabled: Bool
    var minLines: Int
    var maxLines: Int
    var scale: CGFloat = 1
    var hasHint = false
    var onSubmit: () -> Void
    var onAcceptHint: () -> Void = { }
    var onDropFiles: ([URL]) -> Void = { _ in }

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
        if tv.text != text {
            tv.text = text
            let len = (text as NSString).length
            let loc = max(0, min(caret, len))
            if let pos = tv.position(from: tv.beginningOfDocument,
                                     offset: loc) {
                tv.selectedTextRange = tv.textRange(from: pos, to: pos)
            }
        }
        // Editable and first-responder changes re-enter SwiftUI's responder
        // graph, so applying them here cycles the AttributeGraph.
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

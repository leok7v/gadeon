import AppKit
import SwiftUI

struct PromptEditor: NSViewRepresentable {

    @Binding var text: String
    @Binding var focused: Bool
    @Binding var caret: Int
    var disabled: Bool
    var minLines: Int
    var maxLines: Int
    var scale: CGFloat = 1
    var onSubmit: () -> Void
    var onDropFiles: ([URL]) -> Void = { _ in }

    static func points(_ scale: CGFloat) -> CGFloat {
        NSFont.preferredFont(forTextStyle: .body).pointSize * scale
    }

    private var font: NSFont {
        NSFont.systemFont(ofSize: PromptEditor.points(scale))
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let tv = SubmitTextView()
        tv.delegate = context.coordinator
        tv.onSubmit = { context.coordinator.parent.onSubmit() }
        tv.onDropFiles = { context.coordinator.parent.onDropFiles($0) }
        tv.isRichText = false
        tv.allowsUndo = true
        tv.drawsBackground = false
        tv.font = font
        tv.textContainerInset = NSSize(width: 0, height: 2)
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.textContainer?.widthTracksTextView = true
        tv.textContainer?.lineFragmentPadding = 0
        tv.string = text

        let scroll = NSScrollView()
        scroll.documentView = tv
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = false
        scroll.autohidesScrollers = true
        context.coordinator.textView = tv
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        if let tv = scroll.documentView as? SubmitTextView {
            if tv.string != text {
                tv.string = text
                let len = (text as NSString).length
                tv.setSelectedRange(
                    NSRange(location: max(0, min(caret, len)), length: 0))
            }
            if tv.font != font { tv.font = font }
            tv.isEditable = !disabled
            tv.isSelectable = !disabled
            if focused, tv.window != nil, tv.window?.firstResponder !== tv {
                tv.window?.makeFirstResponder(tv)
            }
        }
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: NSScrollView,
                      context: Context) -> CGSize? {
        let tv = nsView.documentView as? SubmitTextView
        let font = tv?.font ?? self.font
        let width = proposal.width ?? 300
        let inset = (tv?.textContainerInset.height ?? 2) * 2
        let line = ceil(font.ascender - font.descender + font.leading)
        let lo = ceil(CGFloat(minLines) * line) + inset
        let hi = ceil(CGFloat(maxLines) * line) + inset
        let measured = context.coordinator.height(
            of: tv?.string ?? "", width: max(1, width), font: font,
            atMost: hi - inset)
        return CGSize(width: width,
                      height: min(max(ceil(measured) + inset, lo), hi))
    }

    @MainActor final class Coordinator: NSObject, NSTextViewDelegate {

        var parent: PromptEditor
        weak var textView: SubmitTextView?

        init(_ parent: PromptEditor) { self.parent = parent }

        private struct Measured {
            let text: String
            let width: CGFloat
            let size: CGFloat
            let height: CGFloat
        }

        private var last: Measured?

        func height(of text: String, width: CGFloat, font: NSFont,
                    atMost ceiling: CGFloat) -> CGFloat {
            let size = font.pointSize
            var result = last?.height ?? 0
            if last?.text != text || last?.width != width
                || last?.size != size {
                result = Coordinator.measure(text, width: width, font: font,
                                             atMost: ceiling)
                last = Measured(text: text, width: width, size: size,
                                height: result)
            }
            return result
        }

        private static func measure(_ text: String, width: CGFloat,
                                    font: NSFont,
                                    atMost ceiling: CGFloat) -> CGFloat {
            let probe = String(text.prefix(overflowProbeLength))
            var result = boundingHeight(probe, width: width, font: font)
            if result < ceiling, text.count > overflowProbeLength {
                result = boundingHeight(text, width: width, font: font)
            }
            return result
        }

        // boundingRect drops a trailing empty line; the space keeps it.

        private static func boundingHeight(_ text: String, width: CGFloat,
                                           font: NSFont) -> CGFloat {
            let box = NSSize(width: width, height: .greatestFiniteMagnitude)
            let options: NSString.DrawingOptions = [.usesLineFragmentOrigin,
                                                    .usesFontLeading]
            let attributes: [NSAttributedString.Key: Any] = [.font: font]
            return ((text + " ") as NSString).boundingRect(
                with: box, options: options, attributes: attributes).height
        }

        private static let overflowProbeLength = 2048

        func textDidChange(_ notification: Notification) {
            if let tv = textView, parent.text != tv.string {
                parent.text = tv.string
            }
        }

        func textViewDidChangeSelection(_ notification: Notification) {
            if let tv = textView {
                let loc = tv.selectedRange().location
                if parent.caret != loc { parent.caret = loc }
            }
        }

        func textDidBeginEditing(_ notification: Notification) {
            if !parent.focused { parent.focused = true }
        }

        func textDidEndEditing(_ notification: Notification) {
            if parent.focused { parent.focused = false }
        }

    }

}

final class SubmitTextView: NSTextView {
    var onSubmit: (() -> Void)?
    // NSTextView takes file drops through its built-in text-drag machinery,
    // which registerForDraggedTypes does not reach.
    var onDropFiles: (([URL]) -> Void)?

    private func droppedFileURLs(_ sender: NSDraggingInfo) -> [URL] {
        (sender.draggingPasteboard.readObjects(
            forClasses: [NSURL.self],
            options: [.urlReadingFileURLsOnly: true]) as? [URL]) ?? []
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        droppedFileURLs(sender).isEmpty
            ? super.draggingEntered(sender) : .copy
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        droppedFileURLs(sender).isEmpty
            ? super.draggingUpdated(sender) : .copy
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        let urls = droppedFileURLs(sender)
        let handled = !urls.isEmpty
        if handled { onDropFiles?(urls) }
        return handled ? true : super.performDragOperation(sender)
    }

    override func doCommand(by selector: Selector) {
        let isReturn = selector == #selector(NSResponder.insertNewline(_:))
        let shift = NSApp.currentEvent?.modifierFlags.contains(.shift) ?? false
        if isReturn && !shift {
            onSubmit?()
        } else {
            super.doCommand(by: selector)
        }
    }

}

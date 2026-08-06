import AppKit
import SwiftUI

// AppKit-backed multiline prompt editor, replacing TextField(axis:.vertical):
// that appends a Shift+Return newline at the string END (not the caret) and
// scrolls choppily. NSTextView inserts at the caret, scrolls smoothly, owns
// paste, and grows minLines..maxLines via sizeThatFits. Return submits;
// Shift+Return breaks at the caret. SDK-split with PromptEditor-iOS.swift.

struct PromptEditor: NSViewRepresentable {

    @Binding var text: String
    @Binding var focused: Bool
    @Binding var caret: Int
    var disabled: Bool
    var minLines: Int
    var maxLines: Int
    // The app's text size. NSTextView takes a font, not an environment, and
    // preferredFont(forTextStyle:) is a fixed 13pt on this platform whatever
    // the app is set to -- so without this the field is the one control that
    // stays put while the card around it grows.
    var scale: CGFloat = 1
    var onSubmit: () -> Void
    // A file dropped ONTO the field: routed to the chat's attach handler (chip
    // + @reference) instead of the field pasting the path. iOS ignores it.
    var onDropFiles: ([URL]) -> Void = { _ in }

    // The body point size at a given scale, shared with the SwiftUI
    // placeholder drawn over the field so the two cannot drift apart.
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
        // x=0 origin so the SwiftUI placeholder overlay lines up.
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
            // Only when input changed under the field (send/clear, or a dropped
            // reference) -- comparing first keeps local typing from resetting
            // the caret each keystroke. On such a change, put the caret where
            // the model wants it (just after an inserted reference).
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

    // Grow with the text, clamped to [min, max] lines. Returning the size here
    // (not feeding a measured height back through a resizing binding) is what
    // keeps SwiftUI's AttributeGraph from cycling.

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: NSScrollView,
                      context: Context) -> CGSize? {
        let tv = nsView.documentView as? SubmitTextView
        let font = tv?.font ?? self.font
        let width = proposal.width ?? 300
        let inset = (tv?.textContainerInset.height ?? 2) * 2
        let line = ceil(font.ascender - font.descender + font.leading)
        // boundingRect drops a trailing empty line; a sentinel keeps that
        // line.
        let text = ((tv?.string ?? "") + " ") as NSString
        let measured = text.boundingRect(
            with: NSSize(width: max(1, width), height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: [.font: font]).height
        let lo = ceil(CGFloat(minLines) * line) + inset
        let hi = ceil(CGFloat(maxLines) * line) + inset
        return CGSize(width: width,
                      height: min(max(ceil(measured) + inset, lo), hi))
    }

    @MainActor final class Coordinator: NSObject, NSTextViewDelegate {

        var parent: PromptEditor
        weak var textView: SubmitTextView?

        init(_ parent: PromptEditor) { self.parent = parent }

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

        // Keep `focused` in step with the real first responder, so it does not
        // re-grab focus after the user clicks away.

        func textDidBeginEditing(_ notification: Notification) {
            if !parent.focused { parent.focused = true }
        }

        func textDidEndEditing(_ notification: Notification) {
            if parent.focused { parent.focused = false }
        }

    }

}

// NSTextView that sends a plain Return to onSubmit and lets Shift+Return (and
// everything else) fall through, so AppKit inserts the break at the caret.

final class SubmitTextView: NSTextView {
    var onSubmit: (() -> Void)?
    // A dropped FILE is routed here so the chat attaches it (chip + @reference
    // at the caret) instead of NSTextView pasting the file PATH as text. Trying
    // to make the field REFUSE drops does not work -- NSTextView accepts file
    // drops through its built-in text-drag machinery, not the overridable
    // registerForDraggedTypes -- so we intercept the drop instead. Non-file
    // drags (selected text) fall through to the default.
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

import SwiftUI

// The one Done. Settings and the debug view dismiss with the same word doing
// the same job, so it has ONE definition -- two hand-rolled copies is how they
// came to render differently, a system toolbar capsule in one and a bare
// tinted button in the other.
//
// On iOS it is deliberately NOT the system's toolbar treatment, which fills a
// glass capsule and sets the label white. Against this app's dark ground that
// is the hardest thing on the screen to find, and it is the only control here
// that is not tinted -- New Chat, Reset Zoom and every top-bar glyph are blue,
// so a white pill reads as a different app's button. Tinted text instead.
// (Reshaping the capsule is not an option: buttonBorderShape has no effect on
// it.)
//
// macOS keeps the system button. There a default-action button is SUPPOSED to
// be prominent, and a bare blue word in a window header would read as a link.

struct DoneButton: View {

    let action: () -> Void

    var body: some View {
        Button("Done", action: action)
            .modifier(TintedOnPhone())
            .keyboardShortcut(.defaultAction)
    }
}

private struct TintedOnPhone: ViewModifier {

    @ViewBuilder
    func body(content: Content) -> some View {
        if isOS {
            content.buttonStyle(.plain).foregroundStyle(Color.accentColor)
        } else {
            content
        }
    }
}

// Done in a navigation bar, WITHOUT the shared glass capsule iOS 26 puts
// behind a toolbar item. The capsule is drawn by the toolbar rather than by
// the button, which is why setting a plain button style does not remove it --
// only opting the item out of the shared background does.
//
// The availability branch repeats the item because a modifier cannot be
// applied conditionally to toolbar content any other way; the deployment
// target is older than the API.

struct DoneToolbar: ToolbarContent {

    let action: () -> Void

    var body: some ToolbarContent {
        if #available(iOS 26.0, macOS 26.0, *) {
            ToolbarItem(placement: .confirmationAction) {
                DoneButton(action: action)
            }
            .sharedBackgroundVisibility(.hidden)
        } else {
            ToolbarItem(placement: .confirmationAction) {
                DoneButton(action: action)
            }
        }
    }
}

// Esc dismisses too, with nothing on screen to say so. Kept apart from the
// visible button because a toolbar slot holds one control, and a hidden
// second one inside it is a layout surprise waiting to happen.

struct EscapeToClose: View {

    let action: () -> Void

    var body: some View {
        Button("", action: action)
            .keyboardShortcut(.cancelAction)
            .hidden()
    }
}

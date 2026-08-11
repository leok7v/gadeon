import SwiftUI

struct DoneButton: View {

    let action: () -> Void

    var body: some View {
        Button("Done", action: action)
            .modifier(TintedOnPhone())
            .keyboardShortcut(.defaultAction)
    }
}

// buttonBorderShape has no effect on the toolbar glass capsule.

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

// The capsule is toolbar-drawn; sharedBackgroundVisibility(.hidden) removes it.

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

struct EscapeToClose: View {

    let action: () -> Void

    var body: some View {
        Button("", action: action)
            .keyboardShortcut(.cancelAction)
            .hidden()
    }
}

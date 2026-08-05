import AppKit
import SwiftUI

@main struct GadeonApp: App {
    // Owned here (not in ContentView) so the View menu commands bind the same
    // instance.
    @State private var model = ChatModel()
    var body: some Scene {
        Window(Bundle.appName, id: "main") {
            ContentView(model: model).frame(minWidth: 800, minHeight: 600)
        }
        .commands { StatusBarCommands(model: model) }
    }
}

// View > Show / Hide Status Line, mirroring the Settings switch.
// macOS-only: iOS has no menu bar.
//
// A GROUP, never `CommandMenu("View")`: SwiftUI already builds a View menu
// (Enter Full Screen lives there), and a CommandMenu of the same name ADDS A
// SECOND ONE rather than merging into it. `after: .toolbar` is where the
// system puts this kind of item -- Finder's own Hide Status Bar sits in that
// section.
//
// A Show/Hide VERB rather than a checked noun, which is the platform's rule
// for a visibility toggle and what every system app does: the item names the
// action it will perform, so it reads correctly with no checkmark column.

struct StatusBarCommands: Commands {
    @Bindable var model: ChatModel
    var body: some Commands {
        CommandGroup(after: .toolbar) {
            Button(model.statusLine ? "Hide Status Line"
                                    : "Show Status Line") {
                model.statusLine.toggle()
            }
        }
    }
}

@MainActor func quitApp() { NSApp.terminate(nil) }

// Cross-platform clipboard write (iOS twin in App-iOS.swift), for the
// transcript tools' Copy without a #if os in the shared view.
func setClipboard(_ s: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(s, forType: .string)
}

// The iOS twin dismisses the soft keyboard; macOS has none.
@MainActor func hideSoftKeyboard() { }

var isOS: Bool { return false }

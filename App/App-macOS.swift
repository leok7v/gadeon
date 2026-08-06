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
        .commands { ViewCommands(model: model) }
    }
}

// The View menu's own items: zoom, then Show / Hide Status Bar mirroring the
// Settings switch. macOS-only: iOS has no menu bar.
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
//
// Zoom In is bound to Command-= and not Command-plus, which would need shift
// held as well. Every browser does the same and shows the friendlier glyph;
// SwiftUI displays whatever it is given, so the menu reads Command-=.
//
// "Reset Zoom" and not the "Actual Size" the browsers use, because its two
// siblings are verbs and a noun phrase reads oddly in a list of actions.
// Greyed out at the default size, which is where the feature says what its
// default IS: a disabled reset marks the resting position AND offers the way
// back to it, which no marker on a slider can do.
//
// These apply immediately, unlike the Settings slider, which defers to Done.
// The difference is not an inconsistency: here the chat is in full view, so
// there is something to watch; there the pane has replaced it.

// Every item carries a symbol. The system's own Enter Full Screen sits in this
// menu with one, which opens an icon column for the whole section -- so an
// item without a symbol does not read as plain, it reads as a hole where its
// symbol should be.
//
// SwiftUI puts a Settings item in the app menu whether or not an app has a
// Settings scene, and this app has none: its settings are a view inside the
// window (model.showSettings). Left alone, the item and its Command-comma are
// dead. Replacing the group is what puts the standard shortcut on the
// settings this app actually has.

struct ViewCommands: Commands {
    @Bindable var model: ChatModel
    var body: some Commands {
        CommandGroup(replacing: .appSettings) {
            Button {
                model.openSettings()
            } label: {
                Label("Settings\u{2026}", systemImage: "gearshape")
            }
            .keyboardShortcut(",", modifiers: .command)
        }
        CommandGroup(after: .toolbar) {
            Button { model.resetZoom() } label: {
                Label("Reset Zoom", systemImage: "1.magnifyingglass")
            }
            .keyboardShortcut("0", modifiers: .command)
            .disabled(model.atDefaultZoom)
            Button { model.zoomIn() } label: {
                Label("Zoom In", systemImage: "plus.magnifyingglass")
            }
            .keyboardShortcut("=", modifiers: .command)
            .disabled(!model.canZoomIn)
            Button { model.zoomOut() } label: {
                Label("Zoom Out", systemImage: "minus.magnifyingglass")
            }
            .keyboardShortcut("-", modifiers: .command)
            .disabled(!model.canZoomOut)
            Divider()
            Button {
                model.statusLine.toggle()
            } label: {
                Label(model.statusLine ? "Hide Status Bar"
                                       : "Show Status Bar",
                      systemImage: "rectangle.bottomthird.inset.filled")
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

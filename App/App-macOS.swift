import AppKit
import SwiftUI

@main struct GadeonApp: App {
    // Owned here, not in ContentView, so ViewCommands binds the same instance.
    @State private var model = ChatModel()
    var body: some Scene {
        Window(Bundle.appName, id: "main") {
            ContentView(model: model).frame(minWidth: 800, minHeight: 600)
        }
        .commands { ViewCommands(model: model) }
    }
}

struct ViewCommands: Commands {
    @Bindable var model: ChatModel
    var body: some Commands {
        // Replaces the default Settings item: this app has no Settings scene
        // (settings live in a window view), so Command-, would otherwise be
        // dead.
        CommandGroup(replacing: .appSettings) {
            Button {
                model.openSettings()
            } label: {
                Label("Settings\u{2026}", systemImage: "gearshape")
            }
            .keyboardShortcut(",", modifiers: .command)
        }
        // A group, not CommandMenu("View"): SwiftUI already builds a View
        // menu (Enter Full Screen lives there), and a same-named CommandMenu
        // adds a second one instead of merging into it.
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

func setClipboard(_ s: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(s, forType: .string)
}

@MainActor func hideSoftKeyboard() { }

var isOS: Bool { return false }

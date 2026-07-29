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

// View > Status Bar (None / Short / Extended), mirroring the Settings control.
// macOS-only: iOS has no menu bar.

struct StatusBarCommands: Commands {
    @Bindable var model: ChatModel
    var body: some Commands {
        CommandMenu("View") {
            Picker("Status Bar", selection: $model.statusBarMode) {
                ForEach(ChatModel.StatusBarMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
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

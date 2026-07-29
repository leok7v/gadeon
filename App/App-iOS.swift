import LLM
import SwiftUI
import UIKit

@main struct GadeonApp: App {
    @State private var model = ChatModel()
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            // ContentView draws its own top bar + overlay drawer; it must NOT be
            // wrapped in a NavigationStack or NavigationSplitView (nested
            // navigation containers render black on iPhone).
            ContentView(model: model)
                // iOS aborts GPU (Metal) work submitted while the app is not
                // active, so mirror the scene phase into the LLM background gate;
                // the Metal engine parks its command-buffer commits until the app
                // returns. The ANE engine keeps running in the background,
                // untouched. macOS does not gate -- background GPU is allowed.
                .onChange(of: scenePhase) { _, phase in
                    BackgroundGate.shared.setBackgrounded(phase != .active)
                    if phase == .background { model.commitCurrent() }
                }
        }
    }
}

@MainActor func quitApp() { exit(0) }

// Cross-platform clipboard write (macOS twin in App-macOS.swift), for the
// transcript tools' Copy without a #if os in the shared view.
func setClipboard(_ s: String) { UIPasteboard.general.string = s }

// Dismiss the soft keyboard (macOS twin is a no-op).
@MainActor func hideSoftKeyboard() {
    UIApplication.shared.sendAction(
        #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
}

var isOS: Bool { return true }

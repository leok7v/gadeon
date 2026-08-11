import LLM
import SwiftUI
import UIKit

@main struct GadeonApp: App {
    @State private var model = ChatModel()
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            // Must not be wrapped in a NavigationStack or NavigationSplitView:
            // nested navigation containers render black on iPhone.
            ContentView(model: model)
                // iOS aborts Metal work submitted while the app is not active;
                // mirror scene phase into the gate so it parks commits until
                // the app returns.
                .onChange(of: scenePhase) { _, phase in
                    BackgroundGate.shared.setBackgrounded(phase != .active)
                    if phase == .background { model.commitCurrent() }
                }
        }
    }
}

@MainActor func quitApp() { exit(0) }

func setClipboard(_ s: String) { UIPasteboard.general.string = s }

@MainActor func hideSoftKeyboard() {
    UIApplication.shared.sendAction(
        #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
}

var isOS: Bool { return true }

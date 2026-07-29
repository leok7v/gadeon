import SwiftUI

// The onboarding screens (EULA, disclaimer, download, optimizing, loading) have
// no app top bar -- the chat's ChatChrome exists only once the model is ready --
// so the app name is pinned at the top of the screen (where a title bar would
// be) and the screen's content fills below it: centered for the progress
// screens, a ScrollView + Accept for EULA/disclaimer (they scroll under the
// title). macOS shows the name in the window title bar (its twin is identity).

struct OnboardingTitle: ViewModifier {
    func body(content: Content) -> some View {
        VStack(spacing: 0) {
            Text(Bundle.appName)
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

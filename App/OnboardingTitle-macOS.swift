import SwiftUI

// macOS shows the app name in the window title bar; the iOS twin draws it as a
// top title on the onboarding screens.

struct OnboardingTitle: ViewModifier {
    func body(content: Content) -> some View { content }
}

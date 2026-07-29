import SwiftUI

// macOS top bar via the native window toolbar (merges into the title bar, so
// no second bar). The iOS twin draws its own HStack inset.

struct ChatChrome<Leading: View, Center: View, Trailing: View>: ViewModifier {
    let leading: Leading
    let center: Center
    let trailing: Trailing

    func body(content: Content) -> some View {
        content.toolbar {
            ToolbarItem(placement: .navigation) { leading }
            ToolbarItem(placement: .principal) { center }
            ToolbarItem(placement: .primaryAction) { trailing }
        }
    }
}

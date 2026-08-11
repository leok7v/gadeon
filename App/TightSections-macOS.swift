import SwiftUI

// listSectionSpacing is unavailable on macOS, where the default gap is
// already the right size.

struct TightSections: ViewModifier {
    func body(content: Content) -> some View { content }
}

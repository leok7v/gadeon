import SwiftUI

// macOS has no fullScreenCover; a sheet is its modal equivalent. The iOS twin
// (Cover-iOS.swift) uses fullScreenCover, so the shared call site stays
// `.cover(isPresented:)` with no #if os.
extension View {
    func cover<Content: View>(isPresented: Binding<Bool>,
                              @ViewBuilder content: @escaping () -> Content)
        -> some View {
        sheet(isPresented: isPresented, content: content)
    }
}

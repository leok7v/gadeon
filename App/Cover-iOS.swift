import SwiftUI

// Present a view full-screen with no app chrome behind it. macOS has no
// fullScreenCover, so its twin (Cover-macOS.swift) uses a sheet; this SDK split
// keeps the shared call site (`.cover(isPresented:)`) free of a #if os.
extension View {
    func cover<Content: View>(isPresented: Binding<Bool>,
                              @ViewBuilder content: @escaping () -> Content)
        -> some View {
        fullScreenCover(isPresented: isPresented, content: content)
    }
}

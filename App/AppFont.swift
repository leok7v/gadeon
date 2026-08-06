import SwiftUI

// The app's text size, as one environment value every label in the app reads.
//
// It carries ONLY the app's own setting (ChatModel.textScale). On iOS the
// platform font this resolves to already has the device's Dynamic Type baked
// into it, so folding a Dynamic Type ratio in here as well would count it
// twice; on macOS there is no such ratio to fold, which is the entire reason
// this exists rather than a `.dynamicTypeSize` override.

private struct AppTextScaleKey: EnvironmentKey {
    static let defaultValue: CGFloat = 1
}

extension EnvironmentValues {
    var appTextScale: CGFloat {
        get { self[AppTextScaleKey.self] }
        set { self[AppTextScaleKey.self] = newValue }
    }
}

// Used wherever a plain SwiftUI text style would otherwise go, so the app's
// text size has ONE definition instead of one per label. The style keeps its
// own weight and design, because the platform font is rebuilt from its
// descriptor rather than guessed at from a table of point sizes.
extension View {
    func appFont(_ style: Font.TextStyle) -> some View {
        modifier(AppFontModifier(style: style))
    }
}

// A modifier rather than a computed Font at each call site: the modifier is a
// child of whoever applies it, so it reads the environment the label actually
// renders in, and no view needs a reference to the model to draw text.
struct AppFontModifier: ViewModifier {

    @Environment(\.appTextScale) private var scale
    let style: Font.TextStyle

    func body(content: Content) -> some View {
        content.font(appTextFont(style, scale))
    }
}

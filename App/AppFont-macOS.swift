import AppKit
import SwiftUI

// A text style at the app's text size. AppKit has no content size category,
// so the base point size is fixed and the scale is the only thing that moves
// it -- which is why the app carries its own multiplier at all.

func appTextFont(_ style: Font.TextStyle, _ scale: CGFloat) -> Font {
    let base = NSFont.preferredFont(forTextStyle: appTextStyle(style))
    let sized = NSFont(descriptor: base.fontDescriptor,
                       size: base.pointSize * scale)
    return Font(sized ?? base)
}

// The style's own point size, before the app's scale. For the places that
// need a size BETWEEN two styles, or one run set slightly under another,
// neither of which a named style can express.

func appTextPoints(_ style: Font.TextStyle) -> CGFloat {
    NSFont.preferredFont(forTextStyle: appTextStyle(style)).pointSize
}

// The two vocabularies differ in two names only, and the fallback covers the
// styles a later SDK adds rather than failing to compile against it.

private func appTextStyle(_ style: Font.TextStyle) -> NSFont.TextStyle {
    let result: NSFont.TextStyle
    switch style {
    case .largeTitle:  result = .largeTitle
    case .title:       result = .title1
    case .title2:      result = .title2
    case .title3:      result = .title3
    case .headline:    result = .headline
    case .subheadline: result = .subheadline
    case .body:        result = .body
    case .callout:     result = .callout
    case .footnote:    result = .footnote
    case .caption:     result = .caption1
    case .caption2:    result = .caption2
    default:           result = .body
    }
    return result
}

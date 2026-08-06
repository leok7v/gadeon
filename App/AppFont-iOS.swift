import SwiftUI
import UIKit

// A text style at the app's text size. The platform font already carries this
// device's Dynamic Type, so the scale here is the app's own setting alone and
// multiplies on top of it -- a notch down from whatever the phone is set to,
// never an absolute size that discards it.

func appTextFont(_ style: Font.TextStyle, _ scale: CGFloat) -> Font {
    let base = UIFont.preferredFont(forTextStyle: appTextStyle(style))
    return Font(base.withSize(base.pointSize * scale))
}

// The style's own point size, before the app's scale but WITH this device's
// Dynamic Type, exactly as appTextFont resolves it. For the places that need
// a size between two styles, or one run set slightly under another.

func appTextPoints(_ style: Font.TextStyle) -> CGFloat {
    UIFont.preferredFont(forTextStyle: appTextStyle(style)).pointSize
}

// The two vocabularies differ in two names only, and the fallback covers the
// styles a later SDK adds rather than failing to compile against it.

private func appTextStyle(_ style: Font.TextStyle) -> UIFont.TextStyle {
    let result: UIFont.TextStyle
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

import SwiftUI

// Maps a semantic role to a platform font. `size` overrides let a style's
// point sizes drive the body / heading / mono fonts.

enum FontRole {
    case body(CGFloat)
    case heading(level: Int, size: CGFloat)
    case mono(CGFloat)

    var platformFont: PlatformFont {
        let result: PlatformFont
        switch self {
            case .body(let size):
                result = PlatformFont.systemFont(ofSize: size)
            case .heading(_, let size):
                result = boldFont(of: PlatformFont.systemFont(ofSize: size))
            case .mono(let size):
                result = monoFont(at: size)
        }
        return result
    }
}

// Applies inline emphasis to a run font. Code runs become monospaced;
// otherwise bold / italic traits are layered onto the base.
func styledRunFont(intent: InlinePresentationIntent,
                   base: PlatformFont,
                   size: CGFloat? = nil,
                   additionalBold: Bool = false) -> PlatformFont {
    let s = size ?? base.pointSize
    let result: PlatformFont
    if intent.contains(.code) {
        result = monoFont(at: s)
    } else {
        result = platformBoldItalicFont(
            of: base,
            bold: additionalBold || intent.contains(.stronglyEmphasized),
            italic: intent.contains(.emphasized))
    }
    return result
}

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

// A script run is set at roughly seven tenths of its base and shifted off
// the baseline. The two directions are not symmetric: a superscript has to
// clear the x-height of the text beside it, a subscript only has to drop
// clear of the baseline, so raising travels further than lowering. Derived
// from the base font's size rather than fixed, so it tracks the text scale,
// the heading level and the PDF's own sizes without any of them knowing.

func scriptRunFont(_ level: Int, base: PlatformFont)
    -> (font: PlatformFont, offset: CGFloat) {
    let size = base.pointSize
    let font = platformResizedFont(base, to: (size * 0.72).rounded())
    let offset = level > 0 ? size * 0.33 : -size * 0.14
    return (font, offset)
}

// Stamp the script runs of `attr` onto an NSAttributedString already built
// from it. The base font is read back out rather than recomputed, because by
// this point it carries the role, the scale and whatever bold or italic the
// run inherited, all of which the shrunken size must keep. It exists for the
// renderers that hand the whole string to NSAttributedString(_:) and lose
// the custom key on the way; one that walks attr.runs itself needs none of
// this.

func applyScriptRuns(_ m: NSMutableAttributedString,
                     from attr: AttributedString) {
    for run in attr.runs {
        let level = run[ScriptAttribute.self]
        let r = NSRange(run.range, in: attr)
        if let level, r.length > 0, NSMaxRange(r) <= m.length {
            stampScript(m, level: level, range: r)
        }
    }
}

private func stampScript(_ m: NSMutableAttributedString,
                         level: Int, range: NSRange) {
    let base = m.attribute(.font, at: range.location,
                           effectiveRange: nil) as? PlatformFont
    if let base {
        let script = scriptRunFont(level, base: base)
        m.addAttribute(.font, value: script.font, range: range)
        m.addAttribute(.baselineOffset, value: script.offset, range: range)
    }
}

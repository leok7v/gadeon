import Foundation
import XCTest
@testable import MD

// A display formula is one run of glyphs at fixed geometry. It has no line
// breaks to give, so a surface narrower than its natural width does not
// reflow it -- without a fit it lays out at full size and the right end is
// silently cut off, which is the half of an equation nobody can guess.
//
// The transcript is where that bites: a bubble takes the width it is offered
// and self-sizes vertically only, so it can neither widen nor scroll
// sideways.
//
// Note what is NOT measured here. `boundingRect(forGlyphRange:in:)` CLAMPS
// to the container, so an over-wide attachment reads as exactly full width
// through it and a fitted one reads the same -- the two are
// indistinguishable that way, which is how a first version of this file
// passed against code that did no fitting at all. So the rule is asked
// directly, and one case proves TextKit honours the answer.
@MainActor
final class MathFitTests: XCTestCase {

    private static let wide =
        "$$ \\sum_{i=1}^{n} \\frac{\\alpha_i + \\beta_i}{\\gamma_i} = "
        + "\\int_0^\\infty e^{-x^2} \\, dx + \\sqrt{\\lambda + \\mu} $$"

    private let natural = CGSize(width: 300, height: 60)

    func testANarrowSurfaceScalesTheFormulaToIt() {
        let fitted = DocumentText.mathFit(natural: natural, available: 180)
        XCTAssertEqual(fitted.width, 180, accuracy: 0.01)
        // Uniform, or the formula is distorted rather than scaled.
        XCTAssertEqual(fitted.height, 60 * 180 / 300, accuracy: 0.01)
    }

    // A ceiling, not a resize: given room, the formula keeps its own size
    // rather than stretching to fill the bubble.
    func testAFormulaThatFitsIsLeftAlone() {
        for available in [300.0, 600.0, 5000.0] as [CGFloat] {
            let fitted = DocumentText.mathFit(natural: natural,
                                              available: available)
            XCTAssertEqual(fitted.width, 300, accuracy: 0.01)
            XCTAssertEqual(fitted.height, 60, accuracy: 0.01)
        }
    }

    // Monotonic, and never zero or negative however absurd the offer: a
    // width of zero arrives while a view is being laid out for the first
    // time, and must leave the formula alone rather than collapse it.
    func testDegenerateWidthsLeaveItAlone() {
        for available in [0.0, -1.0] as [CGFloat] {
            let fitted = DocumentText.mathFit(natural: natural,
                                              available: available)
            XCTAssertEqual(fitted.width, 300, accuracy: 0.01)
        }
    }

    #if os(macOS)
    // The rule is only worth anything if TextKit asks and obeys. A roomy
    // container must draw the formula at its natural width, which is also
    // what says the fit is not silently scaling everything.
    func testTextKitDrawsTheFittedWidth() throws {
        let ns = DocumentText.attributed(
            from: Markdown.parse(Self.wide), style: .default)
        let storage = NSTextStorage(attributedString: ns)
        let manager = NSLayoutManager()
        let box = NSTextContainer(size: CGSize(width: 900, height: 1e6))
        box.lineFragmentPadding = 0
        storage.addLayoutManager(manager)
        manager.addTextContainer(box)
        manager.ensureLayout(for: box)
        var natural: CGFloat = 0
        var drawn: CGFloat = 0
        let full = NSRange(location: 0, length: storage.length)
        storage.enumerateAttribute(.attachment, in: full,
                                   options: []) { value, range, _ in
            if let cell = (value as? NSTextAttachment)?
                .attachmentCell as? NSTextAttachmentCell {
                natural = cell.cellSize().width
                let glyphs = manager.glyphRange(forCharacterRange: range,
                                                actualCharacterRange: nil)
                drawn = manager.boundingRect(forGlyphRange: glyphs,
                                             in: box).width
            }
        }
        XCTAssertGreaterThan(natural, 0, "no math cell was laid out")
        XCTAssertEqual(drawn, natural, accuracy: 0.5,
                       "TextKit did not draw the size the cell asked for")
    }
    #endif
}

import XCTest
@testable import MD

// The typesetting path, from the markdown a model emits to a laid-out
// formula.
//
// The one that matters most is fontIsBundled. KaTeX resolves STIXTwoMath
// from Bundle.module and falls back to the copy macOS keeps in
// /System/Library/Fonts/Supplemental -- so on a Mac the engine works
// whether or not the resource shipped, and the failure would surface only
// on a phone, where there is no system copy and every formula silently
// degrades to its Unicode spelling. Asking the bundle directly is what
// makes that visible here.
@MainActor
final class MathTypesetTests: XCTestCase {

    func testFontIsBundled() {
        let url = Bundle.module.url(forResource: "STIXTwoMath",
                                    withExtension: "otf")
        XCTAssertNotNil(url, "the math font is not in the module bundle, so "
                             + "iOS has no MATH-table font at all")
    }

    func testDisplayBecomesItsOwnBlock() {
        let doc = Markdown.parse("before\n\n$$\nx^2 + y^2 = z^2\n$$\n\nafter")
        let maths = doc.items.compactMap { item -> String? in
            if case .math(let tex) = item.block { return tex }
            return nil
        }
        XCTAssertEqual(maths, ["x^2 + y^2 = z^2"])
        XCTAssertEqual(doc.items.count, 3)
    }

    func testOneLineDisplayIsAlsoABlock() {
        let doc = Markdown.parse("$$ E = mc^2 $$")
        XCTAssertEqual(doc.items.count, 1)
        if case .math(let tex) = doc.items[0].block {
            XCTAssertEqual(tex, "E = mc^2")
        } else {
            XCTFail("expected a math block, got \(doc.items[0].block)")
        }
    }

    // Math off means the dollars are literal text, so nothing may open a
    // display and swallow the rest of the document.
    func testMathOffLeavesTheDollarsAlone() {
        let doc = Markdown.parse("$$ E = mc^2 $$", math: false)
        XCTAssertEqual(doc.items.count, 1)
        if case .math = doc.items[0].block {
            XCTFail("a display was opened with math disabled")
        }
    }

    func testEngineLaysOutAFormula() throws {
        let layout = try XCTUnwrap(TeX.layout("\\frac{a}{b}", size: 20),
                                   "the engine refused an ordinary fraction")
        XCTAssertGreaterThan(layout.width, 0)
        XCTAssertGreaterThan(layout.height, 0)
    }

    // A refusal must fall back rather than vanish: the Unicode spelling is
    // readable where an empty box is not.
    func testAnUnknownMacroStillReads() {
        XCTAssertNil(TeX.layout("\\notarealmacro{x}", size: 20))
        let spelled = String(TeX.render("\\alpha + \\beta",
                                        display: true).characters)
        XCTAssertEqual(spelled, "\u{03B1} + \u{03B2}")
    }

    // The control-word boundary. Without it a token map substitution fires
    // inside a longer command and the fallback text reads as gibberish.
    func testASubstitutionStopsAtTheWordBoundary() {
        let spelled = String(TeX.render("\\newcommand", display: false)
                                .characters)
        XCTAssertFalse(spelled.contains("\u{2260}"),
                       "\\ne matched inside \\newcommand: \(spelled)")
    }

    // Rows and columns without an environment around them. A converter
    // lifting equations out of a PDF drops the \begin{aligned} and leaves
    // exactly this, and a stack of rows is unambiguous enough to draw.
    func testRowsWithoutAnEnvironment() throws {
        let rows = try XCTUnwrap(TeX.layout("a \\\\ b", size: 20))
        let one = try XCTUnwrap(TeX.layout("a", size: 20))
        XCTAssertGreaterThan(rows.height, one.height)
        XCTAssertNotNil(TeX.layout("x &= 1 \\\\ y &= 2", size: 20))
    }

    // A paragraph that is nothing but TeX becomes a display, so it is
    // typeset rather than read out as a wall of backslashes.
    func testBareTeXParagraphBecomesADisplay() {
        let doc = Markdown.parse("\\frac{a}{b} + \\sqrt{c}")
        XCTAssertEqual(doc.items.count, 1)
        if case .math(let tex) = doc.items[0].block {
            XCTAssertEqual(tex, "\\frac{a}{b} + \\sqrt{c}")
        } else {
            XCTFail("expected a math block, got \(doc.items[0].block)")
        }
    }

    // The test has to be strict enough that prose can never pass it. In
    // maths neighbouring letters are separate variables multiplied together,
    // so a sentence mentioning a macro parses perfectly and would be typeset
    // if parsing alone decided it.
    func testProseMentioningAMacroStaysProse() {
        for source in ["\\alpha is the first letter",
                       "\\frac is how you write a fraction",
                       "\\\\server\\share\\path",
                       "just ordinary words"] {
            let doc = Markdown.parse(source)
            if case .math = doc.items[0].block {
                XCTFail("typeset prose: \(source)")
            }
        }
    }

    func testScriptTagsBecomeRunsAndUnicode() {
        let doc = Markdown.parse("m<sup>2</sup> and H<sub>2</sub>O")
        guard case .paragraph(let attr) = doc.items[0].block else {
            return XCTFail("expected a paragraph")
        }
        let levels = attr.runs.compactMap { run in run[ScriptAttribute.self] }
        XCTAssertEqual(levels, [1, -1])
        XCTAssertEqual(TeX.unicodeScript("2", superscript: true), "\u{00B2}")
        XCTAssertEqual(TeX.unicodeScript("2", superscript: false), "\u{2082}")
    }

    // Every character or none: a half-mapped run reads as a typo.
    func testAnUnmappableScriptFallsBackWhole() {
        XCTAssertEqual(TeX.unicodeScript("qq", superscript: true), "(qq)")
    }

    // Nesting unwinds inside out, and a raw cell is what the table measurers
    // see, so this is the form a column width is computed against.
    func testNestedScriptTagsUnwind() {
        XCTAssertEqual(TeX.scriptsToUnicode("m<sub>DO<sub>2</sub></sub>"),
                       "m(DO\u{2082})")
    }

    func testPlainExportSpendsUnicodeForScripts() {
        let doc = Markdown.parse("m<sup>2</sup>")
        let out = Markdown.plainText(doc)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertEqual(out, "m\u{00B2}")
    }

    func testHtmlExportGivesTheTagBack() {
        let doc = Markdown.parse("m<sup>2</sup>")
        let out = Markdown.html(doc, title: "t")
        XCTAssertTrue(out.contains("<sup>2</sup>"), out)
    }

    // A display survives the exports too, in the spelling each medium wants.
    func testExportsCarryTheDisplay() {
        let doc = Markdown.parse("$$ E = mc^2 $$")
        XCTAssertTrue(Markdown.plainText(doc).contains("$$"),
                      Markdown.plainText(doc))
        XCTAssertTrue(Markdown.html(doc, title: "t").contains("mc"),
                      Markdown.html(doc, title: "t"))
    }
}

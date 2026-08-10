import Foundation
import XCTest
@testable import MD

// One MathFontFile serves every formula in the process, and it fills its
// glyph, CTFont and metric caches lazily -- so laying a formula out mutates
// shared state, and so does drawing one, which asks the font for a CTFont at
// whatever size it is being drawn at.
//
// This is reachable on the ordinary path, not a contrived one: MarkdownPDF
// .export is a nonisolated async function and the transcript starts it with
// `async let` from a SwiftUI .task, so an export lays out and rasterizes
// formulas off the main thread while the screen draws its own.
//
// Run it under ThreadSanitizer to see the difference:
//
//     swift test --sanitize=thread --filter MathConcurrencyTests
//
// With the lock covering layout alone, TSan reports races at
// MathFontFile.ctFont. With draw holding it too, it is quiet.
final class MathConcurrencyTests: XCTestCase {

    private static let formulas = [
        "\\frac{a}{b}", "x^2 + y^2 = z^2", "\\sum_{i=1}^{n} i",
        "\\sqrt{\\alpha + \\beta}", "\\int_0^1 f(x)\\,dx",
        "\\frac{\\partial u}{\\partial t}", "a \\\\ b", "x &= 1 \\\\ y &= 2",
    ]

    // Lay out AND rasterize on every thread, because the two touch different
    // caches and only doing both reaches the one that raced.
    private func work(_ tex: String, _ size: CGFloat) -> Bool {
        var ok = false
        if let layout = TeX.layout(tex, size: size) {
            ok = layout.width > 0
                && layout.cgImage(scale: 1, padding: 2, background: nil,
                                  color: CGColor(gray: 0, alpha: 1)) != nil
        }
        return ok
    }

    func testConcurrentLayoutAndDraw() {
        let formulas = Self.formulas
        let done = expectation(description: "every thread finished")
        done.expectedFulfillmentCount = formulas.count
        let drawn = NSCountedSet()
        let tally = NSLock()
        // Sizes differ per thread on purpose: one CTFont cache entry per
        // size, so identical sizes would let the first thread warm the cache
        // and hide the write the others race on.
        for (i, tex) in formulas.enumerated() {
            DispatchQueue.global().async {
                var made = 0
                for round in 0..<8 {
                    let size = CGFloat(14 + i) + CGFloat(round) * 0.5
                    if self.work(tex, size) { made += 1 }
                }
                tally.lock()
                drawn.add(made)
                tally.unlock()
                done.fulfill()
            }
        }
        wait(for: [done], timeout: 60)
        XCTAssertEqual(drawn.count(for: 8), formulas.count,
                       "a thread failed to lay out or rasterize")
    }

    // The same font instance is handed out every time, which is the whole
    // reason the lock has to exist rather than each caller parsing its own.
    func testEveryThreadSharesOneFont() throws {
        let first = try MathFontFile.shared()
        let second = try MathFontFile.shared()
        XCTAssertTrue(first === second)
    }
}

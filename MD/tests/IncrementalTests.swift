import XCTest
@testable import MD

// applyIncremental splices only the changed span of an attributed string into
// the text storage (shared prefix/suffix preserved), so a streaming re-render
// re-lays out O(delta). Correctness is the whole point: after the splice the
// storage MUST equal the target byte-for-byte AND attribute-for-attribute, or
// the transcript (and the reading/Find surface, which shares this path) shows
// stale text/formatting.
final class IncrementalTests: XCTestCase {

    private let red: [NSAttributedString.Key: Any] =
        [.foregroundColor: PlatformColor.red]

    private func s(_ str: String,
                   _ attrs: [NSAttributedString.Key: Any] = [:])
        -> NSAttributedString {
        NSAttributedString(string: str, attributes: attrs)
    }

    private func cat(_ parts: NSAttributedString...) -> NSAttributedString {
        let m = NSMutableAttributedString()
        for p in parts { m.append(p) }
        return m
    }

    private func assertSplice(_ base: NSAttributedString,
                              _ target: NSAttributedString,
                              _ msg: String) {
        let storage = NSMutableAttributedString(attributedString: base)
        applyIncremental(storage, target)
        XCTAssertTrue(storage.isEqual(to: target),
            "\(msg): got \(storage.string.debugDescription) "
            + "want \(target.string.debugDescription)")
    }

    func testAppend() {
        assertSplice(s("Hello"), s("Hello World"), "append")
        assertSplice(s(""), s("fresh"), "empty->full")
    }

    func testPrefixChange() {
        assertSplice(s("Hello"), s("Jello"), "first char changed")
    }

    // A markdown close (`*text*` -> italic) changes attributes on characters
    // already emitted -- same string, different attributes. A string-only diff
    // would wrongly keep it in the shared prefix and leave the run unformatted.
    func testAttributeOnlyChange() {
        assertSplice(s("Hello"), cat(s("He", red), s("llo")), "attr added")
        assertSplice(cat(s("He", red), s("llo")), s("Hello"), "attr removed")
        assertSplice(cat(s("see "), s("text")),
                     cat(s("see "), s("text", red)), "retroactive attr")
    }

    func testSuffixPreserved() {
        assertSplice(s("abcXYZ"), s("abQYZ"), "middle replaced")
        assertSplice(cat(s("head ", red), s("mid"), s(" tail", red)),
                     cat(s("head ", red), s("MID"), s(" tail", red)),
                     "middle run replaced, attributed ends kept")
    }

    func testShrink() {
        assertSplice(s("Hello World"), s("Hello"), "shrink")
        assertSplice(s("stuff"), s(""), "full->empty")
    }

    func testIdentical() {
        assertSplice(s("same"), s("same"), "no-op")
        assertSplice(cat(s("a", red), s("b")),
                     cat(s("a", red), s("b")), "no-op attributed")
    }

    // Mimics DocumentText.code streaming: a background-tinted body plus an
    // appended NO-background "\n\n", grown line by line. After the last line
    // the storage must equal the fresh render AND the last body character must
    // still carry the background (the "print(df) rendered outside the block"
    // report). If this passes, the splice is correct and the artifact is
    // NSTextView rendering, not applyIncremental.
    func testStreamingBackgroundRunGrowth() {
        func codeRender(_ body: String) -> NSAttributedString {
            let m = NSMutableAttributedString(string: body,
                attributes: [.backgroundColor: PlatformColor.gray])
            m.append(s("\n\n"))
            return m
        }
        let lines = ["import x", "a = 1", "b = 2", "print(a)"]
        var body = ""
        let storage = NSMutableAttributedString()
        for (i, line) in lines.enumerated() {
            body += (i == 0 ? "" : "\n") + line
            applyIncremental(storage, codeRender(body))
        }
        let final = codeRender(body)
        XCTAssertTrue(storage.isEqual(to: final),
            "storage diverged from the fresh code render")
        let last = (body as NSString).length - 1
        let bg = storage.attribute(.backgroundColor, at: last,
                                   effectiveRange: nil)
        XCTAssertNotNil(bg, "the last code line lost its background")
    }

    // Token-by-token growth (the streaming case): each step must land equal to
    // the freshly-resolved whole, splicing only the new tail.
    func testStreamingGrowth() {
        let target = "First block.\n\nSecond block streaming in slowly."
        var acc = ""
        let prev = NSMutableAttributedString(string: "")
        for ch in target {
            acc.append(ch)
            let next = s(acc, red)
            applyIncremental(prev, next)
            XCTAssertTrue(prev.isEqual(to: next),
                "growth diverged at \(acc.count) chars")
        }
    }
}

import Foundation
import Testing
@testable import LLM

// What a voice should say about a Markdown answer, and -- as much -- what it
// should NOT say. The interesting cases are the ones where reading the source
// verbatim is worse than describing it, and the ones where a full stop does
// not end a sentence.

// Everything a chunker emits for one whole input, pushed in one go.

private func spoken(_ text: String) -> [String] {
    var s = SpeakableText()
    var out = s.push(text)
    out.append(contentsOf: s.finish())
    return out
}

// The same input delivered one character at a time, which is closer to how a
// token stream actually arrives.

private func spokenByCharacter(_ text: String) -> [String] {
    var s = SpeakableText()
    var out: [String] = []
    for ch in text { out.append(contentsOf: s.push(String(ch))) }
    out.append(contentsOf: s.finish())
    return out
}

@Suite struct SpeakableTextTests {

    @Test func sentencesComeOutOneAtATime() {
        let out = spoken("One thing. Two things! Three?\n")
        #expect(out == ["One thing.", "Two things!", "Three?"])
    }

    // The whole point of streaming: sentence one is speakable before the rest
    // of the paragraph has been generated.
    @Test func aCompleteSentenceIsEmittedBeforeItsLineEnds() {
        var s = SpeakableText()
        let first = s.push("Ready to go. And then")
        #expect(first == ["Ready to go."])
    }

    // Chunk boundaries are an artifact of the token stream and must not be
    // able to change what is said. The text carries markup and a numeral on
    // purpose: with plain words this passes even when the mid-line path skips
    // inline shaping entirely, which is exactly how that went unnoticed.
    @Test func chunkingDoesNotChangeTheResult() {
        let text = "A **first** line. A second one!\n\nAnd 1999?\n"
        #expect(spoken(text) == spokenByCharacter(text))
    }

    // A sentence that ends before its line does is the COMMON case while
    // streaming, and it has to be shaped like any other.
    @Test func aMidLineSentenceIsStillShaped() {
        var s = SpeakableText()
        let out = s.push("It cost 2,925.26 in **1999**. And then")
        #expect(out == ["It cost two thousand nine hundred twenty-five "
                        + "point two six in nineteen ninety-nine."])
    }

    // Marker stripping belongs to the START of a line only; a hash or a
    // "1." arriving mid-sentence is text.
    @Test func markersAreNotStrippedMidLine() {
        var s = SpeakableText()
        _ = s.push("First one. ")
        let out = s.push("Tagged #top and - dashed. ")
        #expect(out == ["Tagged #top and - dashed."])
    }

    // "1,000" reached the phonemizer whole only when its sentence ended at a
    // line break; mid-line it arrived as "1" and "000", which that engine
    // expands separately as "one" and "zero".
    @Test func aGroupedThousandIsNotReadAsOneZero() {
        var s = SpeakableText()
        let out = s.push("Starting with 1,000 units. ")
        #expect(out == ["Starting with one thousand units."])
    }

    @Test func codeBlocksAreDescribedNotRead() {
        let out = spoken("""
        Here it is:

        ```python
        for i in range(10):
            print(i)
        ```

        That was the loop.
        """)
        #expect(out == ["Here it is:", "A python code block.",
                        "That was the loop."])
    }

    @Test func anUnlabelledCodeBlockStillGetsDescribed() {
        let out = spoken("```\nx = 1\n```\n")
        #expect(out == ["A code block."])
    }

    // An unterminated fence at end of turn (a stopped generation) must still
    // close, or the rest of the answer is swallowed.
    @Test func anUnclosedFenceIsStillDescribed() {
        let out = spoken("Look:\n\n```swift\nlet x = 1\n")
        #expect(out == ["Look:", "A swift code block."])
    }

    // A table read cell by cell loses the geometry that made it a table, so
    // the count is the useful part. The header and its rule are not rows.
    @Test func tablesAreCountedNotRead() {
        let out = spoken("""
        Results:

        | Model | Size |
        |---|---|
        | A | 1 |
        | B | 2 |
        | C | 3 |

        Done.
        """)
        #expect(out == ["Results:", "A table of 3 rows.", "Done."])
    }

    @Test func listsKeepTheirItemsAndLoseTheirBullets() {
        let out = spoken("- first item\n- second item\n3. third item\n")
        #expect(out == ["first item", "second item", "third item"])
    }

    @Test func headingsAreTheirOwnBreath() {
        let out = spoken("## Overview\nThe body follows.\n")
        #expect(out == ["Overview", "The body follows."])
    }

    @Test func emphasisAndBackticksAreNotSpoken() {
        let out = spoken("It is **bold**, _italic_ and `code`.\n")
        #expect(out == ["It is bold, italic and code."])
    }

    @Test func aLinkKeepsItsTextAndDropsItsTarget() {
        let out = spoken("See [the docs](https://example.com/a) for more.\n")
        #expect(out == ["See the docs for more."])
    }

    // The three ways a full stop lies about ending a sentence. The decimal
    // arrives as words, because numbers are expanded on the way through.
    @Test func abbreviationsDoNotEndSentences() {
        #expect(spoken("Dr. Smith arrived.\n") == ["Dr. Smith arrived."])
        #expect(spoken("Pi is 3.14 exactly.\n")
                == ["Pi is three point one four exactly."])
        #expect(spoken("J. R. R. Tolkien wrote it.\n")
                == ["J. R. R. Tolkien wrote it."])
    }

    @Test func aHorizontalRuleIsSilent() {
        let out = spoken("Before.\n\n---\n\nAfter.\n")
        #expect(out == ["Before.", "After."])
    }

    // A turn that ends without punctuation (a stopped generation) must not
    // lose its tail.
    @Test func anUnterminatedTailIsStillSpoken() {
        let out = spoken("This one just trails off")
        #expect(out == ["This one just trails off"])
    }

    @Test func emptyInputSaysNothing() {
        #expect(spoken("").isEmpty)
        #expect(spoken("\n\n   \n").isEmpty)
    }
}

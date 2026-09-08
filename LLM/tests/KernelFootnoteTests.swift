import Foundation
import Testing

// The footnote convention in Kernels.metal is only worth having if it cannot
// rot, and it rots more easily than an adjacent comment does: editing a
// kernel does not put the footnote block in the same diff hunk, so a
// reference whose footnote was deleted -- or a footnote whose last reference
// went away -- passes review unnoticed. Both failures are mechanical to
// detect, and nothing else in the build would catch either.
//
// A dangling [tag] is the worse of the two: it sends a reader looking for an
// explanation that no longer exists, which is strictly worse than having
// written no comment at all.
struct KernelFootnoteTests {
    // Tags must contain a hyphen. That is what separates a reference like
    // [gemm-tiles] from the array subscripts MSL is full of (x[i], acc[r]),
    // and it is why the convention names tags rather than numbering them:
    // inserting a footnote never renumbers, so a reference cannot go stale
    // by drifting.
    private static let tag = try! NSRegularExpression(
        pattern: "\\[([a-z][a-z0-9]*(?:-[a-z0-9]+)+)\\]")

    private func kernelSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()      // LLM/tests
            .deletingLastPathComponent()      // LLM
            .appendingPathComponent("metal/Kernels.metal")
        return try String(contentsOf: url, encoding: .utf8)
    }

    @Test func everyFootnoteTagResolvesAndIsUsed() throws {
        let text = try kernelSource()
        var defined: Set<String> = []
        var referenced: Set<String> = []
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let s = String(line)
            let range = NSRange(s.startIndex..., in: s)
            let hits = KernelFootnoteTests.tag.matches(in: s, range: range)
            for h in hits {
                guard let r = Range(h.range(at: 1), in: s) else { continue }
                let name = String(s[r])
                // A DEFINITION is the footnote's own markdown heading,
                // `### [tag]  title`. Every other occurrence is a reference:
                // from code it is preceded by comment text, and inside the
                // block a cross-reference is mid-sentence.
                if s.hasPrefix("### [\(name)]") {
                    defined.insert(name)
                } else {
                    referenced.insert(name)
                }
            }
        }
        #expect(!defined.isEmpty, "no footnotes found; has the block moved?")
        let dangling = referenced.subtracting(defined).sorted()
        #expect(dangling.isEmpty,
                "referenced but never defined: \(dangling)")
        let orphaned = defined.subtracting(referenced).sorted()
        #expect(orphaned.isEmpty,
                "defined but never referenced: \(orphaned)")
    }
}

import XCTest
@testable import LLM

// Pure-math gates (no model) for the multi-turn / multi-image vision plumbing:
// visionPositionsMulti must seed its M-RoPE scalar at startScalar so a
// mid-conversation image continues from the running position (the extend
// carry), and expandPads must replicate each image placeholder into the tower's
// merged-token run for every image in a turn.
final class EngineVisionTests: XCTestCase {

    // Text-only positions advance a shared 3D scalar from startScalar; next is
    // the scalar just past the last token (the decode start). startScalar 0
    // reproduces the fresh-prefill numbering exactly.
    func testTextPositionsStartAtScalar() {
        let fresh = Engine.visionPositionsMulti(3, [])
        XCTAssertEqual(fresh.next, 3)
        XCTAssertEqual(fresh.pos.map { $0.0 }, [0, 1, 2])

        let plan = Engine.visionPositionsMulti(3, [], startScalar: 5)
        XCTAssertEqual(plan.next, 8)
        XCTAssertEqual(plan.pos.map { $0.0 }, [5, 6, 7])
        XCTAssertEqual(plan.pos.map { $0.1 }, [5, 6, 7])
        XCTAssertEqual(plan.pos.map { $0.2 }, [5, 6, 7])
    }

    // One image span (4x4 grid -> 2x2 = 4 merged tokens) based at startScalar:
    // token (r,c) -> (s, s+r, s+c); after the image the scalar advances by
    // max(gh,gw)/merge = 2, so next = startScalar + 2.
    func testImagePositionsBasedAtScalar() {
        let plan = Engine.visionPositionsMulti(
            4, [(start: 0, gh: 4, gw: 4)], startScalar: 10)
        XCTAssertEqual(plan.pos[0].0, 10)
        XCTAssertEqual([plan.pos[0].1, plan.pos[0].2], [10, 10])
        XCTAssertEqual([plan.pos[1].1, plan.pos[1].2], [10, 11])
        XCTAssertEqual([plan.pos[2].1, plan.pos[2].2], [11, 10])
        XCTAssertEqual([plan.pos[3].1, plan.pos[3].2], [11, 11])
        XCTAssertEqual(plan.next, 12)
    }

    // A text token then an image: the image bases at the post-text scalar, and
    // text resumes past the image's grid extent.
    func testTextThenImageChainsScalar() {
        let plan = Engine.visionPositionsMulti(
            5, [(start: 1, gh: 4, gw: 4)], startScalar: 10)
        XCTAssertEqual(plan.pos[0].0, 10)          // text at the seed scalar
        XCTAssertEqual(plan.pos[1].0, 11)          // image bases at 11
        XCTAssertEqual(plan.next, 13)              // 11 + 4/2
    }

    // expandPads replicates EACH image placeholder into `count` copies and
    // reports every run's start, so a multi-image turn splices each image's
    // feats at the right rows.
    func testExpandPadsMultiImage() {
        let out = Continuation.expandPads([7, 5, 7, 9], pad: 7, count: 3)
        XCTAssertEqual(out.ids, [7, 7, 7, 5, 7, 7, 7, 9])
        XCTAssertEqual(out.starts, [0, 4])
    }
}

import CoreML
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import LLM

// The tile/fit auto-demotion: tiling must preserve at least tileGain more
// resolution than the single fit tile, else imageSet returns just the
// thumbnail even when tiling was requested -- a near-tile-sized image would
// otherwise slice into mostly-padding slivers for a near-equal read.
final class VisionTilingTests: XCTestCase {
    // The 768px tower grid (48x48 patches) the 2B+ sets bake.
    private let grid = VisionGrid(perSide: 48)

    private func png(_ w: Int, _ h: Int) -> Data {
        let space = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.setFillColor(red: 0.5, green: 0.2, blue: 0.8, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        let out = NSMutableData()
        let dst = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dst, ctx.makeImage()!, nil)
        CGImageDestinationFinalize(dst)
        return out as Data
    }

    // Barely wider than one tile: tiling would buy ~1.1x resolution for 3
    // tiles (one an ~85% padding sliver) -- demoted to fit.
    func testNearTileImageDemotesToFit() throws {
        let tiles = try VisionPreprocess.imageSet(png(852, 82), tiled: true,
                                                  grid: grid)
        XCTAssertEqual(tiles.count, 1, "near-tile image should fit, not tile")
    }

    // A full screenshot: fit would crush it ~2.6x, so tiling stands.
    func testLargeImageStillTiles() throws {
        let tiles = try VisionPreprocess.imageSet(png(2000, 1200), tiled: true,
                                                  grid: grid)
        XCTAssertGreaterThan(tiles.count, 1, "large image should tile")
    }

    // Smaller than a tile on both edges: fit loses nothing; never tile.
    func testSmallImageNeverTiles() throws {
        let tiles = try VisionPreprocess.imageSet(png(500, 300), tiled: true,
                                                  grid: grid)
        XCTAssertEqual(tiles.count, 1)
    }

    // A long thin banner is nowhere near square yet must still tile: fit
    // would crush its text ~3.9x.
    func testThinBannerStillTiles() throws {
        let tiles = try VisionPreprocess.imageSet(png(3000, 100), tiled: true,
                                                  grid: grid)
        XCTAssertGreaterThan(tiles.count, 1, "thin banner should tile")
    }

    // The committed real-photo fixture (a 512x382 progressive JPEG): the
    // real decode path, and smaller than a tile on both edges, so it always
    // fits one tile.
    func testFixtureJpegFitsOneTile() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures/ViT-512x382.jpeg")
        let tiles = try VisionPreprocess.imageSet(
            try Data(contentsOf: url), tiled: true, grid: grid)
        XCTAssertEqual(tiles.count, 1)
    }
}

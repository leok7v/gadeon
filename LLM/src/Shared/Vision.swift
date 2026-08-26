import CoreGraphics
import CoreML
import Foundation
import ImageIO

// Host-side vision glue for the fixed v1 grid: an image becomes the patch
// tensor the vision tower consumes, and a user turn becomes the token sequence
// carrying the image-placeholder span. Both mirror HF (the Qwen2VL image
// processor + the chat template) and are gated byte/id-exact against it
// (gadeon-cli --vl-preprocess / --vl-image). Image decode needs CoreGraphics +
// ImageIO, Apple system frameworks with no third-party dependency.

public enum VisionError: Error { case decode }

// A per-turn tile bundle crossing Task / actor boundaries. MLMultiArray is not
// Sendable, so wrap it @unchecked -- the tiles are freshly built for one turn
// and handed off, never mutated concurrently -- to carry them from the app
// through ChatSession to the engine without a per-hop `sending` dance.
public struct VisionTiles: @unchecked Sendable {
    public let tiles: [MLMultiArray]
    public init(_ tiles: [MLMultiArray]) { self.tiles = tiles }
}

// The tower's baked square grid: `perSide` patches per edge. Read from the
// loaded tower (Engine.visionGrid), so a 256 vs 512 tower needs no Swift
// change -- side / merged-token count derive here. `canonical` is the v1 256px
// default for the preprocessor gate that runs without a loaded tower.

public struct VisionGrid: Sendable {
    public let perSide: Int
    public var gridH: Int { perSide }
    public var gridW: Int { perSide }
    public var side: Int { perSide * VisionPreprocess.patch }
    public var mergedTokens: Int {
        let m = perSide / VisionPreprocess.merge
        return m * m
    }
    public init(perSide: Int) { self.perSide = perSide }
    public static let canonical = VisionGrid(perSide: 16)
}

public enum VisionPreprocess {
    // Patch geometry (config.json vision_config), constant across all qwen3.5
    // VL sets: 16px patches, 2x temporal, 2x2 spatial merge. Only the grid
    // (patches per side) varies with the tower resolution, and it rides in
    // VisionGrid rather than a constant here.
    static let patch = 16
    static let temporal = 2
    static let merge = 2
    // Aspect-preserving tiling (multi-image): resize within a budget (<=
    // maxTiles detail tiles and <= maxLongPx on the long edge), slice into
    // side x side tiles overlapping by `overlap` px (so content at a seam
    // appears whole in a neighbour), tail padded neutral grey. All tiles share
    // the one tower's grid. A bigger maxTiles reads finer text (dense pages) at
    // the cost of mergedTokens more vision tokens per tile.
    static let overlap = 32
    static let maxTiles = 20
    static let maxLongPx = 2048
    static let padByte: UInt8 = 0x88
    // Cap the DECODED bitmap's long edge (~4K), so a pathological source (a
    // 100MP photo) does not decode into a huge transient RGBA buffer before it
    // is scaled to a tile. Downscale-only (no upscaling of smaller images). The
    // byte-exact HF gate path (normalizedRGB/patches) keeps the full-res decode.
    static let maxDecodePx = 4096

    // Decode `data` to a CGImage no larger than maxDecodePx on the long edge.
    // CGImageSourceCreateThumbnailFromImageAlways scales down at decode time, so
    // the full-resolution bitmap is never materialized. No EXIF transform: the
    // tower path matches HF's raw pixel order.
    static func decodeCapped(_ data: Data) -> CGImage? {
        decode(data, maxPx: maxDecodePx, transform: false)
    }

    // A small display thumbnail (a chip preview): EXIF-transformed so a phone
    // photo shows upright. Cross-platform CGImage -- SwiftUI's
    // Image(decorative:scale:) renders it with no NSImage/UIImage SDK split.
    public static func thumbnail(_ data: Data, maxPx: Int) -> CGImage? {
        decode(data, maxPx: maxPx, transform: true)
    }

    private static func decode(_ data: Data, maxPx: Int,
                               transform: Bool) -> CGImage? {
        var result: CGImage? = nil
        if let src = CGImageSourceCreateWithData(data as CFData, nil) {
            let opts: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPx,
                kCGImageSourceCreateThumbnailWithTransform: transform,
            ]
            result = CGImageSourceCreateThumbnailAtIndex(
                src, 0, opts as CFDictionary)
        }
        return result
    }

    // A decoded image scaled to rw x rh, drawn top-left into a cw x ch canvas
    // filled with neutral grey, normalized (rescale /255 then mean/std 0.5) into
    // row-major HWC floats.
    static func canvas(_ img: CGImage, _ rw: Int, _ rh: Int,
                       _ cw: Int, _ ch: Int) throws -> [Float] {
        var out: [Float]? = nil
        let space = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: cw, height: ch, bitsPerComponent: 8,
            bytesPerRow: cw * 4, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        if let ctx, let base = ctx.data {
            let g = Double(padByte) / 255.0
            ctx.setFillColor(red: g, green: g, blue: g, alpha: 1)
            ctx.fill(CGRect(x: 0, y: 0, width: cw, height: ch))
            // Buffer row 0 is the image top (unflipped draw), so place the image
            // at the top: y = ch - rh in CoreGraphics' y-up space.
            ctx.draw(img, in: CGRect(x: 0, y: ch - rh, width: rw, height: rh))
            let p = base.bindMemory(to: UInt8.self, capacity: cw * ch * 4)
            var arr = [Float](repeating: 0, count: cw * ch * 3)
            for i in 0 ..< cw * ch {
                for c in 0 ..< 3 {
                    arr[i * 3 + c] = Float(p[i * 4 + c]) / 127.5 - 1.0
                }
            }
            out = arr
        }
        if out == nil { throw VisionError.decode }
        return out!
    }

    // One side x side tile's patches, read from a canvas at pixel offset
    // (x0, y0) in Qwen2VL merge-block order (see patches()).
    static func patchTensor(_ px: [Float], _ cw: Int, _ x0: Int,
                            _ y0: Int, _ grid: VisionGrid) throws
        -> MLMultiArray {
        let gwB = grid.gridW / merge
        let nRow = grid.gridH * grid.gridW
        let dim = 3 * temporal * patch * patch
        let a = try MLMultiArray(shape: [NSNumber(value: nRow),
                                         NSNumber(value: dim)], dataType: .float16)
        a.withUnsafeMutableBytes { d, _ in
            let dp = d.bindMemory(to: Float16.self)
            for r in 0 ..< nRow {
                let mw = r % merge
                let mh = (r / merge) % merge
                let gwb = (r / (merge * merge)) % gwB
                let ghb = r / (merge * merge * gwB)
                for k in 0 ..< dim {
                    let pw = k % patch
                    let ph = (k / patch) % patch
                    let c = k / (patch * patch * temporal)
                    let y = y0 + (ghb * merge + mh) * patch + ph
                    let x = x0 + (gwb * merge + mw) * patch + pw
                    dp[r * dim + k] = Float16(px[(y * cw + x) * 3 + c])
                }
            }
        }
        return a
    }

    // The image set for one photo. tiled -> a whole-image thumbnail (global
    // gist) FIRST then aspect-preserving detail tiles (AnyRes, best detail);
    // else just the thumbnail (fit: one tile, cheapest). Each is one side x
    // side tile through the same tower; all grey-padded, no distortion.
    // Tiling self-demotes to fit when it would preserve under tileGain more
    // resolution than the thumbnail alone (a near-tile-sized image).
    public static func imageSet(_ data: Data, tiled: Bool, grid: VisionGrid)
        throws -> [MLMultiArray] {
        let img = decodeCapped(data)
        var out: [MLMultiArray] = []
        if let img {
            out.append(try thumbnail(img, grid))
            if tiled && tilingGain(img, grid) >= tileGain {
                out.append(contentsOf: try detailTiles(img, grid))
            }
        } else {
            throw VisionError.decode
        }
        return out
    }

    // Tiling must EARN its cost: every detail tile adds a tower encode and
    // mergedTokens of context, so it pays only when it preserves at least
    // this much more resolution than the single fit tile. Below the gain a
    // near-tile-sized image slices into mostly-padding slivers for a
    // near-equal read.
    static let tileGain = 1.5

    // Largest content scale the detail-tile budget allows: no upscaling,
    // <= maxLongPx on the long edge, ~maxTiles tiles
    // (tiles ~ w*h*scale^2/stride^2).
    static func tileScale(_ img: CGImage, _ grid: VisionGrid) -> Double {
        let stride = grid.side - overlap
        let byLong = Double(maxLongPx) / Double(max(img.width, img.height))
        let byCount = Double(stride)
            * (Double(maxTiles) / Double(img.width * img.height)).squareRoot()
        return min(1.0, byLong, byCount)
    }

    // Resolution the detail tiles would keep over the single fit tile.
    static func tilingGain(_ img: CGImage, _ grid: VisionGrid) -> Double {
        let long = Double(max(img.width, img.height))
        let fitScale = min(1.0, Double(grid.side) / long)
        return tileScale(img, grid) / fitScale
    }

    // Whole image letterboxed (aspect-preserved) into one side x side tile.
    static func thumbnail(_ img: CGImage, _ grid: VisionGrid) throws
        -> MLMultiArray {
        let side = grid.side
        let scale = Double(side) / Double(max(img.width, img.height))
        let rw = max(1, Int((Double(img.width) * scale).rounded()))
        let rh = max(1, Int((Double(img.height) * scale).rounded()))
        return try patchTensor(canvas(img, rw, rh, side, side),
                               side, 0, 0, grid)
    }

    // Aspect-preserving detail tiles: resize to the tile budget, slice
    // overlapping side x side tiles, tail padded grey.
    static func detailTiles(_ img: CGImage, _ grid: VisionGrid) throws
        -> [MLMultiArray] {
        let side = grid.side
        let w = img.width, h = img.height
        let stride = side - overlap
        let scale = tileScale(img, grid)
        let rw = max(1, Int((Double(w) * scale).rounded()))
        let rh = max(1, Int((Double(h) * scale).rounded()))
        let cols = rw <= side ? 1 : (rw - overlap + stride - 1) / stride
        let rows = rh <= side ? 1 : (rh - overlap + stride - 1) / stride
        let cw = (cols - 1) * stride + side
        let ch = (rows - 1) * stride + side
        let px = try canvas(img, rw, rh, cw, ch)
        var out: [MLMultiArray] = []
        for r in 0 ..< rows {
            for c in 0 ..< cols {
                out.append(try patchTensor(px, cw, c * stride, r * stride,
                                           grid))
            }
        }
        return out
    }

    // Decode + force-resize to side x side, drop alpha, and normalize (rescale
    // /255 then mean/std 0.5) into row-major HWC floats. Force-resize (aspect
    // not preserved); used by the single-image gate fixtures. The multi-image
    // path (imageSet) letterboxes instead.
    static func normalizedRGB(_ data: Data, _ grid: VisionGrid) throws
        -> [Float] {
        let side = grid.side
        let n = side * side
        var out = [Float](repeating: 0, count: n * 3)
        let src = CGImageSourceCreateWithData(data as CFData, nil)
        let img = src.flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }
        let space = CGColorSpaceCreateDeviceRGB()
        let ctx = img == nil ? nil : CGContext(
            data: nil, width: side, height: side, bitsPerComponent: 8,
            bytesPerRow: side * 4, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        if let img, let ctx, let base = ctx.data {
            // A plain draw already lands buffer row 0 at the image top (matching
            // HF/PIL row-major); an explicit y-flip here double-flips it.
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: side, height: side))
            let p = base.bindMemory(to: UInt8.self, capacity: n * 4)
            for i in 0 ..< n {
                for c in 0 ..< 3 {
                    out[i * 3 + c] = Float(p[i * 4 + c]) / 127.5 - 1.0
                }
            }
        } else {
            throw VisionError.decode
        }
        return out
    }

    // image -> patches[N, 3*T*patch*patch] fp16 in Qwen2VL merge-block order:
    // row = GHb*merge^2*gwB + GWb*merge^2 + MH*merge + MW (gwB = gridW/merge),
    // col = C*512 + T*256 + PH*16 + PW, pixel at y=(GHb*merge+MH)*patch+PH,
    // x=(GWb*merge+MW)*patch+PW. The two temporal copies are identical (a still
    // image), so T does not index the pixel.
    public static func patches(_ data: Data, _ grid: VisionGrid) throws
        -> MLMultiArray {
        try patchTensor(normalizedRGB(data, grid), grid.side, 0, 0, grid)
    }
}

public enum VLPrompt {
    // Sent when an image is attached with no text: assert the vision channel
    // first (a text-tuned checkpoint with a grafted tower reflexively claims
    // it cannot see images even while describing one), then ask. The app
    // keeps it out of the transcript (an image-only turn).
    public static let defaultPrompt =
        "The attached image is already encoded by your vision tower; you can "
        + "see it. Describe in detail what you see in this picture."

    // Multi-image (tiling) prompt: a short hint priming the model that the N
    // images are one photo (overview then row-major detail tiles), then the
    // question, carried as template list-content so the model's OWN chat
    // template emits each image's vision placeholder (and its im_start /
    // <think> scaffolding, tracking the template across model versions). Each
    // rendered <|image_pad|> is then expanded into tokensPerImage grid copies,
    // as the HF processor does; returns the ids and each image span's start
    // row.
    public static func buildTiles(_ tok: Tokenizer, template: String,
                                  _ text: String, _ nImages: Int,
                                  _ tokensPerImage: Int)
        throws -> (ids: [Int32], imageStarts: [Int]) {
        var parts: [ContentPart] = []
        if nImages > 1 {
            parts.append(.text(
                "The following \(nImages) images are one picture: an overview "
                + "then left-to-right, top-to-bottom detail tiles. Transcribe "
                + "or describe only what is clearly legible; do not guess or "
                + "invent any text.\n"))
        }
        for _ in 0 ..< nImages { parts.append(.image) }
        parts.append(.text(text))
        let prompt = try renderPrompt(
            template: template,
            messages: [AgentMessage(role: "user", content: text,
                                    contentParts: parts)],
            tools: [], addGenerationPrompt: true, enableThinking: false)
        let pad = tok.encode("<|image_pad|>", addSpecial: true)[0]
        let expanded = Continuation.expandPads(
            tok.encode(prompt, addSpecial: true), pad: pad,
            count: tokensPerImage)
        return (expanded.ids, expanded.starts)
    }
}

// OCR follow-up: this tower is tuned for general vision understanding, not
// dense text. For real OCR, wire IBM's SmolDocling (~256M params, ~250MB) as a
// SEPARATE channel -- a purpose-built document model beats pushing tile
// resolution here. Orthogonal to the fit/tile knob and the vision tower.

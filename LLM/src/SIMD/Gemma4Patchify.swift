import CoreGraphics
import Foundation
import ImageIO

// An image into the pair the vision tower consumes: [patches][patchDim]
// pixels and a per-patch (x, y). Mirrors Gemma4ImageProcessor.
//
// NATIVE RESOLUTION is the whole design. Rather than squaring the image to a
// fixed grid, it is resized to the largest size that still fits the patch
// BUDGET and whose sides divide pool_kernel * patch_size, then patchified and
// PADDED to that budget with (-1, -1) positions. So the patch count varies
// with the aspect ratio, and the tower's pooler later drops the slots no real
// patch reached -- which is where the variable soft-token count comes from.
//
// Every constant comes from the file (patch_size, pool_kernel,
// max_soft_tokens, rescale_factor, normalize, mean/std), so a re-emit with a
// different budget needs no code change here.

public struct Gemma4Patchify {
    public let cfg: Gemma4VisionConfig
    private let rescale: Float
    private let normalize: Bool
    private let mean: [Float]
    private let std: [Float]

    // Each pooled soft token covers a pool_kernel square of patches, so a
    // patch budget is a soft-token budget times that area. A still image and
    // a video FRAME get different budgets from the file (280 against 70), so
    // it is a parameter rather than a property of the tower.
    public var maxPatches: Int { patchBudget(cfg.maxSoftTokens) }

    public func patchBudget(_ softTokens: Int) -> Int {
        softTokens * cfg.poolKernel * cfg.poolKernel
    }

    public init(_ model: Gemma4Model) throws {
        let g = model.gguf
        cfg = try Gemma4VisionConfig(g)
        rescale = Float(g.double("gemma4.vision.rescale_factor") ?? (1 / 255))
        normalize = g.bool("gemma4.vision.normalize") ?? false
        mean = (g.doubles("gemma4.vision.image_mean") ?? [0, 0, 0])
            .map { v in Float(v) }
        std = (g.doubles("gemma4.vision.image_std") ?? [1, 1, 1])
            .map { v in Float(v) }
        let want = g.string("gemma4.vision.resample") ?? "bicubic"
        // Any other filter would be silently approximated by `.high`.
        if want != "bicubic" {
            throw GGUFErr.parse("gemma4.vision.resample is \(want); this "
                                + "build resizes bicubic")
        }
    }

    // The largest size that yields at most `maxPatches` patches and whose
    // sides divide pool_kernel * patch_size. Both degenerate cases upstream
    // guards (a side rounding to zero) would need an image thinner than one
    // pooled block, so they are refused rather than silently reshaped.
    func target(width: Int, height: Int,
                budget: Int) throws -> (w: Int, h: Int) {
        let patch = cfg.patchSize
        let sideMult = cfg.poolKernel * patch
        let targetPx = Double(budget * patch * patch)
        let totalPx = Double(width * height)
        let factor = (targetPx / totalPx).squareRoot()
        let w = Int((factor * Double(width) / Double(sideMult)).rounded(.down))
            * sideMult
        let h = Int((factor * Double(height) / Double(sideMult)).rounded(.down))
            * sideMult
        if w == 0 || h == 0 {
            throw GGUFErr.parse(
                "\(width)x\(height) resizes to \(w)x\(h); a side must reach "
                + "\(sideMult)px, one pooled block")
        }
        return (w, h)
    }

    // `data` is an encoded image (any format ImageIO reads).
    public func patches(_ data: Data)
        -> (pixels: [Float], pos: [(Int, Int)])? {
        var out: (pixels: [Float], pos: [(Int, Int)])? = nil
        if let img = VisionPreprocess.decodeCapped(data) {
            out = patches(img, budget: maxPatches)
        }
        return out
    }

    // A decoded frame at an explicit budget. Video frames all share one size,
    // so every frame yields the same count and none of them pads.
    public func patches(_ img: CGImage, budget: Int)
        -> (pixels: [Float], pos: [(Int, Int)])? {
        var out: (pixels: [Float], pos: [(Int, Int)])? = nil
        if let size = try? target(width: img.width, height: img.height,
                                  budget: budget),
           let rgb = draw(img, size.w, size.h) {
            out = split(rgb, size.w, size.h, budget)
        }
        return out
    }

    // The resized image as row-major RGB, row 0 the image's top.
    private func draw(_ img: CGImage, _ w: Int, _ h: Int) -> [UInt8]? {
        var out: [UInt8]? = nil
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        if let ctx, let base = ctx.data {
            // Closest CoreGraphics offers; switch if it ever exposes bicubic.
            ctx.interpolationQuality = .high
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
            let p = base.bindMemory(to: UInt8.self, capacity: w * h * 4)
            var rgb = [UInt8](repeating: 0, count: w * h * 3)
            for i in 0..<(w * h) {
                for c in 0..<3 { rgb[i * 3 + c] = p[i * 4 + c] }
            }
            out = rgb
        }
        return out
    }

    // Row-major patches with x fastest, each laid out [y][x][channel], then
    // padded to the budget. A padded slot keeps zero pixels and the (-1, -1)
    // that marks it for the tower's mask and pooler.
    private func split(_ rgb: [UInt8], _ w: Int, _ h: Int, _ budget: Int)
        -> (pixels: [Float], pos: [(Int, Int)]) {
        let patch = cfg.patchSize
        let cols = w / patch, rows = h / patch
        let dim = cfg.patchDim
        var pixels = [Float](repeating: 0, count: budget * dim)
        var pos = [(Int, Int)](repeating: (-1, -1), count: budget)
        for y in 0..<rows {
            for x in 0..<cols {
                let slot = y * cols + x
                pos[slot] = (x, y)
                for iy in 0..<patch {
                    let row = (y * patch + iy) * w + x * patch
                    for ix in 0..<patch {
                        for c in 0..<3 {
                            let raw = Float(rgb[(row + ix) * 3 + c])
                            let v = raw * rescale
                            let n = normalize ? (v - mean[c]) / std[c] : v
                            pixels[slot * dim + (iy * patch + ix) * 3 + c] = n
                        }
                    }
                }
            }
        }
        return (pixels, pos)
    }
}

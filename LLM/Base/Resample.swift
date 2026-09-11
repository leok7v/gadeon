import CoreGraphics
import Foundation

// Pillow's separable resampler, which is what an HF image processor means
// when its config says `resample: bicubic`. CoreGraphics offers no bicubic at
// all -- `.high` is Lanczos-family, and against HF's own pixels that reads
// 0.055 max abs where this reads 0.004, one quantisation step of 1/255.
//
// THE TWO BICUBICS ARE DIFFERENT CURVES, and the familiar one is wrong here:
// GGML and PyTorch use a = -0.75, Pillow uses a = -0.5.
//
// Fixed point at 22 fractional bits, and the signed half-unit bias before the
// truncation, are Pillow's own arithmetic rather than an optimization. They
// are what makes this reproduce its reference to the last unit; a float
// accumulator lands a lane either side of a tie instead.
//
// Every stage works four channels at a time, so the alpha the raster carries
// rides along free and one tap is one vector multiply-accumulate.

public enum Resample {
    // 32 bits, less 8 for a pixel and 2 of headroom: normalized weights sum
    // to one but a cubic overshoots, so a tap sum reaches ~255 * 1.2 * 2^22
    // and stays inside Int32.
    private static let bits = 22

    // Where a pixel sits in a buffer. `tap` steps along the axis being
    // resampled and `line` along the one carried through, so the horizontal
    // and vertical passes are one convolution with the two exchanged.
    private struct Step {
        let tap: Int
        let line: Int
    }

    // One axis' coefficients: per output pixel the first input index it
    // reads, how many it reads, and its weights at a fixed row pitch.
    private struct Kernel {
        let size: Int
        let start: [Int]
        let count: [Int]
        let weight: [Int32]
    }

    // A decoded image as row-major RGB at w x h, row 0 the image's top.

    public static func bicubicRGB(_ img: CGImage, _ w: Int,
                                  _ h: Int) -> [UInt8]? {
        var out: [UInt8]? = nil
        if var rows = raster(img) {
            if w != img.width {
                rows = convolve(rows, kernel(img.width, w),
                                lines: img.height,
                                from: Step(tap: 4, line: img.width * 4),
                                to: Step(tap: 4, line: w * 4))
            }
            if h != img.height {
                rows = convolve(rows, kernel(img.height, h), lines: w,
                                from: Step(tap: w * 4, line: 4),
                                to: Step(tap: w * 4, line: 4))
            }
            out = packed(rows, w * h)
        }
        return out
    }

    // Pillow's cubic: a = -0.5, zero outside [-2, 2].

    private static func cubic(_ x: Double) -> Double {
        let a = -0.5
        let t = abs(x)
        var w = 0.0
        if t < 1 {
            w = ((a + 2) * t - (a + 3)) * t * t + 1
        } else if t < 2 {
            w = (((t - 5) * t + 8) * t - 4) * a
        }
        return w
    }

    // Downsampling WIDENS the support so every input pixel collapsing into
    // one output is averaged, and stretches the curve by the same factor;
    // upsampling leaves both alone and stays sharp. That widening is the
    // antialias an HF processor asks for by default, and dropping it is what
    // makes a naive bicubic alias on a photo.

    private static func kernel(_ inSize: Int, _ outSize: Int) -> Kernel {
        let scale = Double(inSize) / Double(outSize)
        let spread = max(scale, 1.0)
        let support = 2.0 * spread
        let size = Int(support.rounded(.up)) * 2 + 1
        let unit = Double(1 << bits)
        var start = [Int](repeating: 0, count: outSize)
        var count = [Int](repeating: 0, count: outSize)
        var weight = [Int32](repeating: 0, count: outSize * size)
        for o in 0..<outSize {
            // Pixel CENTERS, which is where the halves come from: output
            // pixel o covers the input interval around (o + 0.5) * scale.
            let center = (Double(o) + 0.5) * scale
            let lo = max(Int(center - support + 0.5), 0)
            let hi = min(Int(center + support + 0.5), inSize)
            var taps = [Double](repeating: 0, count: max(hi - lo, 0))
            var sum = 0.0
            for t in 0..<taps.count {
                taps[t] = cubic((Double(t + lo) - center + 0.5) / spread)
                sum += taps[t]
            }
            // Normalizing to sum one is what preserves brightness; a truncated
            // support at an edge would otherwise darken it.
            for t in 0..<taps.count {
                let v = (sum != 0 ? taps[t] / sum : 0) * unit
                weight[o * size + t] = Int32(v + (v < 0 ? -0.5 : 0.5))
            }
            start[o] = lo
            count[o] = taps.count
        }
        return Kernel(size: size, start: start, count: count, weight: weight)
    }

    // One output pixel: its taps summed in fixed point, four channels at
    // once. The half unit seeds the rounding that the shift then completes.

    private static func blend(_ s: UnsafePointer<UInt8>, _ step: Int,
                              _ w: UnsafePointer<Int32>,
                              _ n: Int) -> SIMD4<UInt8> {
        var acc = SIMD4<Int32>(repeating: 1 << (bits - 1))
        for t in 0..<n {
            let p = t * step
            let px = SIMD4<UInt8>(s[p], s[p + 1], s[p + 2], s[p + 3])
            acc &+= SIMD4<Int32>(truncatingIfNeeded: px)
                &* SIMD4<Int32>(repeating: w[t])
        }
        return SIMD4<UInt8>(clamping: acc &>> Int32(bits))
    }

    // One separable pass over `lines` of them, RGBA in and RGBA out.

    private static func convolve(_ src: [UInt8], _ k: Kernel, lines: Int,
                                 from: Step, to: Step) -> [UInt8] {
        let outs = k.start.count
        var out = [UInt8](repeating: 0, count: outs * lines * 4)
        src.withUnsafeBufferPointer { s in
            k.weight.withUnsafeBufferPointer { w in
                out.withUnsafeMutableBufferPointer { d in
                    for line in 0..<lines {
                        for o in 0..<outs {
                            let px = blend(
                                s.baseAddress! + k.start[o] * from.tap
                                    + line * from.line, from.tap,
                                w.baseAddress! + o * k.size, k.count[o])
                            let q = o * to.tap + line * to.line
                            for c in 0..<4 { d[q + c] = px[c] }
                        }
                    }
                }
            }
        }
        return out
    }

    // The image's own pixels as RGBA, drawn 1:1 so nothing resamples here.

    private static func raster(_ img: CGImage) -> [UInt8]? {
        var out: [UInt8]? = nil
        let w = img.width, h = img.height
        let ctx = space(img).flatMap { space in
            CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                      bytesPerRow: w * 4, space: space,
                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        }
        if let ctx, let base = ctx.data {
            ctx.interpolationQuality = .none
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
            let p = base.bindMemory(to: UInt8.self, capacity: w * h * 4)
            out = [UInt8](UnsafeBufferPointer(start: p, count: w * h * 4))
        }
        return out
    }

    // The image's OWN space, so the draw above copies rather than converts.
    // ImageIO honours a PNG's gAMA and cHRM chunks and a JPEG's ICC profile;
    // PIL honours none of them and hands back what the decoder produced. So
    // naming any fixed space here applies a color conversion the reference
    // never applied, and on a gamma-tagged PNG that alone moves pixels by 11
    // of 255. sRGB is the fallback for a source that is not RGB at all, where
    // the draw must convert whatever it is asked for.

    private static func space(_ img: CGImage) -> CGColorSpace? {
        var out = CGColorSpace(name: CGColorSpace.sRGB)
        if let own = img.colorSpace, own.model == .rgb { out = own }
        return out
    }

    private static func packed(_ rgba: [UInt8], _ pixels: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: pixels * 3)
        for i in 0..<pixels {
            for c in 0..<3 { out[i * 3 + c] = rgba[i * 4 + c] }
        }
        return out
    }
}

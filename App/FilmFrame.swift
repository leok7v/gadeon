import CoreGraphics
import SwiftUI

// One video frame as it appears behind the composer, and the box that carries
// it there. Kept apart from FilmStrip -- which owns the cross-fade and reads
// the model -- so the LOOK can be rendered on its own: a mask gradient, a dim
// level and a fit are things that have to be looked at, and a view wired to
// ChatModel cannot be put in front of a renderer.

// A frame crossing from the encoding thread to the main actor. CGImage is not
// Sendable and is immutable in practice; the box says so once rather than at
// every hop.
//
// Compared by INDEX, never by pixels: `onChange` needs equality and two
// frames of one clip can be visually identical without being the same frame.
struct VideoPeek: @unchecked Sendable, Equatable {
    let index: Int
    let image: CGImage

    static func == (a: VideoPeek, b: VideoPeek) -> Bool {
        a.index == b.index
    }

    // Scaled to composer size HERE, on the thread that decoded the frame --
    // the full one is a source-resolution bitmap that CoreAnimation would
    // otherwise upload whole to draw a hundred points of it, and it is
    // released the moment the encoder moves on. Not on a View, which is
    // main-actor isolated where this runs anywhere but.
    init?(index: Int, full: CGImage, side: Int = 480) {
        let w = full.width, h = full.height
        let factor = min(1, Double(side) / Double(max(w, h)))
        let tw = max(1, Int(Double(w) * factor))
        let th = max(1, Int(Double(h) * factor))
        var made: CGImage? = nil
        if let space = CGColorSpace(name: CGColorSpace.sRGB),
           let ctx = CGContext(
               data: nil, width: tw, height: th, bitsPerComponent: 8,
               bytesPerRow: 0, space: space,
               bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) {
            ctx.interpolationQuality = .medium
            ctx.draw(full, in: CGRect(x: 0, y: 0, width: tw, height: th))
            made = ctx.makeImage()
        }
        if let made {
            self.index = index
            self.image = made
        } else {
            return nil
        }
    }

    // For rendering the look without an encoder in front of it.
    init(index: Int, ready: CGImage) {
        self.index = index
        self.image = ready
    }
}

struct FilmFrame: View {

    let image: CGImage
    // This layer's own alpha within the cross-fade. The DIM IS NOT APPLIED
    // HERE: it belongs to the pair, not to a member of it -- see FilmStrip.
    let fade: Double

    // TUNED BY EYE, five pairs rendered side by side over a real zoo frame,
    // and the render corrected the reasoning behind them. The first guess
    // (0.32 / 6) was picked to keep the controls readable and left the
    // picture an unrecognisable smudge -- losing the only thing it is for.
    //
    // What actually protects the controls is the MASK, not the dim: they sit
    // at the left and right edges, which is exactly where the frame is
    // faded to nothing. So the middle can be far brighter than it looks like
    // it should be. At 0.70 the elephant behind the man is visible, which is
    // the whole point, and the glyphs are untouched.
    //
    // Not brighter still (0.85 / no blur reads well too) because this clip is
    // DARK footage; a beach or a snow slope at that level would wash out the
    // placeholder text, and the value has to hold for any video.
    static let dim: Double = 0.70
    var blur: CGFloat = 1

    var body: some View {
        Image(decorative: image, scale: 1)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .blur(radius: blur)
            .mask(FilmFrame.edges)
            .opacity(fade)
    }

    // Opaque through the middle, gone at ALL FOUR edges, so the frame
    // dissolves into the card on every side instead of ending on a cut. The
    // vertical feather is what lets it fill the card's height: without it the
    // picture is sliced flat where the card ends.
    //
    // Masking a mask multiplies the two alphas, which is the product wanted
    // -- a corner is faded by both.
    static var edges: some View {
        LinearGradient(
            stops: [.init(color: .clear, location: 0),
                    .init(color: .black, location: 0.18),
                    .init(color: .black, location: 0.82),
                    .init(color: .clear, location: 1)],
            startPoint: .leading, endPoint: .trailing)
        .mask {
            LinearGradient(
                stops: [.init(color: .clear, location: 0),
                        .init(color: .black, location: 0.22),
                        .init(color: .black, location: 0.78),
                        .init(color: .clear, location: 1)],
                startPoint: .top, endPoint: .bottom)
        }
    }
}

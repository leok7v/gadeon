import CoreGraphics
import SwiftUI

struct VideoPeek: @unchecked Sendable, Equatable {

    let index: Int
    let image: CGImage

    static func == (a: VideoPeek, b: VideoPeek) -> Bool {
        a.index == b.index
    }

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

    init(index: Int, ready: CGImage) {
        self.index = index
        self.image = ready
    }
}

struct FilmFrame: View {

    let image: CGImage
    let fade: Double
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

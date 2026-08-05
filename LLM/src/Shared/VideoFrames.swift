// Sample frames from a video, with the timestamp of each.
//
// Model-agnostic: the frame COUNT is a parameter, because it is a property of
// a model's processor rather than of video. gemma-4 takes it from
// `gemma4.video.num_frames`, and its `gemma4.video.fps` of 0 records that
// frames are spread across the WHOLE duration rather than sampled at a rate.
//
// Fewer frames than asked for is not an error here. Upstream raises, because
// its sampler asserts -- but the soft-token count is already variable per
// item, so a short clip simply yields fewer frames rather than being refused.
import AVFoundation
import CoreGraphics
import ImageIO
import Foundation

public enum VideoFrames {
    public enum Failure: Error {
        case noVideoTrack(String)
    }

    // `count` frames spread evenly over the clip, and the seconds at which
    // each was taken -- the caller needs those to label them.
    public static func sample(url: URL, count: Int) async throws
        -> (images: [CGImage], seconds: [Double]) {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        var out: ([CGImage], [Double]) = ([], [])
        if tracks.isEmpty {
            throw Failure.noVideoTrack(url.lastPathComponent)
        } else {
            let duration = try await asset.load(.duration).seconds
            let rate = try await tracks[0].load(.nominalFrameRate)
            let total = max(1, Int((duration * Double(rate)).rounded()))
            let take = min(count, total)
            let gen = AVAssetImageGenerator(asset: asset)
            gen.appliesPreferredTrackTransform = true
            // Exact frames, not the nearest keyframe: a 19s clip has few
            // keyframes, and snapping would sample the same picture repeatedly.
            gen.requestedTimeToleranceBefore = .zero
            gen.requestedTimeToleranceAfter = .zero
            var images: [CGImage] = []
            var seconds: [Double] = []
            for i in 0..<take {
                // Frame indices spread endpoint to endpoint, matching a
                // linspace over the clip rather than a fixed cadence.
                let index = take == 1 ? 0
                    : Int((Double(i) * Double(total - 1)
                           / Double(take - 1)).rounded())
                let at = Double(index) / Double(max(rate, 1))
                let time = CMTime(seconds: at, preferredTimescale: 600)
                if let img = try? await gen.image(at: time).image {
                    images.append(img)
                    seconds.append(at)
                }
            }
            out = (images, seconds)
        }
        return out
    }

    // mm:ss, the label gemma-4 puts before each frame's span.
    public static func stamp(_ seconds: Double) -> String {
        String(format: "%02d:%02d", Int(seconds) / 60, Int(seconds) % 60)
    }
}

public extension VideoFrames {
    // The sampled frames as PNGs, for looking at what the tower was handed.
    static func write(_ images: [CGImage], to dir: String) throws {
        let base = URL(fileURLWithPath: dir)
        try FileManager.default.createDirectory(
            at: base, withIntermediateDirectories: true)
        for (i, img) in images.enumerated() {
            let url = base.appendingPathComponent(String(format: "%02d.png", i))
            if let dst = CGImageDestinationCreateWithURL(
                url as CFURL, "public.png" as CFString, 1, nil) {
                CGImageDestinationAddImage(dst, img, nil)
                CGImageDestinationFinalize(dst)
            }
        }
    }
}

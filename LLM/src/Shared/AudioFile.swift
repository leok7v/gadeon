// Decode audio into mono f32 samples at a requested rate, for any container
// and codec the system reads.
//
// Model-agnostic on purpose: the target rate is a PARAMETER, because it is a
// property of a model's frontend rather than of audio. gemma-4 takes it from
// `gemma4.audio.mel.sample_rate`; another model reads its own key, and this
// file never learns which.
//
// AVAssetReader rather than AVAudioFile because the source may be a VIDEO
// container whose audio is one track among several, and AVAudioFile opens
// only audio files. The reader also performs the resample and the
// stereo-to-mono downmix itself, through its output settings, so no separate
// converter stage exists to disagree with them.
import AVFoundation
import Foundation

public enum AudioFile {
    public enum Failure: Error {
        case noAudioTrack(String)
        case unreadable(String)
    }

    // The whole file unless `seconds` clips it, starting at `offset`.
    public static func samples(url: URL, sampleRate: Double,
                               offset: Double = 0,
                               seconds: Double? = nil) async throws
        -> [Float] {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        var out: [Float] = []
        if tracks.isEmpty {
            throw Failure.noAudioTrack(url.lastPathComponent)
        } else {
            let reader = try AVAssetReader(asset: asset)
            let scale = CMTimeScale(sampleRate)
            let from = CMTime(seconds: max(0, offset), preferredTimescale: scale)
            let span = seconds.map { s in
                CMTime(seconds: max(0, s), preferredTimescale: scale)
            } ?? CMTime.positiveInfinity
            reader.timeRange = CMTimeRange(start: from, duration: span)
            let output = AVAssetReaderAudioMixOutput(
                audioTracks: tracks, audioSettings: settings(sampleRate))
            reader.add(output)
            reader.startReading()
            out = drain(output)
            if reader.status == .failed {
                throw Failure.unreadable(
                    reader.error?.localizedDescription ?? "read failed")
            }
        }
        return out
    }

    // Interleaved is irrelevant at one channel, but stating it keeps the
    // request unambiguous for the mixer that performs the downmix.
    private static func settings(_ rate: Double) -> [String: Any] {
        [AVFormatIDKey: kAudioFormatLinearPCM,
         AVSampleRateKey: rate,
         AVNumberOfChannelsKey: 1,
         AVLinearPCMBitDepthKey: 32,
         AVLinearPCMIsFloatKey: true,
         AVLinearPCMIsBigEndianKey: false,
         AVLinearPCMIsNonInterleaved: false]
    }

    private static func drain(_ output: AVAssetReaderOutput) -> [Float] {
        var out: [Float] = []
        var sample = output.copyNextSampleBuffer()
        while sample != nil {
            if let block = CMSampleBufferGetDataBuffer(sample!) {
                let n = CMBlockBufferGetDataLength(block)
                var bytes = [UInt8](repeating: 0, count: n)
                CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: n,
                                           destination: &bytes)
                bytes.withUnsafeBytes { raw in
                    let f = raw.bindMemory(to: Float.self)
                    out.append(contentsOf: f)
                }
            }
            sample = output.copyNextSampleBuffer()
        }
        return out
    }
}

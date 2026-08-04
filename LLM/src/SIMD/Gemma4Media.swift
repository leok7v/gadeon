import CoreGraphics
import Foundation

// An attachment into the SoftSpan a turn carries for it: patchify or
// log-mel it, run the tower, and bracket the rows in the markers the file
// itself names.
//
// ONE definition on purpose. The markup, the per-frame budget and the tower
// choice have to agree between the CLI probes and the app, and every constant
// in them is a `gemma4.*` metadata read -- so a re-emit that moves a marker or
// changes a budget moves both callers at once.
//
// Decoding is the CALLER's: this takes encoded image bytes, PCM samples, and
// already-sampled frames, so it stays synchronous and free of AVFoundation.
// A CLASS, and the towers are cached: building one reads its config and, on
// the GPU arm, a CPU twin for the shared pooling -- which a conversation that
// attaches on several turns would otherwise pay every time.
// @unchecked Sendable on the same argument the backends make: the app builds
// one per loaded model and every send is serialized behind `busy`, so the
// cached towers are never touched concurrently.
// An operator- AND user-facing failure. CustomStringConvertible so a caller
// that interpolates it prints the sentence rather than the enum case: the
// clip ceiling is something a user can act on, and `parse("clip is 40.0s...")`
// buries that in syntax.
public struct Gemma4MediaError: Error, CustomStringConvertible {
    public let description: String
}

public final class Gemma4Media: @unchecked Sendable {
    let model: Gemma4Model
    let tokenizer: GemmaTokenizer
    // The GPU towers are each other's oracle with the CPU ones, so which arm
    // runs is a caller's choice rather than a property of the attachment. A
    // context IS that choice: the GPU arm needs one and cannot make its own
    // (the text engine owns the model's single mapping), so nil is the CPU arm
    // and there is no second flag to disagree with it.
    let ctx: MetalContext?
    private var vision: VisionTower?
    private var audio: AudioTower?

    public init(_ chat: GemmaChat, ctx: MetalContext? = nil) {
        model = chat.model
        tokenizer = chat.tokenizer
        self.ctx = ctx
    }

    // A still image. The soft-token count comes from the TOWER, never a
    // constant: this tower is native-resolution, so the count follows the
    // aspect ratio.
    // `softTokens` overrides the file's per-image ceiling. A smaller budget
    // is what a MULTI-image turn wants: this model windows 28 of its 35
    // layers at 512, so two images at the full ceiling push the first one out
    // of view exactly as the second arrives.
    public func image(_ data: Data,
                      softTokens: Int? = nil) throws -> SoftSpan {
        let wire = try Gemma4VisionWire(model.gguf)
        let patch = try Gemma4Patchify(model)
        var out: SoftSpan? = nil
        let budget = softTokens.map { n in patch.patchBudget(n) }
            ?? patch.maxPatches
        if let img = VisionPreprocess.decodeCapped(data),
           let cut = patch.patches(img, budget: budget) {
            let got = try tower(cut.pixels, cut.pos)
            out = SoftSpan.bracketed(begin: wire.boi, placeholder: wire.token,
                                     end: wire.eoi, count: got.count,
                                     features: got.proj)
        }
        if out == nil {
            throw Gemma4MediaError(description:
                "That picture could not be read.")
        }
        return out!
    }

    // A clip, as mono samples at the frontend's own rate (Gemma4MelConfig
    // names it, and AudioFile resamples to it).
    //
    // SEVERAL spans, because the tower hears `maxSeconds` at a time and a
    // recording is not obliged to be that short. The clip is cut at its own
    // pauses (AudioChunks) and each piece becomes a span; the turn then emits
    // one <|audio|> per piece and the template decides where they sit.
    //
    // MEASURED before it was built, since two IMAGES in one turn degrade
    // badly and same-kind multi-span could not be assumed: a 40 s recording
    // cut into 29.7 s + 10.3 s returns all ten of its sentences in order,
    // across the seam, and matches what the two pieces transcribe separately.
    public func audio(_ pcm: [Float]) throws -> [SoftSpan] {
        let wire = try Gemma4AudioWire(model.gguf)
        let mel = Gemma4Mel(Gemma4MelConfig(model.gguf) ?? .processorDefault)
        let tower = try audioTower()
        return AudioChunks.split(pcm, rate: Double(mel.cfg.sampleRate),
                                 maxSeconds: wire.maxSeconds)
            .map { chunk in
                let feats = mel.features(Array(pcm[chunk.range]))
                let got = tower.run(feats.values, feats.frames, mel.cfg.bins)
                return SoftSpan.bracketed(
                    begin: wire.boa, placeholder: wire.token, end: wire.eoa,
                    count: got.count, features: got.proj)
            }
    }

    // A video: the vision tower per frame, since there are no video weights.
    // What differs from a still is the BUDGET (a frame gets
    // video.max_soft_tokens, far under an image's) and the markup -- each
    // frame carries its own mm:ss stamp and its own begin/end pair, which is
    // how the model is told when it happened.
    public func video(frames: [CGImage], seconds: [Double]) throws -> SoftSpan {
        let iw = try Gemma4VisionWire(model.gguf)
        let film = try Gemma4VideoWire(model.gguf)
        let patch = try Gemma4Patchify(model)
        let budget = patch.patchBudget(film.softTokensPerFrame)
        // Built ONCE: each tower construction prewarms a Metal context and a
        // CPU twin, which per frame would dominate the encode.
        let vit = try visionTower()
        var ids: [Int32] = []
        var feats: [Float] = []
        var i = 0
        while i < frames.count,
              let cut = patch.patches(frames[i], budget: budget) {
            let got = vit.run(cut.pixels, cut.pos)
            feats.append(contentsOf: got.proj)
            ids.append(contentsOf: tokenizer.encode(
                VideoFrames.stamp(seconds[i]) + " ", addSpecial: false))
            ids.append(contentsOf: SoftSpan.bracket(
                begin: iw.boi, placeholder: film.token, end: iw.eoi,
                count: got.count))
            i += 1
        }
        if i < frames.count {
            throw Gemma4MediaError(description:
                "Frame \(i) of that video could not be read.")
        }
        return SoftSpan(placeholder: film.token, ids: ids, features: feats)
    }

    // ---- from a file ----------------------------------------------------
    // The URL forms own the two numbers a caller would otherwise have to know
    // and could get wrong silently: the frontend's sample rate, and how many
    // frames this processor samples from a clip. Both are file metadata.

    public func audio(url: URL) async throws -> [SoftSpan] {
        let rate = Double((Gemma4MelConfig(model.gguf)
            ?? .processorDefault).sampleRate)
        return try audio(await AudioFile.samples(url: url, sampleRate: rate))
    }

    // The audio track of a video is NOT included here: a caller that wants
    // both attaches both, which is also what lets it put them in its own
    // order.
    public func video(url: URL) async throws -> SoftSpan {
        let film = try Gemma4VideoWire(model.gguf)
        let shot = try await VideoFrames.sample(url: url, count: film.frames)
        return try video(frames: shot.images, seconds: shot.seconds)
    }

    // What a microphone must capture at for this model's frontend, so a
    // caller never has to know which metadata key names it.
    public var audioSampleRate: Double {
        Double((Gemma4MelConfig(model.gguf) ?? .processorDefault).sampleRate)
    }

    // How long a clip may be, so a picker can refuse before the decode.
    public var maxAudioSeconds: Double {
        ((try? Gemma4AudioWire(model.gguf))?.maxSeconds) ?? 0
    }

    // ---- towers ---------------------------------------------------------

    // Either vision tower behind one call, so the frame loop above is written
    // once. Both expose the identical contract.
    struct VisionTower {
        let run: ([Float], [(Int, Int)]) -> (tower: [Float], proj: [Float],
                                             count: Int)
    }

    func visionTower() throws -> VisionTower {
        if vision == nil {
            if let ctx {
                let gpu = try Gemma4MetalViT(model, ctx: ctx)
                vision = VisionTower { pixels, pos in
                    gpu.forward(pixels: pixels, pos: pos)
                }
            } else {
                let cpu = try Gemma4ViT(model)
                vision = VisionTower { pixels, pos in
                    cpu.forward(pixels: pixels, pos: pos)
                }
            }
        }
        return vision!
    }

    private func tower(_ pixels: [Float], _ pos: [(Int, Int)])
        throws -> (tower: [Float], proj: [Float], count: Int) {
        try visionTower().run(pixels, pos)
    }

    struct AudioTower {
        let run: ([Float], Int, Int) -> (tower: [Float], proj: [Float],
                                         count: Int)
    }

    func audioTower() throws -> AudioTower {
        if audio == nil {
            if let ctx {
                let gpu = try Gemma4MetalAudio(model, ctx: ctx)
                audio = AudioTower { mel, frames, bins in
                    gpu.forward(mel: mel, frames: frames, bins: bins)
                }
            } else {
                let cpu = try Gemma4Audio(model)
                audio = AudioTower { mel, frames, bins in
                    cpu.forward(mel: mel, frames: frames, bins: bins)
                }
            }
        }
        return audio!
    }
}

import Foundation
import Testing
@testable import LLM

// The gemma-4 speech path end to end, on REAL audio: decode the file,
// log-mel it, run the tower, splice the soft tokens into a turn the template
// rendered, and read back what the model says.
//
// WHY end-to-end rather than another cosine. The tower's last two layers
// amplify their input about 175x, so a cosine against a reference measures
// how badly that reference reproduces ITSELF (0.9585 across dtypes) rather
// than anything about this port. What a transcription cannot fake is whether
// the whole chain -- resample, mel, tower, projection, splice, PLE pad row --
// is right: get any one of them wrong and the words stop being words. The
// SRQ-off checkpoint failed exactly here, answering "the content is unclear"
// while every cosine still looked plausible.
//
// The SAME sentence in two languages, so a pass is about the model and not
// about one lucky clip. Fixtures + attribution in tests/audio/.
//
// Network-free but weight-gated on GADEON_GEMMA_GGUF (see
// needsGemmaWeights), so without the file these report as SKIPPED rather
// than passing on an empty body.
struct Gemma4AudioTests {
    // Repo root: this file sits in LLM/tests.
    private static func fixture(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("tests/audio")
            .appendingPathComponent(name)
    }

    // One clip through the whole path, decoded GREEDILY -- no sampler, so the
    // engine takes the argmax and the result is reproducible. The Metal arm
    // runs it because it is this model's shipping path; the two engines are
    // gated against each other in GemmaMetalKernelTests.
    private func transcribe(_ path: String, _ file: String,
                            _ ask: String) async throws -> String {
        let chat = try GemmaChat(ggufPath: path)
        let mel = chat.melConfig
        let pcm = try await AudioFile.samples(
            url: Gemma4AudioTests.fixture(file),
            sampleRate: Double(mel.sampleRate))
        let engine = try Gemma4MetalEngine(chat.model)
        let spans = try Gemma4Media(chat, ctx: engine.ctx).audio(pcm)
        let parts = [ContentPart](repeating: .audio, count: spans.count)
        let prompt = try renderPrompt(
            template: chat.chatTemplate,
            messages: [AgentMessage(role: "user", content: ask,
                                    contentParts: parts + [.text(ask)])],
            tools: [], addGenerationPrompt: true, enableThinking: false,
            bosToken: chat.bosToken)
        let ids = Continuation.expandSpans(chat.encode(prompt), spans)
        engine.reset()
        let feed = SoftFeed(spans)
        var next = engine.extend(ids, softAt: { id in feed.row(id) })
        var out: [Int32] = []
        while out.count < 24 && !chat.eosIds.contains(next) {
            out.append(next)
            next = engine.decode(next)
        }
        return chat.decode(out)
    }

    private static let ask = "Transcribe this audio exactly."

    // A CONTENT word rather than the whole string: the sentence is what the
    // chain has to recover, while the punctuation and any leading politeness
    // are the sampler's business and would make this brittle without making
    // it stronger.
    @Test(needsGemmaWeights) func englishClipTranscribes() async throws {
        let path = try #require(gemmaGgufPath)
        let said = try await transcribe(
            path, "en-lets-leave-tomorrow.wav", Gemma4AudioTests.ask)
        let why = "expected \"Let's leave tomorrow.\", heard: \(said)"
        #expect(said.lowercased().contains("tomorrow"), "\(why)")
        #expect(said.lowercased().contains("leave"), "\(why)")
    }

    // The same sentence in French. It also proves the turn is not merely
    // echoing an English prior: "demain" cannot come from anywhere but the
    // audio.
    @Test(needsGemmaWeights) func frenchClipTranscribes() async throws {
        let path = try #require(gemmaGgufPath)
        let said = try await transcribe(
            path, "fr-partons-demain.wav", Gemma4AudioTests.ask)
        let why = "expected \"Partons demain.\", heard: \(said)"
        #expect(said.lowercased().contains("demain"), "\(why)")
        #expect(said.lowercased().contains("partons"), "\(why)")
    }

    // A clip past the tower's ceiling is CUT, not refused: it comes back as
    // several spans covering the whole recording, because the tower hears
    // maxSeconds at a time and a recording is not obliged to be that short.
    // Silence is the input on purpose -- there are no pauses to find in it,
    // so the packer must still cover the clip by falling back to hard cuts
    // at the ceiling rather than emitting one span or looping.
    @Test(needsGemmaWeights) func overlongClipIsChunked() throws {
        let path = try #require(gemmaGgufPath)
        let chat = try GemmaChat(ggufPath: path)
        let wire = try chat.audioWire()
        let rate = chat.melConfig.sampleRate
        let tooLong = Int(Double(rate) * (wire.maxSeconds + 5))
        let media = Gemma4Media(chat)
        let spans = try media.audio([Float](repeating: 0, count: tooLong))
        #expect(spans.count > 1,
                "a clip past the ceiling must be cut: \(spans.count) span(s)")
        #expect(spans.allSatisfy { s in s.rows > 0 })
    }
}
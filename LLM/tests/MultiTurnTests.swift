import CoreML
import Foundation
import XCTest
@testable import LLM

// In-process state-fidelity oracle for the O(delta) bookmark continuation
// (model-gated: needs models/Qwen3.5-0.8B). Free-running decode cannot be
// bit-compared on the ANE -- fp16 is not reproducible run to run, so two
// identical greedy runs diverge once any argmax tie flips. The one thing stable
// WITHIN a process is the next-token distribution at a fixed generation point,
// so this computes it several ways over identical content and compares argmax +
// logit-vector cosine, as a LADDER that localizes any divergence:
//   s1  seq=1 token-by-token ingest of the whole render  (gold: the decode path)
//   c1  one batched carry block of the whole render       (isolates carryBlock)
//   c2  two carry blocks: base then delta, no bookmark     (isolates continuation)
//   bm  base, bookmark, dirty the KV, restore, delta       (isolates bookmark)
// Each stage should match s1 (same argmax, cosine ~1). The first stage that
// drops tells us where the bug is. This gate is what task 0 (route the app
// through jinja + bookmark) and multicontext both rest on. reasoning-effort
// none (enableThinking false) so the render closes <think>.
final class MultiTurnTests: XCTestCase {
    // The set sits either directly under models/Qwen3.5-0.8B (a dev copy) or
    // in a {sha} subdir (the HubFetch staging layout).
    private func modelsDir() -> URL? {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let base = root.appendingPathComponent("models/Qwen3.5-0.8B")
        let fm = FileManager.default
        var found: URL? = nil
        if fm.fileExists(
            atPath: base.appendingPathComponent("tokenizer.json").path) {
            found = base
        } else if let subs = try? fm.contentsOfDirectory(
            at: base, includingPropertiesForKeys: nil) {
            for sub in subs where fm.fileExists(
                atPath: sub.appendingPathComponent("tokenizer.json").path) {
                found = sub
            }
        }
        return found
    }

    private func argmax(_ v: [Float]) -> Int {
        var best = 0
        var i = 1
        while i < v.count {
            if v[i] > v[best] { best = i }
            i += 1
        }
        return best
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        var i = 0
        while i < a.count && i < b.count {
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
            i += 1
        }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
    }

    private func firstSubsequence(_ hay: [Int32], _ pat: [Int32]) -> Int? {
        var result: Int? = nil
        var i = 0
        let last = hay.count - pat.count
        while result == nil && i <= last {
            var j = 0
            while j < pat.count && hay[i + j] == pat[j] { j += 1 }
            if j == pat.count { result = i }
            i += 1
        }
        return result
    }

    private func report(_ name: String, _ ref: [Float], _ v: [Float]) {
        let a = argmax(ref)
        let b = argmax(v)
        let c = cosine(ref, v)
        let flag = (a == b && c >= 0.99) ? "OK  " : "FAIL"
        print("  \(flag) \(name): argmax=\(b) (ref \(a)) "
            + "cosine=\(String(format: "%.5f", c))")
    }

    func testContinuationFidelityLadder() async throws {
        guard let dir = modelsDir() else {
            throw XCTSkip("models/Qwen3.5-0.8B not present")
        }
        let template = try String(
            contentsOf: dir.appendingPathComponent("chat_template.jinja"),
            encoding: .utf8)
        let chat = try AneChat(modelsDir: dir)
        try await chat.loadHeavy()
        try await chat.loadCarry()
        let tok = chat.tokenizer
        let eng = chat.engine
        // temperature-0, penalty-free sampler: pickToken fills logitsF32 with the
        // RAW distribution (no genSeen penalty), so lastLogits() is the clean
        // next-token vector at whatever position was last ingested.
        await eng.useSampler(Sampler(vocabSize: tok.vocabCount,
            config: SamplerConfig(temperature: 0, setMask: [.temperature])))

        let msgs = [
            AgentMessage(role: "system",
                         content: "You are a helpful assistant."),
            AgentMessage(role: "user",
                         content: "My name is Leo. Remember it."),
            AgentMessage(role: "assistant",
                         content: "Nice to meet you, Leo!"),
            AgentMessage(role: "user", content: "What is my name?"),
        ]
        let promptText = try renderPrompt(
            template: template, messages: msgs, tools: [],
            addGenerationPrompt: true, enableThinking: false)
        XCTAssertTrue(promptText.contains("</think>"),
            "enable_thinking=false did not emit a closed <think></think>")

        let full = tok.encode(promptText, addSpecial: true)
        let openIds = tok.encode("<|im_start|>assistant\n", addSpecial: true)
        let at = firstSubsequence(full, openIds)
        XCTAssertNotNil(at, "no assistant turn found in the render")
        let baseLen = (at ?? 0) + openIds.count
        let base = Array(full[0 ..< baseLen])
        let delta = Array(full[baseLen ..< full.count])
        let raw = tok.encode(
            "<think>\n\n</think>\n\nNice to meet you, Leo!<|im_end|>\n",
            addSpecial: true)

        // The first post-compile execution of a carry program differs
        // numerically on the ANE (the converter's verify() warms the same way
        // before measuring), and a cold run made the bookmark path diverge
        // grossly. Warm every path once, discarded, then measure the steady
        // state -- which is how the app runs after its first turn.
        for _ in 0 ..< 2 {
            _ = try await ladder(eng, full, base, delta, raw)
        }

        // The bookmark bug was intermittent, so gate on EVERY iteration, not the
        // last: track the worst bookmark cosine and any argmax miss across runs.
        var worstBm = 1.0
        var argMisses = 0
        for iter in 1 ... 8 {
            let m = try await ladder(eng, full, base, delta, raw)
            print("iter \(iter) (base=\(base.count) delta=\(delta.count) "
                + "full=\(full.count)) vs s1 (seq=1 gold):")
            report("c1 one-carry-block ", m.s1, m.c1)
            report("c2 two-blocks      ", m.s1, m.c2)
            report("bm bookmark+restore", m.s1, m.bm)
            report("bm vs c2           ", m.c2, m.bm)
            let cos = cosine(m.s1, m.bm)
            if cos < worstBm { worstBm = cos }
            if argmax(m.bm) != argmax(m.s1) { argMisses += 1 }
        }
        print("bookmark over 8 iters: worst-cosine="
            + "\(String(format: "%.5f", worstBm)) argmax-misses=\(argMisses)")
        XCTAssertEqual(argMisses, 0,
            "bookmark continuation diverged from seq=1 in \(argMisses) iters")
        XCTAssertGreaterThanOrEqual(worstBm, 0.99,
            "bookmark-continued state diverges (worst cosine \(worstBm))")

        // Multicontext: park conversation A, run a whole different conversation B
        // over the same engine, resume A -- A's next-token prediction must be
        // unchanged. This is park/resume (Engine.bookmark/restore) transparency,
        // the primitive multicontext rests on.
        let convB = tok.encode(
            "<|im_start|>user\nName three colors.<|im_end|>\n"
            + "<|im_start|>assistant\n", addSpecial: true)
        let probe: [Int32] = [198]                     // a benign next token
        await eng.reset()
        _ = try await eng.ingestBatched(full)
        _ = try await eng.ingest(probe)
        let mcBase = await eng.lastLogits()
        await eng.reset()
        _ = try await eng.ingestBatched(full)
        let parkA = try await eng.bookmark()
        await eng.reset()
        _ = try await eng.ingestBatched(convB)         // run conversation B
        try await eng.restore(parkA)                   // resume A
        _ = try await eng.ingest(probe)
        let mcResumed = await eng.lastLogits()
        print("multicontext park/resume: base=\(argmax(mcBase)) "
            + "resumed=\(argmax(mcResumed)) "
            + "cosine=\(String(format: "%.5f", cosine(mcBase, mcResumed)))")
        XCTAssertEqual(argmax(mcResumed), argmax(mcBase),
            "resumed conversation A diverges after an intervening B")
        XCTAssertGreaterThanOrEqual(cosine(mcBase, mcResumed), 0.99,
            "park/resume is not transparent across an intervening conversation")

        // Persistence: serialize A's bookmark to Data, deserialize it, restore --
        // A's next token must be unchanged (lossless disk round-trip, the
        // precooked-prompt-cache primitive).
        await eng.reset()
        _ = try await eng.ingestBatched(full)
        let bmToSave = try await eng.bookmark()
        let blob = await eng.serialize(bmToSave)
        let restored = try await eng.deserialize(blob)
        try await eng.restore(restored)
        _ = try await eng.ingest(probe)
        let mcRoundtrip = await eng.lastLogits()
        print("serialize round-trip: base=\(argmax(mcBase)) "
            + "roundtrip=\(argmax(mcRoundtrip)) "
            + "cosine=\(String(format: "%.5f", cosine(mcBase, mcRoundtrip))) "
            + "bytes=\(blob.count)")
        XCTAssertEqual(argmax(mcRoundtrip), argmax(mcBase),
            "serialized bookmark diverges from the original")
        XCTAssertGreaterThanOrEqual(cosine(mcBase, mcRoundtrip), 0.99,
            "bookmark serialization is lossy")
    }

    // One pass of the four continuation stages over identical content, each
    // preceded by a reset so it starts from a clean engine.
    private func ladder(_ eng: Engine, _ full: [Int32], _ base: [Int32],
                        _ delta: [Int32], _ raw: [Int32]) async throws
        -> (s1: [Float], c1: [Float], c2: [Float], bm: [Float]) {
        await eng.reset()
        _ = try await eng.ingest(full)                  // seq=1 gold
        let s1 = await eng.lastLogits()

        await eng.reset()
        _ = try await eng.ingestBatched(full)           // one carry block
        let c1 = await eng.lastLogits()

        await eng.reset()
        _ = try await eng.ingestBatched(base)           // two blocks, no bookmark
        _ = try await eng.ingestBatched(delta)
        let c2 = await eng.lastLogits()

        await eng.reset()
        _ = try await eng.ingestBatched(base)           // bookmark + restore
        let mark = try await eng.bookmark()
        _ = try await eng.ingestBatched(raw)
        try await eng.restore(mark)
        _ = try await eng.ingestBatched(delta)
        let bm = await eng.lastLogits()

        return (s1, c1, c2, bm)
    }
}

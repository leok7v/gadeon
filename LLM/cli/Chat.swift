import CoreML
import Foundation
import LLM

// The chat paths: the ChatSession turn loop (the shipping route),
// the raw token-by-token / longdoc references, and the ternary
// GGUF (SIMD / Metal) main that serves a .gguf argument end to
// end. Dispatched from main.swift.

// The default --bench prompt: exactly 512 tokens
// under the Qwen3.5-0.8B and QwenPaw-2B tokenizers. Kept VERBATIM and byte-exact
// -- the token count is load-bearing (it must equal llama-bench -p 512), so do
// NOT reflow these paragraphs or the count shifts.
let benchPrompt = """
You are advising the treasury team of a mid-sized manufacturing company. The company has taken on a large amount of floating-rate debt tied to a short-term reference rate, and management is worried that interest rates may rise over the next few years, which would increase their interest expense and squeeze margins. The CFO has heard that interest rate swaps can be used to manage this risk but does not understand how they actually work, what they cost, or what could go wrong. She has asked you to prepare a thorough written explanation that she can share with the board of directors at the next quarterly meeting.

Please write a clear, well-structured explanation that covers all of the following points in detail. First, define what an interest rate swap is and describe the two legs of a plain vanilla fixed-for-floating swap, explaining who pays what to whom and how the net settlement is calculated on each payment date. Second, walk through a concrete numerical example: assume a notional amount of fifty million dollars, a fixed rate of four percent, a floating rate that starts at three percent, and semi-annual payments, and show what happens to the cash flows if the floating rate rises to five percent. Third, explain how entering this swap changes the company's overall interest rate exposure and why it can be described as converting floating-rate debt into synthetic fixed-rate debt. Fourth, describe the main risks the company still faces after entering the swap, including counterparty credit risk, basis risk, and the consequences of wanting to exit the swap early if rates move against them. Finally, summarize the situations in which a swap is a good idea and the situations in which the company might prefer an interest rate cap, a collar, or simply refinancing into fixed-rate debt instead.

Before you begin, note that the board is also concerned about how a swap would appear in the company's financial statements and whether it introduces earnings volatility, so please include a short, non-technical note on how hedge accounting can align the swap's gains and losses with the underlying debt, and what happens if the hedge is later judged to be ineffective. Assume the debt has seven years remaining and cannot be prepaid without a significant penalty.

Please explain everything in plain language suitable for board members who are intelligent but not financial specialists, define any technical terms the first time you use them, and use short worked examples wherever they help make the mechanics concrete. Finally, close with a brief numbered checklist the board can review during the meeting, along with three short questions the directors should ask management before approving any hedging decision.

"""

// Thread-safe accumulator: onReasoning / onTool fire on the session's task
// while the content stream is consumed on this one.
final class Acc: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [String] = []
    func add(_ s: String) { lock.lock(); items.append(s); lock.unlock() }
    var text: String { lock.lock(); defer { lock.unlock() }; return items.joined() }
    var list: [String] { lock.lock(); defer { lock.unlock() }; return items }
}

final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func next() -> Int { lock.lock(); defer { lock.unlock() }; n += 1; return n }
}
let traceCounter = Counter()
func traceWrite(_ obj: [String: Any]) {
    if let traceURL,
       let data = try? JSONSerialization.data(withJSONObject: obj),
       let handle = try? FileHandle(forWritingTo: traceURL) {
        handle.seekToEndOfFile()
        handle.write(data)
        handle.write(Data("\n".utf8))
        try? handle.close()
    }
}

@MainActor
func reportHints(_ s: ChatSession) async {
    let title = await s.makeTitle()
    let hint = await s.makeFollowup()
    err("[hint] title    = \(title.debugDescription)\n")
    err("[hint] followup = \(hint.debugDescription)\n")
}

@MainActor
func runSession(_ s: ChatSession, _ user: String) async {
    let turn = traceCounter.next()
    let t0 = Date()
    print("\nUSER: \(user)\nASSISTANT: ", terminator: ""); fflush(stdout)
    // Reasoning (<think>) streams to stderr, the answer to stdout -- the two
    // distinct streams. In reasoning-effort none there is no reasoning. The
    // accumulators feed the --trace record; onTool captures the call names.
    let reasoning = Acc()
    let tools = Acc()
    var content = ""
    let stream = s.reply(user,
        onReasoning: { r in err(r); reasoning.add(r) },
        onTool: { name in tools.add(name) })
    for await piece in stream {
        print(piece, terminator: ""); fflush(stdout); content += piece
    }
    print()
    let m = await s.lastMetrics
    err(String(format: "[pp %.1f t/s | tg %.1f t/s | ctx %d | think %d | "
        + "gen %d]\n", m.pp, m.tg, m.ctx, m.thinkTokens, m.contentTokens))
    traceWrite([
        "turn": turn,
        "wall_ms": Int(Date().timeIntervalSince(t0) * 1000),
        "user": user,
        "reasoning": reasoning.text,
        "content": content,
        "tool_calls": tools.list,
        "ctx": m.ctx,
        "think_tokens": m.thinkTokens,
        "gen_tokens": m.contentTokens,
        "pp_tps": m.pp.isFinite ? m.pp : 0,
        "tg_tps": m.tg.isFinite ? m.tg : 0,
    ])
}

func segment(_ user: String, first: Bool) -> String {
    (first ? "" : "\n")
        + "<|im_start|>user\n\(user)<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
}

@MainActor
func runPlain(_ user: String, first: Bool) async throws {
    let ids = tok.encode(segment(user, first: first), addSpecial: true)
    print("\nUSER: \(user)\nASSISTANT: ", terminator: ""); fflush(stdout)
    let t0 = Date()
    var next = try await (first ? eng.prefill(ids)
                                : (forceIngest ? eng.ingest(ids)
                                               : eng.ingestBatched(ids)))
    let pfDt = Date().timeIntervalSince(t0)
    var out: [Int32] = []; var stream = StreamDecoder(); let g0 = Date()
    while out.count < maxTokens, next != tok.eosId {
        out.append(next)
        let piece = stream.step(out, tok)           // whole UTF-8 scalars only
        if !piece.isEmpty { print(piece, terminator: ""); fflush(stdout) }
        next = try await eng.decode(next)
    }
    let gDt = Date().timeIntervalSince(g0)
    if next == tok.eosId { try await eng.feed(tok.eosId) }
    print()
    err(String(format: "[tps] prefill %d tok in %.0fms (%.1f t/s) · "
        + "gen %d tok in %.2fs (%.1f t/s) · ctx %d\n",
        ids.count, pfDt * 1000, pfDt > 0 ? Double(ids.count) / pfDt : 0,
        out.count, gDt, gDt > 0 ? Double(out.count) / gDt : 0, await eng.position()))
}

@MainActor
func runTurn(_ user: String, first: Bool) async throws {
    if let session {
        await runSession(session, user)
    } else {
        try await runPlain(user, first: first)
    }
}

// Prove the >512 block-tiled carry prefill against the token-by-token path on
// the SAME prompt: (a) prefill() routes >512 to the ANE carry trunk (time it,
// decode a few tokens); (b) reset + ingest the same ids token-by-token (the
// reference) and decode the same way. A matching first token shows the carry
// wiring reproduces the reference at a large multiple of the throughput.
@MainActor
func runLongDoc(_ user: String) async throws {
    let ids = tok.encode(segment(user, first: true), addSpecial: true)
    err("[longdoc] prompt \(ids.count) tokens (512-block carry)\n")
    let gen = 24
    let t0 = Date()
    var nc = try await eng.prefill(ids)  // routes to carryPrefill
    let pfSec = Date().timeIntervalSince(t0)
    let carry0 = nc
    var carryOut: [Int32] = []
    let g0 = Date()
    while carryOut.count < gen, nc != tok.eosId {
        carryOut.append(nc); nc = try await eng.decode(nc)
    }
    let gSec = Date().timeIntervalSince(g0)
    await eng.reset()
    let s0 = Date()
    var ns = try await eng.ingest(ids)  // token-by-token reference
    let seqSec = Date().timeIntervalSince(s0)
    let seq0 = ns
    var seqOut: [Int32] = []
    while seqOut.count < gen, ns != tok.eosId {
        seqOut.append(ns); ns = try await eng.decode(ns)
    }
    let n = min(carryOut.count, seqOut.count)
    var match = 0
    for i in 0 ..< n where carryOut[i] == seqOut[i] { match += 1 }
    print("\nCARRY: \(tok.decode(carryOut))")
    print("SEQ  : \(tok.decode(seqOut))")
    err(String(format: "[longdoc] carry-prefill %d tok in %.0fms (%.1f t/s) · "
        + "seq-prefill %.0fms (%.1f t/s) · first-tok carry=%d seq=%d · "
        + "continuation match %d/%d · gen %d tok in %.2fs (%.1f t/s)\n",
        ids.count, pfSec * 1000, pfSec > 0 ? Double(ids.count) / pfSec : 0,
        seqSec * 1000, seqSec > 0 ? Double(ids.count) / seqSec : 0,
        carry0, seq0, match, n,
        carryOut.count, gSec, gSec > 0 ? Double(carryOut.count) / gSec : 0))
    // The first token is the argmax over the WHOLE carried context, so an
    // exact match to the token-serial reference is the real correctness gate;
    // a <n continuation match is fp16 greedy divergence, not a block-boundary
    // bug.
    err(carry0 == seq0
        ? "[longdoc] first-token MATCH vs token-serial reference "
          + "(continuation \(match)/\(n); any drift is fp16 greedy divergence)\n"
        : "[longdoc] MISMATCH -- first token differs, inspect block boundaries\n")
}

// Bonsai / ternary GGUF path: arg1 is a .gguf file -> run the pure-Swift SIMD
// (CPU) engine through the SAME backend-agnostic ChatSession the CoreML path
// uses. Everything above the AgentBackend seam (tokenizer, jinja template,
// sampler, multi-turn continuation) is shared; only the compute backend differs.
@MainActor func runGgufMain() async throws {
    // BEFORE the architecture routing on purpose: the kernels are shared, so
    // a golden capture has to run over BOTH lineages from one entry point.
    if let dir = metalGoldenDir {
        print(try MetalGolden.run(ggufPath: arg1, dir: dir))
        exit(0)
    }
    // gemma-4 is a different architecture behind the same .gguf extension, so
    // route on the FILE's own general.architecture, never on its name.
    if Gemma4Model.isGemma4(path: arg1) {
        try await runGemmaMain(arg1, rawArgs, turnArgs, capVal)
    }
    // Metal backend bring-up: diff each GPU kernel against the SIMD reference
    // on the loaded model, then exit. No chat -- correctness only.
    if rawArgs.contains("--metal-selftest") {
        err("Metal self-test on \(arg1)...\n")
        print(try MetalSelfTest.run(ggufPath: arg1))
        exit(0)
    }
    // Slugs semantic-search bring-up (minilm.gguf is a BERT embedder + index,
    // not a chat model): --slugs-embed prints the 384-d embedding for the
    // C-reference cosine gate; --slugs prints the top-K article matches.
    if rawArgs.contains("--slugs-embed") {
        let text = rawArgs.dropFirst(2).first { !$0.hasPrefix("--") } ?? ""
        if let w = WikiSlugs(ggufPath: arg1) {
            print(w.embed(text)
                .map { String(format: "%.7f", $0) }.joined(separator: " "))
        } else {
            err("slugs: \(arg1) has no index trailer\n")
        }
        exit(0)
    }
    if rawArgs.contains("--slugs") {
        let text = rawArgs.dropFirst(2).first { !$0.hasPrefix("--") } ?? ""
        if let w = WikiSlugs(ggufPath: arg1) {
            err("slugs: \(w.articleCount) articles\n")
            for (r, h) in w.query(text, topK: 5).enumerated() {
                let low = h.distance > 82 ? "  [low]" : ""
                print("  \(r + 1). d=\(h.distance)  id=\(h.id)  \(h.title)\(low)")
            }
        } else {
            err("slugs: \(arg1) has no index trailer\n")
        }
        exit(0)
    }
    // Full wikipedia_query tool: on-device search + live extracts-API fetch +
    // paragraph/sentence-aware truncation (hits the network).
    if rawArgs.contains("--wiki") {
        let text = rawArgs.dropFirst(2).first { !$0.hasPrefix("--") } ?? ""
        print(await Tools.wikipediaQuery(text, slugsPath: arg1))
        exit(0)
    }
    // Tokenizer probe: encode each following arg, print ids + decoded pieces to
    // verify merge-rank BPE segmentation and atomic special tokens.
    if let ti = rawArgs.firstIndex(of: "--tok") {
        let tok = try BonsaiChat(ggufPath: arg1).tokenizer
        for s in rawArgs[(ti + 1)...] {
            let ids = tok.encode(s, addSpecial: true)
            let pieces = ids.map { tok.decode([$0]) }
            print("\(ids.count) ids  \(ids)\n  pieces \(pieces)")
        }
        exit(0)
    }
    // GDN layer-0 reference dump for the CoreML/ANE port gate: run the SIMD
    // BonsaiEngine over 64 synthetic tokens from a fresh reset (so the layer-0
    // recurrence starts at S0=0) and tap the GDN block's input (attn_norm) and
    // output (attn_out) per token. The offline emit gates its ANE program against
    // these raw f32 [64, nEmbd] tensors. Layer 0 is GDN ((0+1)%4 != 0).
    if let gi = rawArgs.firstIndex(of: "--gdn-dump") {
        let dir = gi + 1 < rawArgs.count && !rawArgs[gi + 1].hasPrefix("--")
            ? rawArgs[gi + 1] : "build/gdn-ref"
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        let model = try BonsaiModel(path: arg1)
        let eng = BonsaiEngine(model)
        let seq = 64
        let ids = (0..<seq).map { Int32(($0 * 6151 + 17) % model.cfg.nVocab) }
        var gin: [Float] = [], gout: [Float] = []
        eng.reset()
        var p = 0
        for id in ids {
            eng.forward(token: Int(id), pos: p) { name, il, v in
                if il == 0 && name == "attn_norm" { gin += v }
                if il == 0 && name == "attn_out" { gout += v }
            }
            p += 1
        }
        func writeBin(_ a: [Float], _ f: String) {
            a.withUnsafeBytes { raw in
                try? Data(raw).write(to: URL(fileURLWithPath: dir + "/" + f))
            }
        }
        writeBin(gin, "gdn_in.bin")
        writeBin(gout, "gdn_out.bin")
        writeBin(ids.map { Float($0) }, "ids.bin")
        err("gdn-dump: layer0 seq=\(seq) nEmbd=\(model.cfg.nEmbd) -> \(dir)\n")
        exit(0)
    }
    if let hi = rawArgs.firstIndex(of: "--hess") {
        let dir = rawArgs[hi + 1]
        let lo = Int(rawArgs[hi + 2])!
        let up = Int(rawArgs[hi + 3])!
        let corpus = rawArgs[hi + 4]
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        let text = (try? String(contentsOfFile: corpus, encoding: .utf8)) ?? ""
        let chat = try MetalChat(ggufPath: arg1)
        let ids = chat.tokenizer.encode(text, addSpecial: true)
        chat.engine.reset()
        chat.engine.collectHessians(from: lo, upto: up)
        let t0 = Date()
        for id in ids {
            _ = chat.engine.decode(id)
        }
        for name in chat.engine.hessianNames() {
            try chat.engine.hessianBytes(name)
                .write(to: URL(fileURLWithPath: dir + "/" + name + ".h32"))
        }
        err(String(format: "[hess] %d tokens, layers %d..%d, %.0fs -> %@\n",
                   ids.count, lo, up, Date().timeIntervalSince(t0), dir))
        exit(0)
    }
    if let mi = rawArgs.firstIndex(of: "--imat") {
        let dir = mi + 1 < rawArgs.count ? rawArgs[mi + 1] : "tmp/imat"
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        let chat = try MetalChat(ggufPath: arg1)
        var ids = chat.tokenizer.encode(benchPrompt, addSpecial: true)
        if let benchCtxVal {
            var padded = ids
            while padded.count < benchCtxVal { padded += ids }
            ids = Array(padded.prefix(benchCtxVal))
        }
        chat.engine.reset()
        chat.engine.collectImatrix()
        for id in ids {
            _ = chat.engine.decode(id)
        }
        let sums = chat.engine.imatrixSums()
        for (name, v) in sums {
            v.withUnsafeBytes { raw in
                try? Data(raw).write(to: URL(
                    fileURLWithPath: dir + "/" + name + ".bin"))
            }
        }
        err("[imat] \(ids.count) tokens, \(sums.count) sites -> \(dir)\n")
        exit(0)
    }
    if let ti = rawArgs.firstIndex(of: "--tap") {
        let dir = ti + 1 < rawArgs.count ? rawArgs[ti + 1] : "tmp/tap"
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        let chat = try MetalChat(ggufPath: arg1)
        let ids = chat.tokenizer.encode(
            probeWrap(turnArgs.first ?? "What is 2+2?"), addSpecial: true)
        chat.engine.reset()
        for id in ids.dropLast() {
            _ = chat.engine.decode(id)
        }
        let layers = chat.engine.tapLayers(token: Int(ids.last!))
        for (i, v) in layers.enumerated() {
            v.withUnsafeBytes { raw in
                try? Data(raw).write(to: URL(
                    fileURLWithPath: String(format: "%@/l%02d.bin", dir, i)))
            }
        }
        err("[tap] \(layers.count) vectors of \(layers[0].count) -> \(dir)\n")
        exit(0)
    }
    // Minimal greedy templated probe (GGUF reference): the same probeWrap +
    // argmax + EOS-stop as the CoreML --probe, over the AgentBackend, so a
    // CoreML-vs-GGUF answer comparison uses the identical prompt and decode rule.
    // --metal routes the fast GPU backend (pp30/tg8 t/s) -- the ground truth on
    // the same 2-bit weights; default is the SIMD/CPU engine.
    if rawArgs.contains("--probe") {
        let backend: any AgentBackend
        let ptok: Tokenizer
        if rawArgs.contains("--metal") {
            let chat = try MetalChat(ggufPath: arg1)
            backend = chat.backend(); ptok = chat.tokenizer
        } else {
            let chat = try BonsaiChat(ggufPath: arg1)
            backend = chat.backend(); ptok = chat.tokenizer
        }
        let ids = ptok.encode(
            probeWrap(turnArgs.first ?? "What is 2+2? Reply with just the number."),
            addSpecial: true)
        let gen = max(1, capVal ?? 4)
        await backend.useSampler(nil)
        await backend.reset()
        let p0 = Date()
        var next = try await backend.extend(ids)
        let ppSec = Date().timeIntervalSince(p0)
        var out: [Int32] = [next]
        let g0 = Date()
        while out.count < gen && next != ptok.eosId {
            next = try await backend.decode(next); out.append(next)
        }
        let tgSec = Date().timeIntervalSince(g0)
        err(String(format: "[probe] prefill %d tok in %.2fs (%.1f t/s) | "
            + "gen %d tok in %.2fs (%.1f t/s)\n",
            ids.count, ppSec, ppSec > 0 ? Double(ids.count) / ppSec : 0,
            out.count, tgSec, tgSec > 0 ? Double(out.count) / tgSec : 0))
        print("PROBE ids: \(out)")
        print("PROBE txt: \(ptok.decode(out))")
        exit(0)
    }
    // --metal routes the same ChatSession through the GPU MetalEngine; default
    // is the pure-Swift SIMD/CPU engine. Both load everything from the one GGUF
    // and expose the identical AgentBackend seam.
    let useMetal = rawArgs.contains("--metal")
    err("loading Bonsai GGUF \(arg1)...\n")
    let backend: any AgentBackend
    let template: String
    let vocabCount: Int
    let bonsaiPresets: SamplingPresets
    if useMetal {
        let chat = try MetalChat(ggufPath: arg1)
        backend = chat.backend(); template = chat.chatTemplate
        vocabCount = chat.tokenizer.vocabCount
        bonsaiPresets = greedyDecode ? SamplingPresets.greedy
                                     : chat.samplingPresets
        err("ready (Metal/GPU ternary decode; vocab \(vocabCount)).\n")
    } else {
        let chat = try BonsaiChat(ggufPath: arg1)
        backend = chat.backend(); template = chat.chatTemplate
        vocabCount = chat.tokenizer.vocabCount
        bonsaiPresets = greedyDecode ? SamplingPresets.greedy
                                     : chat.samplingPresets
        err("ready (SIMD/CPU ternary decode; vocab \(vocabCount)).\n")
    }
    // Same raw pp512/tg128 protocol as the CoreML --bench below, over the shared
    // AgentBackend seam: prefill benchPrompt once (Metal batches it, SIMD is
    // serial) for pp t/s, then 128 greedy decodes for tg t/s. One warmup first.
    if rawArgs.contains("--bench") {
        try await runBackendBench(backend,
                                  useMetal ? "Metal/GPU" : "SIMD/CPU ")
    }
    let bsession = ChatSession(
        backend: backend, template: template,
        system: systemPrompt, systemTail: systemTimeTail,
        vocabSize: vocabCount, presets: bonsaiPresets,
        enableThinking: enableThinking, reasoningEffort: reasoningEffort,
        maxTokens: maxTokens,
        maxReasoning: maxReasoning, softReasoningCap: softReasoning,
        overthink: overthink, runner: toolRunner)
    if let pkVal {
        let url = URL(fileURLWithPath: pkVal)
        let t0 = Date()
        if await bsession.prime(from: url) {
            err(String(format: "[precook] primed in %.2fs\n",
                       Date().timeIntervalSince(t0)))
        } else {
            try await bsession.precook(to: url)
            err(String(format: "[precook] cooked + saved in %.1fs\n",
                       Date().timeIntervalSince(t0)))
        }
    }
    // "img:PATH question" -> a vision turn (Metal backend with a sibling
    // mmproj; the SIMD backend and mmproj-less sets report no vision).
    func runBonsaiVision(_ path: String, _ text: String) async {
        let grid = await backend.visionGrid()
        let data = try? Data(contentsOf: URL(fileURLWithPath: path))
        if let grid, let data,
           let tiles = try? VisionPreprocess.imageSet(data, tiled: false,
                                                      grid: grid) {
            err("[vl] \(path): \(tiles.count) tile(s) @\(grid.side)px, "
                + "\(grid.mergedTokens) tok/tile\n")
            print("\nUSER: [image] \(text)\nASSISTANT: ", terminator: "")
            fflush(stdout)
            let stream = bsession.replyVision(
                text, tiles: tiles, gridH: grid.gridH, gridW: grid.gridW,
                tokensPerImage: grid.mergedTokens,
                onReasoning: { r in err(r) })
            for await piece in stream {
                print(piece, terminator: ""); fflush(stdout)
            }
            print()
        } else {
            err("no vision: mmproj missing or this backend has no tower\n")
        }
    }

    func runBonsai(_ user: String) async {
        if user.hasPrefix("img:") {
            let body = user.dropFirst("img:".count)
            let cut = body.firstIndex(of: " ") ?? body.endIndex
            let text = cut < body.endIndex
                ? String(body[body.index(after: cut)...])
                : VLPrompt.defaultPrompt
            await runBonsaiVision(String(body[..<cut]), text)
        } else {
            print("\nUSER: \(user)\nASSISTANT: ", terminator: "")
            fflush(stdout)
            for await piece in bsession.reply(user,
                                              onReasoning: { r in err(r) }) {
                print(piece, terminator: ""); fflush(stdout)
            }
            print()
            let m = await bsession.lastMetrics
            err(String(format:
                "[ctx %d | think %d | gen %d | pp %.1f t/s | tg %.1f t/s]\n",
                m.ctx, m.thinkTokens, m.contentTokens, m.pp, m.tg))
        }
    }
    // --title gates conversation-title generation: run a turn, then generate
    // the (greedy, deterministic) title twice. A mismatch means the first gen
    // did NOT restore the KV + recurrent state; the follow-up turn must also
    // stay coherent, proving the mark survived the rollback.
    if rawArgs.contains("--title") {
        let q = turnArgs.first ?? "Explain why the sky is blue, briefly."
        await runBonsai(q)
        let t1 = await bsession.makeTitle()
        let t2 = await bsession.makeTitle()
        err("[title] \"\(t1)\"\n")
        let ok = t1 == t2 && !t1.isEmpty
        err(ok ? "[title] PASS: identical across regeneration "
                + "(state preserved)\n"
               : "[title] FAIL: t1=\"\(t1)\" t2=\"\(t2)\"\n")
        await runBonsai("Now answer in one word.")
        exit(ok ? 0 : 1)
    }
    if !turnArgs.isEmpty {
        for turn in turnArgs { await runBonsai(turn) }
    } else {
        err("enter messages (Ctrl-D to end):\n")
        var line = readLine(strippingNewline: true)
        while let text = line, text != "/quit" {
            if !text.isEmpty { await runBonsai(text) }
            line = readLine(strippingNewline: true)
        }
    }
    if rawArgs.contains("--hint") { await reportHints(bsession) }
    exit(0)
}

// The raw pp/tg protocol, apples-to-apples with `llama-bench -p N -n 128`:
// prefill benchPrompt once for pp, then `gen` greedy decodes for tg, after a
// warmup. Shared so every GGUF lineage is measured the same way rather than
// each mode growing its own timer.
@MainActor func runBackendBench(_ backend: any AgentBackend,
                                _ label: String) async throws {
    var ids = backend.encode(benchPrompt)
    // --ctx N repeats the prompt to N tokens. Repetition is fine here: the
    // cost of a decode step depends on how MANY positions are cached, not on
    // what they say, and greedy output is discarded either way.
    if let benchCtxVal {
        var padded = ids
        while padded.count < benchCtxVal { padded += ids }
        ids = Array(padded.prefix(benchCtxVal))
    }
    let gen = capVal ?? 128
    await backend.useSampler(nil)
    await backend.reset()
    var warm = try await backend.extend(ids)
    for _ in 0 ..< 8 { warm = try await backend.decode(warm) }
    await backend.reset()
    let p0 = Date()
    var next = try await backend.extend(ids)
    let ppSec = Date().timeIntervalSince(p0)
    let g0 = Date()
    for _ in 0 ..< gen { next = try await backend.decode(next) }
    let tgSec = Date().timeIntervalSince(g0)
    print(String(format: "%@  pp%d %.1f t/s  |  tg%d %.1f t/s", label,
                 ids.count, Double(ids.count) / ppSec, gen,
                 Double(gen) / tgSec))
    exit(0)
}

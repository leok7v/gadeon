import Foundation
import LLM

// gemma-4 drive modes, dispatched from runGgufMain when the file's own
// `general.architecture` says gemma4. Three of them:
//
//   --gemma-gate DIR   replay the EXACT token ids the Phase 0 reference dump
//                      was taken at (DIR/manifest.json) and write this
//                      engine's hidden states + logits beside them, for
//                      scripts/convert/gemma4gguf/gate_swift.py to score.
//   --gemma-tok        encode each turn arg and print ids + round-trip.
//   (default)          encode a prompt through the file's own chat template
//                      and greedily decode, stopping on the eos SET.
//
// The gate is the point: free-running text proves nothing, and every bug this
// architecture can hide (the rope pairing, the sandwich norms, the scale-free
// V, the shared histories) shows up as a diverging hidden state long before
// it shows up as a bad word.

// The turn args with every gemma flag's VALUE removed. main.swift drops bare
// `--flags` but cannot know which of the words after them were consumed as an
// argument, so a mode taking BOTH a value and a question has to say which
// words it already ate -- otherwise the file path arrives as the question.
private func gemmaTurns(_ args: [String], _ turns: [String]) -> [String] {
    let flags = ["--gemma-gate", "--gemma-metal-gate", "--gemma-vit-gate",
                 "--gemma-audio-gate", "--gemma-mel-gate",
                 "--gemma-audio-say", "--seconds", "--offset",
                 "--dump-proj", "--gemma-patch-gate", "--image",
                 "--gemma-image-say", "--with-audio",
                 "--gemma-video-say", "--clip", "--film", "--attach-turn",
                 "--dump-frames",
                 "--image-budget", "--system", "--gemma-batch-gate",
                 "--gemma-mic", "--gate", "--gemma-unified-gate",
                 "--gemma-image-gate"]
    let eaten = Set(flags.compactMap { flag in valueAfter(args, flag) })
    return turns.filter { turn in !eaten.contains(turn) }
}

// An operator-facing message; the top level prints it and stops.
enum GemmaCLIError: Error, CustomStringConvertible {
    case msg(String)

    var description: String {
        let out: String
        switch self {
        case .msg(let s): out = s
        }
        return out
    }
}

@MainActor func runGemmaMain(_ path: String, _ args: [String],
                             _ rawTurns: [String], _ cap: Int?) async throws {
    let turns = gemmaTurns(args, rawTurns)
    let t0 = Date()
    let chat = try GemmaChat(ggufPath: path)
    let c = chat.model.cfg
    err("gemma4: \(c.nLayer) layers, \(c.nEmbd) wide, \(c.nVocab) vocab, "
        + "\(c.fullLayers) full / \(c.slidingLayers) sliding, loaded in "
        + String(format: "%.1fs\n", Date().timeIntervalSince(t0)))
    err("tokenizer: \(chat.vocabCount) tokens, "
        + "eos \(chat.eosIds.sorted())\n")

    if let dir = valueAfter(args, "--gemma-gate") {
        try gemmaGate(chat, dir)
        exit(0)
    }
    if let dir = valueAfter(args, "--gemma-metal-gate") {
        try gemmaMetalGate(chat, dir)
        exit(0)
    }
    if let dir = valueAfter(args, "--gemma-vit-gate") {
        try gemmaViTGate(chat, dir)
        exit(0)
    }
    if let dir = valueAfter(args, "--gemma-audio-gate") {
        try gemmaAudioGate(chat, dir)
        exit(0)
    }
    if args.contains("--gemma-park-gate") {
        try await gemmaParkGate(chat, args.contains("--metal"))
        exit(0)
    }
    if let dir = valueAfter(args, "--gemma-image-gate") {
        try gemmaImageGate(chat, dir, args.contains("--ref-embeds"),
                           args.contains("--metal"))
        exit(0)
    }
    if let dir = valueAfter(args, "--gemma-unified-gate") {
        try gemmaUnifiedGate(chat, dir,
                             valueAfter(args, "--image")
                                 ?? "tests/vl/ad-rexall.jpg",
                             valueAfter(args, "--clip") ?? "")
        exit(0)
    }
    // A release build strips precondition text, so asking a file for a tower
    // it does not have would abort on a missing tensor with nothing said.
    // Name the reason before any of those paths is entered.
    // Only the paths that genuinely need a TOWER. Image, video and audio
    // turns all work on a unified checkpoint through its encoder-free
    // embedder; what cannot work is a gate that scores a tower's own
    // intermediates against a dump, since there are no intermediates.
    let wantsVision = ["--gemma-patch-gate", "--gemma-vit-gate"]
    let wantsAudio = ["--gemma-mel-gate", "--gemma-audio-gate"]
    if !chat.model.hasVisionTower
        && wantsVision.contains(where: { f in args.contains(f) }) {
        err("this checkpoint carries no vision tower: it embeds images "
            + "with no encoder behind them, which is not wired yet\n")
        exit(2)
    }
    if !chat.model.hasAudioTower
        && wantsAudio.contains(where: { f in args.contains(f) }) {
        err("this checkpoint carries no audio tower: it embeds audio with "
            + "no encoder behind it, which is not wired yet\n")
        exit(2)
    }
    if let file = valueAfter(args, "--gemma-video-say") {
        try await gemmaVideoSay(chat, file, turns.first, cap,
                                metal: args.contains("--metal"),
                                withAudio: args.contains("--with-audio"),
                                dumpFrames: valueAfter(args, "--dump-frames"))
        exit(0)
    }
    if let file = valueAfter(args, "--gemma-image-say") {
        try await gemmaImageSay(chat, file, turns.first, cap,
                                metal: args.contains("--metal"),
                                clip: valueAfter(args, "--with-audio"),
                                seconds: valueAfter(args, "--seconds")
                                    .flatMap { s in Double(s) })
        exit(0)
    }
    if let dir = valueAfter(args, "--gemma-patch-gate") {
        try gemmaPatchGate(chat, dir, valueAfter(args, "--image")
            ?? "tests/vl/ad-rexall.jpg")
        exit(0)
    }
    if let dir = valueAfter(args, "--gemma-mel-gate") {
        try gemmaMelGate(dir)
        exit(0)
    }
    if let file = valueAfter(args, "--gemma-audio-say") {
        try await gemmaAudioSay(chat, file, turns.first, cap,
                                seconds: valueAfter(args, "--seconds")
                                    .flatMap { s in Double(s) },
                                offset: valueAfter(args, "--offset")
                                    .flatMap { s in Double(s) } ?? 0,
                                metal: args.contains("--metal"),
                                dump: valueAfter(args, "--dump-proj"),
                                gated: args.contains("--gate"))
        exit(0)
    }
    if args.contains("--gemma-mic") {
        try await gemmaMic(chat, turns.first, cap,
                           metal: args.contains("--metal"),
                           seconds: valueAfter(args, "--seconds")
                               .flatMap { s in Double(s) })
        exit(0)
    }
    if let dir = valueAfter(args, "--gemma-batch-gate") {
        try gemmaBatchGate(chat, dir)
        exit(0)
    }
    if args.contains("--bench") {
        let metal = args.contains("--metal")
        try await runBackendBench(
            metal ? try chat.metalBackend() : chat.backend(),
            metal ? "Metal/GPU" : "SIMD/CPU ")
    }
    if args.contains("--gemma-tok") {
        for s in turns {
            let ids = chat.encode(s)
            print("\(ids.count) ids  \(ids)")
            print("  round-trip \(chat.decode(ids).debugDescription)")
        }
        exit(0)
    }
    // --gemma-chat drives the REAL turn loop (ChatSession over the gemma
    // AgentBackend), so the derived wire, the reasoning channel and the tool
    // round are exercised against the actual tokenizer rather than a fixture.
    if args.contains("--gemma-chat") {
        // The backend comes first because the attachments' towers bind
        // through the engine's context; on the SIMD arm there is none and
        // they run on the CPU.
        let backend: any AgentBackend
        var ctx: MetalContext? = nil
        if args.contains("--metal") {
            let gpu = try chat.metalBackend()
            ctx = gpu.ctx
            backend = gpu
        } else {
            backend = chat.backend()
        }
        try await gemmaChat(chat, backend, turns, cap,
                            thinking: args.contains("--think"),
                            system: valueAfter(args, "--system"),
                            attach: try await gemmaAttachments(chat, args,
                                                               ctx))
        exit(0)
    }
    let question = turns.first ?? "What is 2+2? Reply with just the number."
    var prompt = question
    if !chat.chatTemplate.isEmpty {
        prompt = (try? renderPrompt(
            template: chat.chatTemplate,
            messages: [AgentMessage(role: "user", content: question)],
            tools: [], addGenerationPrompt: true,
            enableThinking: false, bosToken: chat.bosToken)) ?? question
    }
    let ids = chat.encode(prompt)
    // --metal runs the SAME weights through the GPU engine; the two paths are
    // each other's oracle, so a divergence here is a port bug rather than a
    // model one.
    let backend: any AgentBackend = args.contains("--metal")
        ? try chat.metalBackend() : chat.backend()
    err("[probe] prompt \(ids.count) tokens on "
        + (args.contains("--metal") ? "Metal/GPU\n" : "SIMD/CPU\n"))
    let p0 = Date()
    var next = try await backend.extend(ids)
    let pp = Date().timeIntervalSince(p0)
    var out: [Int32] = []
    let g0 = Date()
    while out.count < (cap ?? 32) && !chat.eosIds.contains(next) {
        out.append(next)
        next = try await backend.decode(next)
    }
    let tg = Date().timeIntervalSince(g0)
    err(String(format: "[probe] prefill %d tok in %.1fs (%.2f t/s) | "
        + "gen %d tok in %.1fs (%.2f t/s)\n", ids.count, pp,
        pp > 0 ? Double(ids.count) / pp : 0, out.count, tg,
        tg > 0 ? Double(out.count) / tg : 0))
    print("PROBE ids: \(out)")
    print("PROBE txt: \(chat.decode(out))")
    exit(0)
}

// The turn's attachments, each encoded once before the conversation starts.
private struct GemmaAttachments {
    var parts: [ContentPart] = []
    var spans: [SoftSpan] = []
    // Which turn carries them. Turn 0 seeds fresh; a later one takes the
    // rewind-and-append path instead, so both halves of the continuation are
    // reachable from the command line.
    var turn = 0
}

// --image / --clip / --film, in that order, so a turn can carry several
// modalities at once. Decoding is here and the markup is the module's, which
// is what keeps this probe and the app on one definition.
@MainActor private func gemmaAttachments(_ chat: GemmaChat,
                                         _ args: [String],
                                         _ ctx: MetalContext?) async throws
    -> GemmaAttachments {
    var out = GemmaAttachments()
    out.turn = valueAfter(args, "--attach-turn").flatMap { s in Int(s) } ?? 0
    let media = chat.media(ctx: ctx)
    let clipSeconds = valueAfter(args, "--seconds").flatMap { s in Double(s) }
    for file in (valueAfter(args, "--image") ?? "")
        .split(separator: ",").map(String.init) {
        let span = try media.image(
            try Data(contentsOf: URL(fileURLWithPath: file)),
            softTokens: valueAfter(args, "--image-budget")
                .flatMap { v in Int(v) })
        err("[chat] \(file) -> \(span.rows) image soft tokens\n")
        out.parts.append(.image)
        out.spans.append(span)
    }
    if let file = valueAfter(args, "--clip") {
        let pcm = try await AudioFile.samples(
            url: URL(fileURLWithPath: file),
            sampleRate: Double(chat.melConfig.sampleRate),
            seconds: clipSeconds)
        for span in try media.audio(pcm) {
            err("[chat] \(file) -> \(span.rows) audio soft tokens\n")
            out.parts.append(.audio)
            out.spans.append(span)
        }
    }
    if let file = valueAfter(args, "--film") {
        let shot = try await VideoFrames.sample(
            url: URL(fileURLWithPath: file),
            count: try chat.videoWire().frames)
        let span = try media.video(frames: shot.images, seconds: shot.seconds)
        err("[chat] \(file) -> \(shot.images.count) frames, \(span.rows) "
            + "video soft tokens\n")
        out.parts.append(.video)
        out.spans.append(span)
    }
    return out
}

// One or more turns through the real ChatSession loop, printing the content
// and reasoning channels separately -- the two the wire has to keep apart.
//
// Attachments ride the FIRST turn and the rest are plain text, which is what
// makes this the multi-turn attachment gate: turn two has to still see what
// turn one attached, without the tower running again.
@MainActor private func gemmaChat(_ chat: GemmaChat,
                                  _ backend: any AgentBackend,
                                  _ turns: [String],
                                  _ cap: Int?, thinking: Bool,
                                  system: String?,
                                  attach: GemmaAttachments) async throws {
    let session = ChatSession(
        backend: backend,
        template: chat.chatTemplate,
        system: system ?? "You are a helpful assistant.",
        vocabSize: chat.vocabCount, presets: chat.samplingPresets,
        enableThinking: thinking, maxTokens: cap ?? 128)
    err("[chat] thinking \(thinking), template reasons "
        + "\(templateSupportsThinking(chat.chatTemplate)), soft tokens "
        + "\(await session.supportsSoftTokens())\n")
    let asked = turns.isEmpty
        ? ["What is 2+2? Reply with just the number."] : turns
    for (i, turn) in asked.enumerated() {
        print("USER: \(turn)")
        let t0 = Date()
        var answer = ""
        // The attachments lead and the question follows, which is the
        // ordinary shape; a caller wanting A-inside-B interleaves the text
        // parts itself, which replySoft now takes verbatim.
        let stream = i == attach.turn && !attach.spans.isEmpty
            ? session.replySoft(turn,
                                parts: Gemma4Media.ordered(
                                    attach.parts, around: .text(turn)),
                                spans: attach.spans,
                                onReasoning: { r in err(r) })
            : session.reply(turn, onReasoning: { r in err(r) })
        for await piece in stream { answer += piece }
        let m = await session.lastMetrics
        print("MODEL: \(answer)")
        err(String(format: "[chat] %@ | ctx %d, think %d, content %d, "
            + "%.2f t/s, %.0fs\n", m.endReason, m.ctx, m.thinkTokens,
            m.contentTokens, m.tg, Date().timeIntervalSince(t0)))
    }
}

// Metal against SIMD over the reference dump's own ids. The SIMD engine is
// already gated against the Python oracle, so agreeing with it token for
// token is what proves the GPU port -- and it costs no Python.
@MainActor private func gemmaMetalGate(_ chat: GemmaChat,
                                       _ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let raw = try Data(
        contentsOf: base.appendingPathComponent("manifest.json"))
    let man = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    let text = man["text"] as! [String: Any]
    let ids = (text["input_ids"] as! [Int]).map { id in Int32(id) }
    err("[metal-gate] replaying \(ids.count) reference tokens\n")
    let gpu = try Gemma4MetalEngine(chat.model)
    let cpu = chat.engine
    cpu.reset()
    gpu.reset()
    var worstHidden = 1.0
    var worstLogits = 1.0
    var argmaxHits = 0
    var lastCpu: Int32 = -1
    var lastGpu: Int32 = -1
    for (i, id) in ids.enumerated() {
        let h = cpu.forward(token: Int(id), pos: i)
        gpu.forward(token: Int(id), pos: i)
        worstHidden = min(worstHidden, cosine(h, gpu.hidden()))
        let lc = cpu.logits(h)
        let lg = gpu.logits()
        worstLogits = min(worstLogits, cosine(lc, lg))
        lastCpu = Int32(cpu.argmax(lc))
        lastGpu = Int32(gpu.argmax(lg))
        if lastCpu == lastGpu { argmaxHits += 1 }
        err(String(format: "\r  %d/%d", i + 1, ids.count))
    }
    err("\n")
    // TWO MODES, because with the clamp on nothing here can make a structural
    // assertion. `round()` turns any arithmetic-ordering difference (Metal's
    // simdgroup reductions against vDSP's) into a discrete one-unit jump, and
    // the reference cannot reproduce ITSELF better than 0.9585 across dtypes,
    // so a tight bar would ask the two engines to agree more closely than the
    // model agrees with itself -- and a loose one is wide enough to hide a
    // wrong window start or a bad page slot. With LLM_SRQ=0 the clamp is the
    // identity and the two f32 paths must agree to fp rounding, which is where
    // an indexing defect has nowhere left to hide. Measured either way:
    // per-token argmax 32/32 and cos 1.0 with SRQ off, 28-30/32 and cos 0.96
    // with it on, on both E2B and E4B. So SRQ-on decides on the GENERATION
    // POINT -- the one token the turn actually emits -- and the counts stay on
    // screen as diagnostics. See gemmaBatchGate, which draws the same line.
    let clamped = ProcessInfo.processInfo.environment["LLM_SRQ"] != "0"
    let floor = clamped ? 0.95 : 0.999
    let pass = worstHidden >= floor && worstLogits >= floor
        && lastCpu == lastGpu
        && (clamped || argmaxHits == ids.count)
    err(clamped
        ? "[metal-gate] SRQ ON: accuracy mode, the generation point decides. "
          + "Re-run with LLM_SRQ=0 for the structural assertion.\n"
        : "[metal-gate] SRQ OFF: structural mode, the two engines must agree "
          + "to fp rounding on every token.\n")
    print(String(format: "worst hidden cosine  %.6f", worstHidden))
    print(String(format: "worst logits cosine  %.6f", worstLogits))
    print("argmax agreement     \(argmaxHits)/\(ids.count)")
    print("generation point     "
        + (lastCpu == lastGpu ? "MATCH (\(lastCpu))"
                              : "DIFFER cpu=\(lastCpu) gpu=\(lastGpu)"))
    print(pass ? "GATE PASS" : "GATE FAIL")
}

// The log-mel frontend against the reference dump's own input_features,
// computed from its own waveform. Pure DSP, so this gates before any tower
// weight is touched.
@MainActor private func gemmaMelGate(_ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let wav = try npy(base.appendingPathComponent("audio.waveform.npy"))
    let want = try npy(base.appendingPathComponent(
        "audio.input_features.npy"))
    let mel = Gemma4Mel(.processorDefault)
    let got = mel.features(wav.values)
    let frames = want.shape.first ?? -1
    err("[mel-gate] \(wav.values.count) samples -> \(got.frames) frames, "
        + "reference \(want.shape)\n")
    let c = cosine(want.values, got.values)
    print("frames               \(got.frames) "
        + (got.frames == frames ? "MATCH" : "vs reference \(frames)"))
    print(String(format: "features cosine      %.6f", c))
    print(got.frames == frames && c >= 0.999 ? "GATE PASS" : "GATE FAIL")
}

// The audio tower against the SRQ-off oracle, fed the dump's own mel
// features so the frontend and the tower are gated separately.
@MainActor private func gemmaAudioGate(_ chat: GemmaChat,
                                       _ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let feats = try npy(base.appendingPathComponent(
        "audio.input_features.npy"))
    let want = try npy(base.appendingPathComponent("audio.tower.npy"))
    let wantProj = try npy(base.appendingPathComponent("audio.proj.npy"))
    let frames = feats.shape.first ?? 0
    let bins = feats.shape.count > 1 ? feats.shape[1] : 0
    err("[audio-gate] \(frames) mel frames x \(bins) -> reference "
        + "\(want.shape) tower\n")
    let tower = try Gemma4Audio(chat.model)
    let t0 = Date()
    var taps: [String: [Float]] = [:]
    let got = tower.forward(mel: feats.values, frames: frames, bins: bins) {
        name, v in taps[name] = v
    }
    for name in ["sub", "s_ff1", "s_q", "s_k", "s_v", "s_relk",
                 "s_attn", "s_lconv", "s_ff2"]
        + (0..<12).map({ i in "l\(i)" }) {
        let f32f = base.appendingPathComponent("audio.f32.\(name).npy")
        if let ref32 = try? npy(f32f), let mine = taps[name] {
            print(String(format: "  %@ vs f32  %.6f", name,
                         cosine(ref32.values, mine)))
        }
        let f = base.appendingPathComponent("audio.dbg.\(name).npy")
        if let ref = try? npy(f), let mine = taps[name] {
            let c = cosine(ref.values, mine)
            print(String(format: "  %@ cosine  %.6f", name, c))
        }
    }
    let secs = Date().timeIntervalSince(t0)
    let cTower = cosine(want.values, got.tower)
    let cProj = cosine(wantProj.values, got.proj)
    let countOK = got.count == (want.shape.first ?? -1)
    print("soft tokens          \(got.count) "
        + (countOK ? "MATCH" : "vs reference \(want.shape.first ?? -1)"))
    print(String(format: "tower cosine         %.6f", cTower))
    print(String(format: "proj  cosine         %.6f", cProj))
    print(String(format: "forward              %.1fs", secs))
    // Seed layers 10-11 with the REFERENCE's layer-9 output: if they then
    // reproduce the reference's layer 11, the arithmetic is right and what
    // diverged was the input those layers were handed.
    // This tower's last two layers amplify any input difference about 175x,
    // so the bf16 reference cannot express its own answer: run the SAME
    // weights in f32 and the two references differ by 0.9947. The f32 dump
    // is therefore the oracle, and the bf16 number stays on screen as the
    // measure of that gap rather than of the port.
    var cTower32 = cTower
    var cProj32 = cProj
    if let t32 = try? npy(base.appendingPathComponent("audio.tower.f32.npy")) {
        cTower32 = cosine(t32.values, got.tower)
        print(String(format: "tower vs f32 ref     %.6f", cTower32))
    }
    if let p32 = try? npy(base.appendingPathComponent("audio.proj.f32.npy")) {
        cProj32 = cosine(p32.values, got.proj)
        print(String(format: "proj  vs f32 ref     %.6f", cProj32))
    }
    let l9 = try? npy(base.appendingPathComponent("audio.dbg.l9.npy"))
    let l11 = try? npy(base.appendingPathComponent("audio.dbg.l11.npy"))
    if let l9, let l11 {
        let re = tower.runLayers(l9.values, got.count, from: 10, through: 11)
        print(String(format: "reseeded 10-11       %.6f",
                     cosine(l11.values, re)))
    }
    // The tower binds through the TEXT engine's context, here as in the app.
    // The engine is not wanted past the handover and the tower holds the
    // context alive; the text tensors it never reads cost nothing, being
    // mapped rather than resident.
    let gpu = try Gemma4MetalAudio(chat.model,
                                   ctx: Gemma4MetalEngine(chat.model).ctx)
    let g0 = Date()
    let mg = gpu.forward(mel: feats.values, frames: frames, bins: bins)
    let gpuSecs = Date().timeIntervalSince(g0)
    let mTower = cosine(got.tower, mg.tower)
    let mProj = cosine(got.proj, mg.proj)
    print(String(format: "metal tower vs simd  %.6f", mTower))
    print(String(format: "metal proj  vs simd  %.6f", mProj))
    print(String(format: "forward simd %.1fs | metal %.2fs", secs, gpuSecs))
    // EVERY pass criterion here is EXACT -- soft-token counts, the rebuilt
    // sequence length, and which token each tower drives the model to predict
    // at the generation point. The cosines above stay on screen as
    // diagnostics, because under SRQ a cosine measures how badly the bf16
    // reference reproduces ITSELF (0.9585 across dtypes on a tower whose last
    // two layers amplify ~175x), so any threshold on one would be a number
    // picked to pass rather than a property of the port.
    let simdSpoke = try gemmaAudioArgmax(chat, base, "simd", got)
    let metalSpoke = try gemmaAudioArgmax(chat, base, "metal", mg)
    print(countOK && mg.count == got.count && simdSpoke && metalSpoke
          ? "GATE PASS" : "GATE FAIL")
}

// The tower cosines above are DIAGNOSTICS, not the verdict. Under SRQ a
// cosine measures the reference's own irreproducibility rather than the port
// -- round() turns any arithmetic-ordering difference into a discrete jump,
// and the checkpoint cannot reproduce ITSELF better than 0.9585 across
// dtypes. What survives is WHICH TOKEN each position predicts, the same call
// the metal gate already makes, and the dump carries the LM logits for its
// own audio turn to score it against.
//
// The turn is rebuilt from the manifest's own prompt and the TOWER's own soft
// token count, and the token TOTAL is the self-check: hitting the dump's
// n_tokens means the sequence was reproduced, and a mismatch says so rather
// than quietly scoring a different sentence.

@MainActor private func gemmaAudioArgmax(
    _ chat: GemmaChat, _ base: URL, _ label: String,
    _ got: (tower: [Float], proj: [Float], count: Int)) throws -> Bool {
    let raw = try Data(
        contentsOf: base.appendingPathComponent("manifest.json"))
    let man = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    let audio = man["audio"] as? [String: Any] ?? [:]
    let prompt = audio["prompt"] as? String ?? ""
    let want = audio["n_tokens"] as? Int ?? -1
    let ref = try npy(base.appendingPathComponent("audio.logits.npy"))
    let wire = try chat.audioWire()
    let span = SoftSpan.bracketed(begin: wire.boa, placeholder: wire.token,
                                  end: wire.eoa, count: got.count,
                                  features: got.proj)
    let ids = try softTurnIds(chat, prompt, [.audio],
                              [wire.token: span.ids])
    var hits = 0
    var textHits = 0
    var textPositions = 0
    var generationPoint = false
    let vocab = chat.model.cfg.nVocab
    if ids.count == want {
        let eng = chat.engine
        eng.reset()
        let feed = SoftFeed([span])
        for (i, id) in ids.enumerated() {
            let hidden = feed.row(id).map { f in
                eng.forward(embedding: f, pos: i)
            } ?? eng.forward(token: Int(id), pos: i)
            let mine = eng.argmax(eng.logits(hidden))
            var best = 0
            var bv = -Float.greatestFiniteMagnitude
            for v in 0..<vocab where ref.values[i * vocab + v] > bv {
                bv = ref.values[i * vocab + v]
                best = v
            }
            let same = mine == best
            if same { hits += 1 }
            // A position INSIDE the audio span is the model predicting a
            // continuation of a waveform it was never trained to continue --
            // near-tied and meaningless. What the turn is actually for is the
            // token predicted at the generation point.
            if id != wire.token {
                textPositions += 1
                if same { textHits += 1 }
            }
            if i == ids.count - 1 { generationPoint = same }
        }
    }
    print("\(label) turn          \(ids.count) tokens vs reference \(want)"
        + (ids.count == want ? " MATCH" : " -- NOT SCORED"))
    print("\(label) argmax        \(hits)/\(max(want, 0)) overall, "
        + "\(textHits)/\(textPositions) outside the audio span")
    print("\(label) gen point     "
        + (generationPoint ? "MATCH" : "DIFFER"))
    return ids.count == want && generationPoint
}

// A clip through the whole audio path and out as text: decode -> mel ->
// tower -> project -> splice the soft tokens into the turn -> decode.
//
// The soft tokens are spliced at the point the template's user content
// begins, found by rendering the SAME template with an empty message and
// taking the common prefix. That keeps the insertion point a property of the
// template rather than of a hardcoded marker string.
@MainActor private func gemmaAudioSay(_ chat: GemmaChat, _ file: String,
                                      _ ask: String?, _ cap: Int?,
                                      seconds: Double?, offset: Double,
                                      metal: Bool, dump: String?,
                                      gated: Bool)
    async throws {
    let wire = try chat.audioWire()
    let mel = Gemma4Mel(chat.melConfig)
    let url = URL(fileURLWithPath: file)
    let t0 = Date()
    let pcm = try await AudioFile.samples(
        url: url, sampleRate: Double(mel.cfg.sampleRate),
        offset: offset, seconds: seconds)
    let secs = Double(pcm.count) / Double(mel.cfg.sampleRate)
    err(String(format: "[say] %@ -> %.2fs at %d Hz (%.1fs decode)\n",
               url.lastPathComponent, secs, mel.cfg.sampleRate,
               Date().timeIntervalSince(t0)))
    if let path = dump {
        try Data(bytes: pcm, count: pcm.count * 4)
            .write(to: URL(fileURLWithPath: path + ".pcm"))
        err("[say] wrote \(pcm.count) f32 samples to \(path).pcm\n")
    }
    // The engine leads on the GPU arm: it owns the mapping the tower binds
    // through, and it is the same engine the turn is prefilled on below.
    var gpu: Gemma4MetalEngine? = nil
    if metal { gpu = try Gemma4MetalEngine(chat.model) }
    let tower: ([Float], Int, Int) -> (tower: [Float], proj: [Float],
                                       count: Int)
    if !chat.model.hasAudioTower {
        // No mel and no conformer: a frame of raw samples IS a token, so the
        // "features" this is handed below are the samples themselves.
        let um = try Gemma4UnifiedMedia(chat.model)
        tower = { samples, _, _ in
            let rows = um.audio(samples)
            return (tower: [], proj: rows.flatMap { r in r },
                    count: rows.count)
        }
    } else if let gpu {
        let gt = try Gemma4MetalAudio(chat.model, ctx: gpu.ctx)
        tower = { m, f, b in gt.forward(mel: m, frames: f, bins: b) }
    } else {
        let ct = try Gemma4Audio(chat.model)
        tower = { m, f, b in ct.forward(mel: m, frames: f, bins: b) }
    }
    // --gate runs the file through the MIC's detector instead of the file
    // splitter, which is what makes the microphone path testable without a
    // microphone: synthesise a dialog (kittens-tts-cli), feed it here, and
    // the utterances are the ones a speaker would have produced. Silence is
    // DROPPED on this arm, where the file arm keeps every second.
    let rate = Double(mel.cfg.sampleRate)
    let chunks: [(range: Range<Int>, samples: [Float], hardCut: Bool)]
    if gated {
        let gate = SpeechGate(rate: rate, maxSeconds: wire.maxSeconds)
        var said = gate.push(pcm)
        said.append(contentsOf: gate.finish())
        err(String(format: "[say] gate bar %.5f\n", gate.speechThreshold))
        chunks = said.map { u in
            let at = Int(u.startSeconds * rate)
            return (at ..< (at + u.samples.count), u.samples, u.hardCut)
        }
    } else {
        chunks = AudioChunks.split(pcm, rate: rate,
                                   maxSeconds: wire.maxSeconds)
            .map { c in (c.range, Array(pcm[c.range]), c.hardCut) }
    }
    var spans: [SoftSpan] = []
    var parts: [ContentPart] = []
    for (i, chunk) in chunks.enumerated() {
        let m0 = Date()
        let got: (tower: [Float], proj: [Float], count: Int)
        if chat.model.hasAudioTower {
            let feats = mel.features(chunk.samples)
            got = tower(feats.values, feats.frames, mel.cfg.bins)
        } else {
            got = tower(chunk.samples, 0, 0)
        }
        err(String(format: "[say] %@ %d %.1f-%.1fs -> %d soft tokens "
                   + "(%.1fs)%@\n", gated ? "utterance" : "chunk", i + 1,
                   Double(chunk.range.lowerBound) / rate,
                   Double(chunk.range.upperBound) / rate,
                   got.count, Date().timeIntervalSince(m0),
                   chunk.hardCut ? "  HARD CUT (no pause under the cap)" : ""))
        spans.append(SoftSpan.bracketed(
            begin: wire.boa, placeholder: wire.token, end: wire.eoa,
            count: got.count, features: got.proj))
        parts.append(.audio)
    }
    let question = ask ?? "Transcribe this audio."
    let ids = try softTurnIds(chat, question, parts, spans)
    let soft = spans.reduce(0) { sum, s in sum + s.rows }
    err("[say] ids \(ids.count) = \(ids.count - soft) text + \(soft) soft "
        + "across \(spans.count) chunk(s)\n")
    // Both engines carry the same soft-token entry point; binding the two
    // calls here keeps the decode loop written once.
    let prefill: ([Int32], @escaping (Int32) -> [Float]?) -> Int32
    let step: (Int32) -> Int32
    if let gpu {
        gpu.reset()
        prefill = { i, fn in gpu.extend(i, softAt: fn) }
        step = { t in gpu.decode(t) }
    } else {
        let cpu = chat.engine
        cpu.reset()
        prefill = { i, fn in cpu.extend(i, softAt: fn) }
        step = { t in cpu.decode(t) }
    }
    let p0 = Date()
    let feed = SoftFeed(spans)
    var next = prefill(ids) { id in feed.row(id) }
    err(String(format: "[say] prefill %d tokens (%d soft) in %.1fs on %@\n",
               ids.count, soft, Date().timeIntervalSince(p0),
               metal ? "Metal" : "SIMD"))
    var out: [Int32] = []
    while out.count < (cap ?? 128) && !chat.eosIds.contains(next) {
        out.append(next)
        next = step(next)
    }
    print("ASKED: \(question)")
    print("MODEL: \(chat.decode(out))")
}

// Listen, keep only what was SAID, and answer once the mic stops.
//
// The gate is the point: the model charges 25 soft tokens for every second of
// audio whatever is in it, so a session that recorded the clock would spend
// most of its context on room tone. SpeechGate hands back utterances and the
// silence between them never becomes tokens. Each utterance is a span, and
// the whole session rides ONE turn -- the shape a 40 s file already proved.
//
// --seconds bounds the listen for an unattended run; without it, Return ends
// it, which is what a person testing a microphone actually does.

@MainActor private func gemmaMic(_ chat: GemmaChat, _ ask: String?,
                                 _ cap: Int?, metal: Bool,
                                 seconds: Double?) async throws {
    let wire = try chat.audioWire()
    let mel = Gemma4Mel(chat.melConfig)
    let rate = Double(mel.cfg.sampleRate)
    guard await Microphone.permission() else {
        throw GemmaCLIError.msg("microphone access was refused")
    }
    let gate = SpeechGate(rate: rate, maxSeconds: wire.maxSeconds)
    let heard = Heard()
    let mic = Microphone(rate: rate)
    try mic.start { block in heard.add(gate.push(block)) }
    err("[mic] listening at \(mel.cfg.sampleRate) Hz -- "
        + (seconds.map { s in String(format: "%.0fs\n", s) }
           ?? "press Return to stop\n"))
    if let seconds {
        try await Task.sleep(for: .seconds(seconds))
    } else {
        _ = readLine()
    }
    mic.stop()
    heard.add(gate.finish())
    let said = heard.take()
    let spoken = said.reduce(0.0) { sum, u in
        sum + Double(u.samples.count) / rate
    }
    err(String(format: "[mic] %d utterance(s), %.1fs of speech kept\n",
               said.count, spoken))
    if said.isEmpty {
        throw GemmaCLIError.msg("nothing was said")
    }
    // The engine leads on the GPU arm: it owns the mapping the tower binds
    // through, and it is the same engine the turn is prefilled on below.
    var gpu: Gemma4MetalEngine? = nil
    if metal { gpu = try Gemma4MetalEngine(chat.model) }
    let tower: ([Float], Int, Int) -> (tower: [Float], proj: [Float],
                                       count: Int)
    if let gpu {
        let gt = try Gemma4MetalAudio(chat.model, ctx: gpu.ctx)
        tower = { m, f, b in gt.forward(mel: m, frames: f, bins: b) }
    } else {
        let ct = try Gemma4Audio(chat.model)
        tower = { m, f, b in ct.forward(mel: m, frames: f, bins: b) }
    }
    var spans: [SoftSpan] = []
    var parts: [ContentPart] = []
    for (i, u) in said.enumerated() {
        let feats = mel.features(u.samples)
        let got = tower(feats.values, feats.frames, mel.cfg.bins)
        err(String(format: "[mic] utterance %d at %.1fs, %.1fs -> %d soft\n",
                   i + 1, u.startSeconds,
                   Double(u.samples.count) / rate, got.count))
        spans.append(SoftSpan.bracketed(
            begin: wire.boa, placeholder: wire.token, end: wire.eoa,
            count: got.count, features: got.proj))
        parts.append(.audio)
    }
    let question = ask ?? "Write out what you hear."
    let ids = try softTurnIds(chat, question, parts, spans)
    let prefill: ([Int32], @escaping (Int32) -> [Float]?) -> Int32
    let step: (Int32) -> Int32
    if let gpu {
        gpu.reset()
        prefill = { i, fn in gpu.extend(i, softAt: fn) }
        step = { t in gpu.decode(t) }
    } else {
        let cpu = chat.engine
        cpu.reset()
        prefill = { i, fn in cpu.extend(i, softAt: fn) }
        step = { t in cpu.decode(t) }
    }
    let feed = SoftFeed(spans)
    var next = prefill(ids) { id in feed.row(id) }
    var out: [Int32] = []
    while out.count < (cap ?? 200) && !chat.eosIds.contains(next) {
        out.append(next)
        next = step(next)
    }
    print("ASKED: \(question)")
    print("MODEL: \(chat.decode(out))")
}

// Utterances arrive on the audio thread and are read on the main one, so the
// handoff takes a lock. Nothing else crosses.
private final class Heard: @unchecked Sendable {
    private let lock = NSLock()
    private var said: [SpeechGate.Utterance] = []

    func add(_ more: [SpeechGate.Utterance]) {
        lock.lock()
        said.append(contentsOf: more)
        lock.unlock()
    }

    func take() -> [SpeechGate.Utterance] {
        lock.lock()
        defer { lock.unlock() }
        return said
    }
}

// One attachment's span: begin + placeholder * count + end -- the shape
// `replace_audio_token` / `replace_video_token` build.
private func span(_ begin: Int32, _ token: Int32, _ end: Int32,
                  _ count: Int) -> [Int32] {
    var out = [begin]
    out.append(contentsOf: [Int32](repeating: token, count: count))
    out.append(end)
    return out
}

// The turn's ids, with each modality marker the TEMPLATE emitted replaced by
// its span. Mirrors HF: the chat template writes one placeholder per
// attachment (`<|image|>`, `<|audio|>`, `<|video|>`) and the PROCESSOR
// expands it -- so jinja decides where a modality sits in the turn, and only
// the expansion lives here.
// One clip to its soft tokens, whichever frontend this checkpoint has. A
// tower reads a mel spectrogram; an encoder-free checkpoint reads the raw
// frames, so the mel is not merely unnecessary there, it is the wrong input.
@MainActor private func gemmaHear(_ chat: GemmaChat, _ pcm: [Float])
    throws -> (proj: [Float], count: Int) {
    let out: (proj: [Float], count: Int)
    if chat.model.hasAudioTower {
        let mel = Gemma4Mel(chat.melConfig)
        let feats = mel.features(pcm)
        let a = try Gemma4Audio(chat.model).forward(
            mel: feats.values, frames: feats.frames, bins: mel.cfg.bins)
        out = (a.proj, a.count)
    } else {
        let rows = try Gemma4UnifiedMedia(chat.model).audio(pcm)
        out = (rows.flatMap { r in r }, rows.count)
    }
    return out
}

@MainActor private func softTurnIds(_ chat: GemmaChat, _ question: String,
                                    _ parts: [ContentPart],
                                    _ expand: [Int32: [Int32]])
    throws -> [Int32] {
    let prompt = try renderPrompt(
        template: chat.chatTemplate,
        messages: [AgentMessage(role: "user", content: question,
                                contentParts: Gemma4Media.ordered(
                                    parts, around: .text(question)))],
        tools: [], addGenerationPrompt: true, enableThinking: false,
        bosToken: chat.bosToken)
    var ids: [Int32] = []
    for id in chat.encode(prompt) {
        if let replacement = expand[id] {
            ids.append(contentsOf: replacement)
        } else {
            ids.append(id)
        }
    }
    return ids
}

// The same turn built from SPANS rather than one block per marker: a clip cut
// into several pieces emits one <|audio|> per piece, and each takes the next
// span's ids. Continuation.expandSpans is what ChatSession uses for the same
// job, so the CLI and the app expand a multi-chunk turn identically.

@MainActor private func softTurnIds(_ chat: GemmaChat, _ question: String,
                                    _ parts: [ContentPart],
                                    _ spans: [SoftSpan]) throws -> [Int32] {
    let prompt = try renderPrompt(
        template: chat.chatTemplate,
        messages: [AgentMessage(role: "user", content: question,
                                contentParts: Gemma4Media.ordered(
                                    parts, around: .text(question)))],
        tools: [], addGenerationPrompt: true, enableThinking: false,
        bosToken: chat.bosToken)
    return Continuation.expandSpans(chat.encode(prompt), spans)
}

// An image through the whole vision path and out as text: patchify -> tower
// -> project -> splice the soft tokens into the turn -> decode. The count
// comes from the TOWER, not a constant: it follows the aspect ratio.
@MainActor private func gemmaImageSay(_ chat: GemmaChat, _ file: String,
                                      _ ask: String?, _ cap: Int?,
                                      metal: Bool, clip: String?,
                                      seconds: Double?) async throws {
    let wire = try chat.visionWire()
    let patch = try Gemma4Patchify(chat.model)
    let data = try Data(contentsOf: URL(fileURLWithPath: file))
    // A unified checkpoint has no tower, so its cut is the MERGED one: the
    // 48-pixel block is the soft token itself.
    let cut = chat.model.hasVisionTower
        ? patch.patches(data)
        : patch.merged(data, softTokens: patch.maxSoftTokens)
    guard let cut else {
        throw GemmaCLIError.msg("could not decode \(file)")
    }
    let real = cut.pos.filter { p in p.0 >= 0 }.count
    err("[see] \(file) -> \(real) patches of \(cut.pos.count)\n")
    // The engine leads on the GPU arm: it owns the mapping the tower binds
    // through, and it is the same engine the turn is prefilled on below.
    var gpu: Gemma4MetalEngine? = nil
    if metal { gpu = try Gemma4MetalEngine(chat.model) }
    let t0 = Date()
    let got: (tower: [Float], proj: [Float], count: Int)
    if !chat.model.hasVisionTower {
        let rows = try Gemma4UnifiedMedia(chat.model)
            .image(pixels: cut.pixels, pos: cut.pos)
        got = (tower: [], proj: rows.flatMap { r in r }, count: rows.count)
    } else if let gpu {
        got = try Gemma4MetalViT(chat.model, ctx: gpu.ctx)
            .forward(pixels: cut.pixels, pos: cut.pos)
    } else {
        got = try Gemma4ViT(chat.model).forward(pixels: cut.pixels,
                                                pos: cut.pos)
    }
    err(String(format: "[see] %d soft tokens in %.1fs on %@\n", got.count,
               Date().timeIntervalSince(t0), metal ? "Metal" : "SIMD"))

    let question = ask ?? "Describe this image."
    var parts: [ContentPart] = [.image]
    var expand = [wire.token: span(wire.boi, wire.token, wire.eoi,
                                   got.count)]
    var heard: [Float] = []
    var heardCount = 0
    if let clip {
        let aw = try chat.audioWire()
        let mel = Gemma4Mel(chat.melConfig)
        let pcm = try await AudioFile.samples(
            url: URL(fileURLWithPath: clip),
            sampleRate: Double(mel.cfg.sampleRate), seconds: seconds)
        let a = try gemmaHear(chat, pcm)
        heard = a.proj
        heardCount = a.count
        err("[hear] \(clip) -> \(a.count) soft tokens\n")
        parts.append(.audio)
        expand[aw.token] = span(aw.boa, aw.token, aw.eoa, a.count)
    }
    let ids = try softTurnIds(chat, question, parts, expand)
    let embd = chat.model.cfg.nEmbd
    let prefill: ([Int32], @escaping (Int32) -> [Float]?) -> Int32
    let step: (Int32) -> Int32
    if let gpu {
        gpu.reset()
        prefill = { i, fn in gpu.extend(i, softAt: fn) }
        step = { t in gpu.decode(t) }
    } else {
        let cpu = chat.engine
        cpu.reset()
        prefill = { i, fn in cpu.extend(i, softAt: fn) }
        step = { t in cpu.decode(t) }
    }
    let p0 = Date()
    let audioToken = try? chat.audioWire().token
    var seenN = 0
    var heardN = 0
    var next = prefill(ids) { id in
        var out: [Float]? = nil
        if id == wire.token {
            out = Array(got.proj[(seenN * embd)..<((seenN + 1) * embd)])
            seenN += 1
        } else if let audioToken, id == audioToken, heardN < heardCount {
            out = Array(heard[(heardN * embd)..<((heardN + 1) * embd)])
            heardN += 1
        }
        return out
    }
    err(String(format: "[see] prefill %d tokens (%d + %d soft) in %.1fs\n",
               ids.count, got.count, heardCount,
               Date().timeIntervalSince(p0)))
    var out: [Int32] = []
    while out.count < (cap ?? 128) && !chat.eosIds.contains(next) {
        out.append(next)
        next = step(next)
    }
    print("ASKED: \(question)")
    print("MODEL: \(chat.decode(out))")
}

// A video through the vision tower, frame by frame, and out as text.
//
// There are no video WEIGHTS: a video is the vision tower over sampled
// frames. What differs from a still is the budget (a frame gets
// video.max_soft_tokens against an image's, so ~63 tokens rather than ~270)
// and the markup -- each frame carries its own mm:ss TIMESTAMP and its own
// begin/end pair, which is how the model is told when each frame happened.
@MainActor private func gemmaVideoSay(_ chat: GemmaChat, _ file: String,
                                      _ ask: String?, _ cap: Int?,
                                      metal: Bool, withAudio: Bool,
                                      dumpFrames: String?) async throws {
    let g = chat.model
    let iw = try chat.visionWire()
    let film = try chat.videoWire()
    let vw = film.token
    let patch = try Gemma4Patchify(g)
    let budget = patch.patchBudget(film.softTokensPerFrame)
    let url = URL(fileURLWithPath: file)
    let shot = try await VideoFrames.sample(url: url,
                                           count: film.frames)
    err("[film] \(file) -> \(shot.images.count) frames, "
        + "\(budget) patches each\n")

    // The engine leads on the GPU arm: it owns the mapping the tower binds
    // through, and it is the same engine the turn is prefilled on below. The
    // tower is built ONCE -- per frame, the build would dominate the encode.
    var gpu: Gemma4MetalEngine? = nil
    if metal { gpu = try Gemma4MetalEngine(g) }
    let vit: ([Float], [(Int, Int)])
        -> (tower: [Float], proj: [Float], count: Int)
    if !chat.model.hasVisionTower {
        // Encoder-free: the merged block is the token, so the pixels handed
        // in below are already 48-wide and there is no tower to forward.
        let media = try Gemma4UnifiedMedia(chat.model)
        vit = { pixels, pos in
            let rows = media.image(pixels: pixels, pos: pos)
            return (tower: [], proj: rows.flatMap { r in r },
                    count: rows.count)
        }
    } else if let gpu {
        let tower = try Gemma4MetalViT(g, ctx: gpu.ctx)
        vit = { pixels, pos in tower.forward(pixels: pixels, pos: pos) }
    } else {
        let tower = try Gemma4ViT(g)
        vit = { pixels, pos in tower.forward(pixels: pixels, pos: pos) }
    }
    // Debug: the frames as the model receives them, so a colour or sampling
    // fault is looked at rather than inferred from an answer.
    if let dir = dumpFrames {
        try VideoFrames.write(shot.images, to: dir)
        err("[film] wrote \(shot.images.count) frames to \(dir)\n")
    }
    let t0 = Date()
    var feats: [Float] = []
    var perFrame = 0
    var block: [Int32] = []
    for (i, img) in shot.images.enumerated() {
        let frame = chat.model.hasVisionTower
            ? patch.patches(img, budget: budget)
            : patch.merged(img, softTokens: film.softTokensPerFrame)
        guard let cut = frame else {
            throw GemmaCLIError.msg("frame \(i) would not patchify")
        }
        let out = vit(cut.pixels, cut.pos)
        feats.append(contentsOf: out.proj)
        perFrame = out.count
        block.append(contentsOf:
            chat.encodeRaw(VideoFrames.stamp(shot.seconds[i]) + " "))
        block.append(contentsOf: span(iw.boi, vw, iw.eoi, out.count))
    }
    err(String(format: "[film] %d soft tokens (%d x %d) in %.1fs on %@\n",
               perFrame * shot.images.count, shot.images.count, perFrame,
               Date().timeIntervalSince(t0), metal ? "Metal" : "SIMD"))

    var parts: [ContentPart] = [.video]
    var expand = [vw: block]
    var heard: [Float] = []
    if withAudio {
        let aw = try chat.audioWire()
        let mel = Gemma4Mel(chat.melConfig)
        let pcm = try await AudioFile.samples(
            url: url, sampleRate: Double(mel.cfg.sampleRate))
        let a = try gemmaHear(chat, pcm)
        heard = a.proj
        err("[film] audio track -> \(a.count) soft tokens\n")
        parts.append(.audio)
        expand[aw.token] = span(aw.boa, aw.token, aw.eoa, a.count)
    }
    let audioToken = withAudio ? try chat.audioWire().token : Int32(-1)
    let question = ask ?? "What happens in this video?"
    let ids = try softTurnIds(chat, question, parts, expand)
    let embd = g.cfg.nEmbd
    let prefill: ([Int32], @escaping (Int32) -> [Float]?) -> Int32
    let step: (Int32) -> Int32
    if let gpu {
        gpu.reset()
        prefill = { i, fn in gpu.extend(i, softAt: fn) }
        step = { t in gpu.decode(t) }
    } else {
        let cpu = chat.engine
        cpu.reset()
        prefill = { i, fn in cpu.extend(i, softAt: fn) }
        step = { t in cpu.decode(t) }
    }
    var seen = 0
    var hear = 0
    let p0 = Date()
    var next = prefill(ids) { id in
        var out: [Float]? = nil
        if id == vw {
            out = Array(feats[(seen * embd)..<((seen + 1) * embd)])
            seen += 1
        } else if id == audioToken {
            out = Array(heard[(hear * embd)..<((hear + 1) * embd)])
            hear += 1
        }
        return out
    }
    err(String(format: "[film] prefill %d tokens (%d soft) in %.1fs\n",
               ids.count, seen + hear, Date().timeIntervalSince(p0)))
    var out: [Int32] = []
    while out.count < (cap ?? 128) && !chat.eosIds.contains(next) {
        out.append(next)
        next = step(next)
    }
    print("ASKED: \(question)")
    print("MODEL: \(chat.decode(out))")
}


// Our patchifier against the reference's own pixel_values / position_ids.
// The tower gate feeds the dump's arrays, so the PROCESSOR side -- resize,
// patch order, position convention, padding -- has never been exercised.
@MainActor private func gemmaPatchGate(_ chat: GemmaChat, _ dir: String,
                                       _ image: String) throws {
    let base = URL(fileURLWithPath: dir)
    let wantPx = try npy(base.appendingPathComponent(
        "vision.pixel_values.npy"))
    let wantPos = try npy(base.appendingPathComponent(
        "vision.position_ids.npy"))
    let p = try Gemma4Patchify(chat.model)
    let data = try Data(contentsOf: URL(fileURLWithPath: image))
    guard let got = p.patches(data) else {
        throw GemmaCLIError.msg("could not decode \(image)")
    }
    let n = wantPos.values.count / 2
    var posHits = 0
    var realWant = 0
    for i in 0..<n {
        let wx = Int(wantPos.values[i * 2]), wy = Int(wantPos.values[i * 2 + 1])
        if wx >= 0 || wy >= 0 { realWant += 1 }
        if got.pos[i].0 == wx && got.pos[i].1 == wy { posHits += 1 }
    }
    let cPx = cosine(wantPx.values, got.pixels)
    let px = min(wantPx.values.count, got.pixels.count)
    var worst: Float = 0
    var total = 0.0
    for i in 0..<px {
        let d = abs(wantPx.values[i] - got.pixels[i])
        worst = max(worst, d)
        total += Double(d)
    }
    print("patches              \(got.pos.count) vs reference \(n)")
    print("real patches         \(got.pos.filter { q in q.0 >= 0 }.count) "
        + "vs \(realWant)")
    print("positions exact      \(posHits)/\(n)")
    print(String(format: "pixels cosine        %.6f", cPx))
    print(String(format: "pixels max abs diff  %.4f  mean %.6f", worst,
                 total / Double(max(px, 1))))
    // What remains in those pixels is the JPEG DECODER, not the resize:
    // ImageIO and libjpeg disagree by up to 12 of 255 on this file before
    // either resamples anything, and that accounts for essentially all of
    // the max above. Read the MEAN -- a max over 1.9 M values reports one
    // pixel and says nothing about what the tower sees.
    let wantTower = try npy(base.appendingPathComponent("vision.tower.npy"))
    let vit = try Gemma4ViT(chat.model)
    let ran = vit.forward(pixels: got.pixels, pos: got.pos)
    let refCount = wantTower.shape.first ?? -1
    let cTower = cosine(wantTower.values, ran.tower)
    print("soft tokens          \(ran.count) vs reference \(refCount)")
    print(String(format: "tower cosine         %.6f", cTower))
    // The structural checks are EXACT -- every position, both patch counts,
    // and the soft-token count that drives a turn's budget. The tower gets a
    // band because the decoder gap above survives into it: 0.992 is what the
    // dump's OWN pixels score against this same reference, so that is the
    // ceiling this comparison has rather than a number picked to pass.
    print(posHits == n && cPx >= 0.9999 && ran.count == refCount
          && cTower >= 0.99 ? "GATE PASS" : "GATE FAIL")
}

// The vision tower against the SRQ-off oracle. Pixels and positions come
// from the dump itself, so what is scored is the TOWER and not a
// reimplementation of the processor's patchifier.
@MainActor private func gemmaViTGate(_ chat: GemmaChat, _ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let px = try npy(base.appendingPathComponent("vision.pixel_values.npy"))
    let ps = try npy(base.appendingPathComponent("vision.position_ids.npy"))
    let want = try npy(base.appendingPathComponent("vision.tower.npy"))
    let wantProj = try npy(base.appendingPathComponent("vision.proj.npy"))
    let n = ps.values.count / 2
    var pos: [(Int, Int)] = []
    for i in 0..<n {
        pos.append((Int(ps.values[i * 2]), Int(ps.values[i * 2 + 1])))
    }
    err("[vit-gate] \(n) patches, reference \(want.shape) tower / "
        + "\(wantProj.shape) proj\n")
    let vit = try Gemma4ViT(chat.model)
    let t0 = Date()
    let got = vit.forward(pixels: px.values, pos: pos)
    let cpuSecs = Date().timeIntervalSince(t0)
    let cTower = cosine(want.values, got.tower)
    let cProj = cosine(wantProj.values, got.proj)
    let countOK = got.count == (want.shape.first ?? -1)
    // The tower binds through the TEXT engine's context, here as in the app.
    // The engine is not wanted past the handover and the tower holds the
    // context alive; the text tensors it never reads cost nothing, being
    // mapped rather than resident.
    let gpu = try Gemma4MetalViT(chat.model,
                                 ctx: Gemma4MetalEngine(chat.model).ctx)
    let g0 = Date()
    let mg = gpu.forward(pixels: px.values, pos: pos)
    let gpuSecs = Date().timeIntervalSince(g0)
    let mTower = cosine(got.tower, mg.tower)
    let mProj = cosine(got.proj, mg.proj)
    print("soft tokens          \(got.count) "
        + (countOK ? "MATCH" : "vs reference \(want.shape.first ?? -1)")
        + ", metal \(mg.count)")
    print(String(format: "tower cosine vs ref  %.6f", cTower))
    print(String(format: "proj  cosine vs ref  %.6f", cProj))
    print(String(format: "metal tower vs simd  %.6f", mTower))
    print(String(format: "metal proj  vs simd  %.6f", mProj))
    print(String(format: "forward simd %.1fs | metal %.2fs", cpuSecs,
                 gpuSecs))
    // 0.95 on the metal arm is the model's own reproducibility band under
    // SRQ, not a threshold picked to pass -- see the metal-gate comment.
    print(countOK && mg.count == got.count && cTower >= 0.99
          && cProj >= 0.97 && mTower >= 0.95 && mProj >= 0.95
          ? "GATE PASS" : "GATE FAIL")
}

// Minimal .npy reader: little-endian float32, C order -- the only shape the
// reference dump writes (dump_reference.save casts everything to float32).
// The encoder-free image and audio paths against HF's own arrays.
//
// Two questions, kept apart on purpose. STRUCTURE: do our patches and their
// positions equal HF's, element for element? That is where a port of the
// 3x3 patch merge goes wrong, and it is exact or it is broken. NUMERICS: fed
// HF's OWN patches, does our little stack land where theirs does? Scored by
// cosine, because ours runs in f32 where the reference runs in bf16.
@MainActor private func gemmaUnifiedGate(_ chat: GemmaChat, _ dir: String,
                                         _ image: String,
                                         _ wav: String) throws {
    let base = URL(fileURLWithPath: dir)
    let media = try Gemma4UnifiedMedia(chat.model)
    let cut = try Gemma4Patchify(chat.model)
    var ok = true

    let refPx = try npy(base.appendingPathComponent("vision.pixel_values.npy"))
    let refPos = try npy(base.appendingPathComponent("vision.position_ids.npy"))
    let refEmb = try npy(base.appendingPathComponent("vision.embed.npy"))
    let dim = refPx.shape.count > 1 ? refPx.shape[1] : media.patchDim
    let count = refPx.shape[0]
    print("[unified] vision \(count) patches x \(dim)")

    // HF pads to the whole budget with (-1, -1); we cut only real patches,
    // because a unified checkpoint scatters exactly what it is handed. So the
    // comparison is against the REAL ones.
    var pos: [(Int, Int)] = []
    for i in 0..<count {
        pos.append((Int(refPos.values[i * 2]), Int(refPos.values[i * 2 + 1])))
    }
    var keep: [Int] = []
    for i in 0..<count where pos[i].0 >= 0 { keep.append(i) }
    let real = keep.count

    let mine = cut.merged(try Data(contentsOf: URL(fileURLWithPath: image)),
                          softTokens: cut.maxSoftTokens)
    if let mine {
        let sameCount = mine.pos.count == real
        var posBad = 0
        for i in 0..<min(mine.pos.count, real) {
            let wantX = Int(refPos.values[i * 2])
            let wantY = Int(refPos.values[i * 2 + 1])
            if mine.pos[i].0 != wantX || mine.pos[i].1 != wantY { posBad += 1 }
        }
        var pixMax: Float = 0
        for i in 0..<min(mine.pixels.count, refPx.values.count) {
            pixMax = max(pixMax, abs(mine.pixels[i] - refPx.values[i]))
        }
        ok = ok && sameCount && posBad == 0
        print("  patch count  \(mine.pos.count) vs \(real) real "
            + "(\(count) padded)  " + (sameCount ? "MATCH" : "MISMATCH"))
        print("  positions    \(posBad) wrong of \(real)  "
            + (posBad == 0 ? "MATCH" : "MISMATCH"))
        // The pixels are allowed to differ where the STRUCTURE above is not:
        // this reads a JPEG through ImageIO where the reference read it
        // through libjpeg, and the two decoders disagree before any resize.
        print(String(format: "  pixels       max abs %.4f (decoder)",
                     pixMax))
    } else {
        print("  patchify FAILED")
        ok = false
    }

    let pixels = refPx.values
    var packed = [Float](repeating: 0, count: keep.count * dim)
    for (n, i) in keep.enumerated() {
        for j in 0..<dim { packed[n * dim + j] = pixels[i * dim + j] }
    }
    let got = media.image(pixels: packed, pos: keep.map { i in pos[i] })
    // The reference keeps a row per PADDED slot; gather the real ones so both
    // sides are the same patches, or every score below is diluted by rows we
    // deliberately never produce.
    let wide = refEmb.values.count / max(count, 1)
    var refRows = [Float](repeating: 0, count: keep.count * wide)
    for (n, i) in keep.enumerated() {
        for j in 0..<wide { refRows[n * wide + j] = refEmb.values[i * wide + j] }
    }
    let rows = got.flatMap { row in row }
    let cosV = cosineOf(rows, refRows)
    ok = ok && cosV >= 0.99
    print(String(format: "  embed        %d tokens  cos %.6f  %@",
                 got.count, cosV, cosV >= 0.99 ? "OK" : "FAIL"))
    // A cosine cannot see MAGNITUDE, and these rows are scattered in beside
    // text embeddings that carry embed_scale -- so a missing or doubled
    // scalar leaves the direction perfect and the model blind. Gate the
    // amplitude separately or that whole class of bug reads as a pass.
    let scaleV = amplitude(rows) / amplitude(refRows)
    ok = ok && abs(scaleV - 1) <= 0.01
    print(String(format: "  embed scale  %.6f of reference  %@", scaleV,
                 abs(scaleV - 1) <= 0.01 ? "OK" : "FAIL"))

    let refFeat = try npy(base.appendingPathComponent(
        "audio.input_features.npy"))
    let refAud = try npy(base.appendingPathComponent("audio.embed.npy"))
    let frames = refFeat.shape[0]
    let tokens = media.audio(refFeat.values)
    let cosA = cosineOf(tokens.flatMap { row in row }, refAud.values)
    let scaleA = amplitude(tokens.flatMap { row in row })
        / amplitude(refAud.values)
    ok = ok && cosA >= 0.99 && abs(scaleA - 1) <= 0.01
    print(String(format: "[unified] audio %d frames -> %d tokens  cos %.6f "
        + " scale %.6f  %@", frames, tokens.count, cosA, scaleA,
        cosA >= 0.99 && abs(scaleA - 1) <= 0.01 ? "OK" : "FAIL"))
    _ = wav
    print(ok ? "GATE PASS" : "GATE FAIL")
}

// The LM's own arithmetic on a whole IMAGE TURN, layer by layer.
//
// Every stage BEFORE this one is already scored and matches HF: the pixels,
// the patch geometry, the embedder in direction and magnitude, and the turn's
// ids. Yet HF answers "Yes" to the elephant where both our engines answer
// "No". So the only place left is what the decoder stack does once the soft
// rows are in front of it, and this is what scores it.
//
// --ref-embeds feeds the reference's OWN inputs_embeds at every position
// instead of building them here. That separates the two halves completely: if
// the answer is still wrong on the reference's embeddings, nothing upstream of
// the decoder can be to blame.

@MainActor private func gemmaImageGate(_ chat: GemmaChat, _ dir: String,
                                       _ useRef: Bool,
                                       _ metal: Bool) throws {
    let base = URL(fileURLWithPath: dir)
    let ids = try npy(base.appendingPathComponent("img.ids.npy"))
        .values.map { v in Int32(v) }
    let probe = try npy(base.appendingPathComponent("img.probe.npy"))
        .values.map { v in Int(v) }
    let embeds = try npy(base.appendingPathComponent("img.embeds.npy"))
    let refLogits = try npy(base.appendingPathComponent("img.logits.npy"))
    let raw = try Data(
        contentsOf: base.appendingPathComponent("manifest.json"))
    let man = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    let turn = man["image_turn"] as? [String: Any] ?? [:]
    let image = turn["image"] as? String ?? "tests/vl/elephant.jpeg"
    let embd = chat.model.cfg.nEmbd
    let nLayer = chat.model.cfg.nLayer
    let wire = try chat.visionWire()
    err("[image-gate] \(ids.count) ids, \(probe.count) probes, "
        + (useRef ? "REFERENCE embeddings\n" : "our own embeddings\n"))

    // Our own soft rows, unless the reference's are standing in for them.
    var mine: [Float] = []
    if !useRef {
        let patch = try Gemma4Patchify(chat.model)
        let data = try Data(contentsOf: URL(fileURLWithPath: image))
        guard let cut = patch.merged(data, softTokens: patch.maxSoftTokens)
        else { throw GemmaCLIError.msg("could not cut \(image)") }
        mine = try Gemma4UnifiedMedia(chat.model)
            .image(pixels: cut.pixels, pos: cut.pos).flatMap { r in r }
    }

    // taps[layer][probe index] -- only the probed positions are kept, which
    // is what the reference stored.
    let empty = [[Float]](repeating: [], count: probe.count)
    var got = [[[Float]]](repeating: empty, count: nLayer + 1)
    var attn = [[[Float]]](repeating: empty, count: nLayer)
    var mlp = [[[Float]]](repeating: empty, count: nLayer)
    let slot = Dictionary(uniqueKeysWithValues:
        probe.enumerated().map { i, p in (p, i) })
    // The BATCHED path, because it is the only one that can express a vision
    // block: a per-token prefill has not appended the block's forward keys
    // when it reaches a query inside it. `probe` + `onLayer` read the chunk
    // buffers between layer groups, which prefillLayers=1 already separates.
    let gpu = try Gemma4MetalEngine(chat.model)
    gpu.reset()
    gpu.probe = Set(probe)
    gpu.onLayer = { il, p, v in
        if let at = slot[p] {
            if il < nLayer { got[il][at] = v } else { got[nLayer][at] = v }
        }
    }
    // softAt is asked EXACTLY once per id and in order, so a plain cursor
    // walks whichever array is standing in for the embeddings.
    var soft = 0
    var seen = 0
    _ = gpu.extend(ids) { id in
        var out: [Float]? = nil
        if useRef {
            out = Array(embeds.values[(seen * embd)..<((seen + 1) * embd)])
        } else if id == wire.token {
            out = Array(mine[(soft * embd)..<((soft + 1) * embd)])
            soft += 1
        }
        seen += 1
        return out
    }
    gpu.onLayer = nil

    var first = -1
    for il in 0...nLayer {
        let want = try npy(base.appendingPathComponent(
            String(format: "img.hidden.%02d.npy", il)))
        var worst = 1.0
        var at = -1
        var best = 0.0
        var row = ""
        for (n, p) in probe.enumerated() where !got[il][n].isEmpty {
            let ref = Array(want.values[(n * embd)..<((n + 1) * embd)])
            let c = cosine(ref, got[il][n])
            if c < worst { worst = c; at = p }
            best = max(best, c)
            row += String(format: "  %d:%.4f", p, c)
        }
        // The FIRST diverging layer in full: whether every soft position is
        // wrong or only some of them is the difference between a path bug and
        // a data-dependent one, and a min/max pair cannot tell them apart.
        if worst < 0.99 && first < 0 {
            first = il
            print("   probes" + row)
        }
        // The two BRANCHES at that same worst position. Whichever is already
        // wrong is where to look; a layer cosine alone cannot separate them.
        var branch = ""
        if il < nLayer, let n = probe.firstIndex(of: at) {
            for (name, mine) in [("attn", attn[il]), ("mlp", mlp[il])]
                where !mine[n].isEmpty {
                let url = base.appendingPathComponent(
                    String(format: "img.%@.%02d.npy", name, il))
                if let ref = try? npy(url) {
                    let want = Array(
                        ref.values[(n * embd)..<((n + 1) * embd)])
                    branch += String(format: "  %@ %.6f", name,
                                     cosine(want, mine[n]))
                }
            }
        }
        if il % 4 == 0 || worst < 0.99 {
            print(String(format:
                "  layer %02d  worst %.6f at pos %d, best %.6f%@%@", il,
                worst, at, best, branch,
                worst < 0.99 ? "   <-- DIVERGED" : ""))
        }
    }
    let mineTop = gpu.argmax(gpu.logits())
    var refTop = 0
    var best = -Float.greatestFiniteMagnitude
    for v in 0..<refLogits.values.count where refLogits.values[v] > best {
        best = refLogits.values[v]
        refTop = v
    }
    print("generation point     ours \(mineTop) "
        + chat.decode([Int32(mineTop)]).debugDescription
        + " vs reference \(refTop) "
        + chat.decode([Int32(refTop)]).debugDescription)
    print(first < 0 ? "GATE PASS" : "GATE FAIL (first divergence at layer "
          + "\(first))")
}

// Park a conversation to bytes and bring it back. The ONE check that an
// empty serializeState cannot pass: restore must reproduce the same next
// token WITHOUT a forward pass, and must do it in disk-read time.
// [[never-reprefill]]

@MainActor private func gemmaParkGate(_ chat: GemmaChat,
                                      _ metal: Bool) async throws {
    let backend: any AgentBackend = metal ? try chat.metalBackend()
                                          : chat.backend()
    let ids = chat.encode(String(repeating: "The quarterly logistics review "
        + "covered warehouse throughput and pallet rotation. ", count: 40))
    let t0 = Date()
    let want = try await backend.extend(ids)
    let cooked = Date().timeIntervalSince(t0)
    let state = try await backend.saveState()
    let bytes = await backend.serializeState(state)
    err(String(format: "[park] %d ids cooked in %.2fs -> %d KiB\n",
               ids.count, cooked, bytes.count / 1024))
    if bytes.isEmpty {
        print("park bytes           EMPTY -- this backend re-prefills")
        print("GATE FAIL")
        exit(1)
    }
    await backend.reset()
    let t1 = Date()
    try await backend.loadState(try await backend.deserializeState(bytes))
    let restored = Date().timeIntervalSince(t1)
    // The engine predicts from the state alone, so the SAME token proves the
    // whole KV came back, not merely the position counter.
    let got = try await backend.decode(want)
    let again = try await backend.decode(got)
    await backend.reset()
    _ = try await backend.extend(ids)
    let ref = try await backend.decode(want)
    let ref2 = try await backend.decode(ref)
    print(String(format: "cook                 %.2fs", cooked))
    print(String(format: "restore              %.3fs  (%.0fx faster)",
                 restored, cooked / max(restored, 1e-6)))
    print("next token           \(got) vs \(ref) "
        + (got == ref ? "MATCH" : "DIFFER"))
    print("and the one after    \(again) vs \(ref2) "
        + (again == ref2 ? "MATCH" : "DIFFER"))
    print(got == ref && again == ref2 && restored < cooked / 4
          ? "GATE PASS" : "GATE FAIL")
}

// Root mean square, the size a cosine deliberately divides away.
private func amplitude(_ a: [Float]) -> Float {
    var sum = 0.0
    for v in a { sum += Double(v) * Double(v) }
    return a.isEmpty ? 0 : Float((sum / Double(a.count)).squareRoot())
}

private func cosineOf(_ a: [Float], _ b: [Float]) -> Float {
    let n = min(a.count, b.count)
    var dot = 0.0, na = 0.0, nb = 0.0
    for i in 0..<n {
        dot += Double(a[i]) * Double(b[i])
        na += Double(a[i]) * Double(a[i])
        nb += Double(b[i]) * Double(b[i])
    }
    let d = (na * nb).squareRoot()
    return d > 0 ? Float(dot / d) : 0
}

private func npy(_ url: URL) throws -> (values: [Float], shape: [Int]) {
    let d = try Data(contentsOf: url)
    let headerLen = Int(d[8]) | (Int(d[9]) << 8)
    let start = 10 + headerLen
    let header = String(decoding: d[10..<start], as: UTF8.self)
    var shape: [Int] = []
    if let open = header.range(of: "'shape': ("),
       let shut = header[open.upperBound...].firstIndex(of: ")") {
        shape = header[open.upperBound..<shut]
            .split(separator: ",")
            .compactMap { part in Int(part.trimmingCharacters(
                in: .whitespaces)) }
    }
    let count = (d.count - start) / 4
    var values = [Float](repeating: 0, count: count)
    _ = values.withUnsafeMutableBytes { dst in
        d.copyBytes(to: dst, from: start..<d.count)
    }
    return (values, shape)
}

private func cosine(_ a: [Float], _ b: [Float]) -> Double {
    var dot = 0.0, na = 0.0, nb = 0.0
    for i in 0..<min(a.count, b.count) {
        dot += Double(a[i]) * Double(b[i])
        na += Double(a[i]) * Double(a[i])
        nb += Double(b[i]) * Double(b[i])
    }
    return dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
}

// Replay the reference dump's own ids and write every intermediate the
// oracle carries: one flat f32 blob per array, so the Python side needs no
// Swift-specific parsing.
@MainActor private func gemmaGate(_ chat: GemmaChat, _ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let raw = try Data(
        contentsOf: base.appendingPathComponent("manifest.json"))
    let man = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    let text = man["text"] as! [String: Any]
    let ids = (text["input_ids"] as! [Int]).map { id in Int32(id) }
    err("[gate] replaying \(ids.count) reference tokens\n")

    let eng = chat.engine
    let nLayer = chat.model.cfg.nLayer
    var hidden = [[Float]](repeating: [], count: nLayer + 1)
    var finals: [Float] = []
    var logits: [Float] = []
    eng.reset()
    let t0 = Date()
    for (i, id) in ids.enumerated() {
        let h = eng.forward(token: Int(id), pos: i) { name, il, v in
            if name == "embed" {
                hidden[0].append(contentsOf: v)
            } else if name == "l_out" {
                hidden[il + 1].append(contentsOf: v)
            }
        }
        finals.append(contentsOf: h)
        logits.append(contentsOf: eng.logits(h))
        err(String(format: "\r  %d/%d tokens (%.0fs)", i + 1, ids.count,
                   Date().timeIntervalSince(t0)))
    }
    err("\n")
    for (i, h) in hidden.enumerated() {
        try writeF32(h, base.appendingPathComponent(
            String(format: "swift.hidden.%02d.bin", i)))
    }
    try writeF32(finals, base.appendingPathComponent("swift.final.bin"))
    try writeF32(logits, base.appendingPathComponent("swift.logits.bin"))
    err("[gate] wrote \(hidden.count) hidden + final + logits to \(dir)\n")
}

private func writeF32(_ a: [Float], _ url: URL) throws {
    try a.withUnsafeBytes { raw in try Data(raw).write(to: url) }
}

// The value after `flag` in an untouched argv copy: main.swift's stripValue
// mutates its array, and this runs after that pass.
private func valueAfter(_ args: [String], _ flag: String) -> String? {
    var out: String? = nil
    if let i = args.firstIndex(of: flag), i + 1 < args.count {
        out = args[i + 1]
    }
    return out
}

// Batched prefill against the per-token path, on the SAME engine and the
// reference dump's own ids.
//
// WHY this and not the free-running text: prefill feeds DECODE, so a wrong
// chunk shows up many tokens later as a plausible different sentence. What
// has to match is the state the prefill leaves behind -- the final hidden and
// the logits it produces -- and it has to match at a position that is NOT
// zero, since a continuation turn prefills onto an existing KV and every
// absolute index (causal length, window start, page slot) is basePos-relative.
@MainActor private func gemmaBatchGate(_ chat: GemmaChat,
                                       _ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let raw = try Data(
        contentsOf: base.appendingPathComponent("manifest.json"))
    let man = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    let text = man["text"] as! [String: Any]
    let ids = (text["input_ids"] as! [Int]).map { id in Int32(id) }
    // The chunk scratch is sized at INIT from defaultBatch, so the capacity
    // this gate compares against has to be asked for before the engine is
    // built. Raising `batch` past it wrote off the end of every chunk buffer
    // -- which is what took WindowServer down, so refuse rather than trap.
    //
    // err + exit rather than throw: a RELEASE build strips the message from
    // preconditions AND the top-level error handler, so a thrown
    // GemmaCLIError aborts with a bare SIGTRAP and tells an operator nothing.
    if Gemma4MetalEngine.defaultBatch < 2 {
        err("set LLM_GEMMA_BATCH=256 (or any N > 1) before this gate: the "
            + "engine sizes its chunk scratch for that many tokens at init, "
            + "and the gate compares batch 1 against it\n")
        exit(2)
    }
    let gpu = try Gemma4MetalEngine(chat.model)
    // Compare against what the engine ALLOCATED, not a literal: asking for
    // more than LLM_GEMMA_BATCH trips the setter, and release strips that
    // message too.
    let wide = Gemma4MetalEngine.defaultBatch
    // The two paths CANNOT agree to 1.0 and it is not a bug that they don't:
    // every quantized linear rint()s its input and output to a trained scale,
    // and rint is discontinuous, so any reordering of a reduction lands some
    // element on the other side of a tie and moves it a WHOLE step. That is
    // why the batch-vs-batch-1 cosine sits at 0.988 whether the tiles are
    // fp16 or f32 (measured both). So the question that decides whether
    // batching may ship is not "do the two agree" but "does batching lose
    // accuracy against the REFERENCE" -- HF's own logits for these ids.
    // Only the from-pos-0 run replays the sequence they came from.
    let ref = try npy(base.appendingPathComponent("text.logits.npy"))
    let vocab = ref.shape.last ?? ref.values.count
    let want = Array(ref.values.suffix(vocab))

    func run(_ b: Int, warm: [Int32],
             _ feed: [Int32]) -> (hidden: [Float], logits: [Float]) {
        gpu.batch = b
        gpu.reset()
        // A non-empty warmup is the point: it leaves pos > 0, so the chunk
        // that follows starts mid-KV exactly as a real second turn does.
        if !warm.isEmpty { _ = gpu.extend(warm) }
        _ = gpu.extend(feed)
        return (gpu.hidden(), gpu.logits())
    }
    // The reference prompt is 32 ids against a 512-position window, so
    // T <= window at every position and the sliding branch NEVER runs. A
    // mutation test proved it: an off-by-one injected into the windowed start
    // passed this gate untouched (the goldens caught it, this did not).
    // Repeating the ids past the window makes the sliding layers evict.
    var long = ids
    while long.count <= chat.model.cfg.slidingWindow { long += ids }
    // With LLM_SRQ=0 the clamp is the identity, so nothing discontinuous sits
    // between the two paths and they must agree to fp rounding. THAT is the
    // structural assertion: the SRQ-on comparison cannot go above ~0.99 no
    // matter how correct the code is, which is wide enough to hide a wrong
    // window start or a bad page slot. Read the env directly rather than
    // widening SRQ's visibility for one gate; see SRQ.enabled for the why.
    let clamped =
        ProcessInfo.processInfo.environment["LLM_SRQ"] != "0"
    let floorHidden = clamped ? 0.95 : 0.999999
    err(clamped
        ? "[batch-gate] SRQ ON: accuracy mode, structural defects are BELOW "
          + "the rint() noise floor here. Re-run with LLM_SRQ=0.\n"
        : "[batch-gate] SRQ OFF: structural mode, the two paths must agree "
          + "to fp rounding.\n")
    var failed = 0
    for (label, warm, feed) in [("from pos 0", [Int32](), ids),
                                ("mid-KV", Array(ids.prefix(7)), ids),
                                ("windowed", [Int32](), long)] {
        let one = run(1, warm: warm, feed)
        let many = run(wide, warm: warm, feed)
        let ch = cosine(one.hidden, many.hidden)
        let cl = cosine(one.logits, many.logits)
        var worst: Float = 0
        for i in 0..<min(one.hidden.count, many.hidden.count) {
            worst = max(worst, abs(one.hidden[i] - many.hidden[i]))
        }
        let a1 = Gemma4MetalEngine.argmaxOf(one.logits)
        let a2 = Gemma4MetalEngine.argmaxOf(many.logits)
        print(String(format: "%-10s hidden %.7f  logits %.7f  maxAbs %.5f  "
                     + "argmax %@", (label as NSString).utf8String!, ch, cl,
                     worst, a1 == a2 ? "MATCH (\(a1))"
                                     : "DIFFER \(a1) vs \(a2)"))
        // Only meaningful with the clamp ON: SRQ-off is a materially
        // different model (it moves text logits to cos 0.65 against the
        // shipping one), so scoring it against HF measures that gap, not ours.
        if warm.isEmpty && clamped {
            let hf = Gemma4MetalEngine.argmaxOf(want)
            print(String(format: "  vs HF    batch-1 %.7f  batched %.7f  "
                         + "HF argmax %d, ours %d / %d",
                         cosine(one.logits, want),
                         cosine(many.logits, want), hf, a1, a2))
        }
        if ch < floorHidden || a1 != a2 {
            failed += 1
            print(String(format: "  FAIL %@: hidden %.7f below %.7f",
                         (label as NSString).utf8String!, ch, floorHidden))
        }
    }
    print(failed == 0 ? "GATE PASS" : "GATE FAIL (\(failed))")
    if failed > 0 { exit(1) }
}

import CoreML
import Foundation
import LLM

// Multi-turn chat CLI over the on-ANE state-carry decode trunk. State persists
// across turns (a real conversation), so each turn ingests only new tokens.
// Reports prefill + decode tokens/sec per turn to stderr; assistant text to
// stdout. Turns come from CLI args (scripted) or stdin lines (interactive).
//
//   gadeon-cli models "who are you?" "name three colors"  # scripted
//   gadeon-cli -n 1024 models "..."  # cap generation at 1024 tokens
//   printf 'hi\nbye\n' | gadeon-cli models  # interactive (stdin lines)

var rawArgs = CommandLine.arguments
// >512 carry-prefill vs the token-serial reference
let longDoc = rawArgs.contains("--longdoc")
// continuation via token-by-token (A/B reference)
let forceIngest = rawArgs.contains("--ingest")
// skip the batched-prefill sets (decode-only profiling)
let noCarry = rawArgs.contains("--no-carry")
// run the trunk on CPU, not the ANE (A/B)
let cpuOnly = rawArgs.contains("--cpu")
// Optional generation cap `-n N`: decode stops after N tokens; absent ->
// unlimited (runs to EOS). No memory reason to cap -- the paged KV grows
// lazily, so context is bounded only by the model's 256K training length, not
// a count.
let capVal = stripValue(&rawArgs, "-n", Int.init)
let maxTokens = capVal ?? Int.max
// --spec-n N: MTP draft count for the spec-decode bench / verify modes.
let specNVal = stripValue(&rawArgs, "--spec-n", Int.init)
// --ctx N: repeat the bench prompt to N tokens before decoding, so tg is
// measured at a REALISTIC context. The plain bench decodes at ~640, where the
// KV cache is ~1% of the bytes a token moves and any KV-side change is
// invisible; at 8K it is the dominant term on a dense model.
let benchCtxVal = stripValue(&rawArgs, "--ctx", Int.init)
// --reasoning-effort none|on|low|medium|high. Qwen3.5 implements only none /
// on, so anything but `none` enables thinking; absent -> none (empty-think,
// the direct-answer default).
let reVal = stripValue(&rawArgs, "--reasoning-effort")
let enableThinking = reVal.map { $0 != "none" } ?? false
// --overthink LAMBDA: bias the curated branch-opening tokens down while
// thinking to shorten chain-of-thought (arxiv 2606.00206). Absent / 0 -> off.
let overthink = stripValue(&rawArgs, "--overthink", Float.init) ?? 0
// --max-reasoning N: force </think> after N think tokens (0 = unbounded), so a
// no-EOS thinking runaway cannot hang. An n-gram loop breaker is always on.
let maxReasoning = stripValue(&rawArgs, "--max-reasoning", Int.init) ?? 0
// --soft-reasoning N: SOFT cap -- end <think> at the next paragraph break once
// it passes N tokens (0 = off), a cleaner cut than the hard --max-reasoning.
let softReasoning = stripValue(&rawArgs, "--soft-reasoning", Int.init) ?? 0
// --vl-gate DIR runs the vision roundtrip: load the fixture (patches.bin plus
// a vl_ref.json holding the HF reference), fuse-prefill the image + prompt,
// greedy-decode, and check the first token against HF's argmax.
let vlDir = stripValue(&rawArgs, "--vl-gate")
// --vl-preprocess PNG REF.bin gates the Swift image preprocessor byte-vs-HF
// (no model needed). --vl-image PNG runs the real path: Swift-preprocess ->
// build the VL prompt -> fuse-prefill -> decode.
let vpPair = stripPair(&rawArgs, "--vl-preprocess")
let vpPng = vpPair?.0
let vpBin = vpPair?.1
let viPng = stripValue(&rawArgs, "--vl-image")
// --vl-chat PNG drives the MULTI-TURN vision path: turn 1 is the image via
// ChatSession.replyVision, later turns are text follow-ups that must still see
// the image (the carry gate for multi-turn image).
let vcPng = stripValue(&rawArgs, "--vl-chat")
// --system PROMPT sets the system message (@path reads it from a file); absent
// -> the neutral default.
let sysVal = stripValue(&rawArgs, "--system")
let systemPrompt = sysVal.map { v in
    v.hasPrefix("@")
        ? ((try? String(contentsOfFile: String(v.dropFirst()),
                        encoding: .utf8)) ?? v)
        : v
} ?? "You are a helpful assistant."
// --wiki-model PATH enables the on-device wikipedia_query tool over that
// minilm.gguf AND the online network tools (websearch/fetch). Omitting it is
// airplane mode: only the LOCAL tools (get_current_time, calculator), which are
// always available. The current date/time is injected into the system prompt so
// the model answers "what time is it" directly.
let wmVal = stripValue(&rawArgs, "--wiki-model")
let toolRunner: (any ToolRunner)? =
    SafeToolRunner(slugsPath: wmVal, wikipedia: wmVal != nil,
                   network: wmVal != nil)
// --trace DIR writes one JSON line per turn to DIR/session.jsonl for later
// analysis of an agentic session: the per-channel messages (user / reasoning /
// content / tool-call names), the think/content token split, ctx, pp/tg, and
// wall time. The full engine log (raw tool calls, ctx trajectory) is the
// captured stderr; this is the structured companion.
let trVal = stripValue(&rawArgs, "--trace")
let traceURL: URL? = trVal.map { p in
    let dir = URL(fileURLWithPath: p, isDirectory: true)
    try? FileManager.default.createDirectory(
        at: dir, withIntermediateDirectories: true)
    let file = dir.appendingPathComponent("session.jsonl")
    FileManager.default.createFile(atPath: file.path, contents: nil)
    return file
}
let nowFmt = DateFormatter()
nowFmt.locale = Locale(identifier: "en_US_POSIX")
nowFmt.dateFormat = "yyyy-MM-dd hh:mm a zzz"
// The date/time rides as the session's DYNAMIC system tail so the precooked
// prefix (system + tools) stays byte-stable across runs.
let systemTimeTail = "\nCurrent Date and Time: \(nowFmt.string(from: Date()))"
// --precook FILE: prime the session's system+tools prefix state from FILE,
// or cook + save it there on a miss -- the TTFT cache, per backend.
let pkVal = stripValue(&rawArgs, "--precook")
// --fit feeds a vl-image as ONE fit-to-tile image (no detail tiling), the A/B
// counterpart to the default AnyRes tiling.
let fitImage = rawArgs.contains("--fit")
// --greedy forces temperature-0 (argmax) decode so a vl-image A/B is
// deterministic -- the tower is then the only variable across runs.
let greedyDecode = rawArgs.contains("--greedy")
rawArgs.removeAll {
    ["--longdoc", "--ingest", "--no-carry", "--cpu", "--fit",
     "--greedy"].contains($0)
}

try probeVit()
try probeVLPreprocess()

// arg1 is either a model-set directory (used as-is when it holds
// tokenizer.json, the test/dev path) or a catalog model NAME (Qwen3.5-0.8B /
// QwenPaw-Flash-2B), fetched from the Hub into ./models/<name>/<sha>/ when the
// pinned set is absent.
let arg1 = rawArgs.count > 1 ? rawArgs[1] : "Qwen3.5-0.8B"
// A turn arg of the form @path is replaced by that file's contents, so long
// prompts (which exceed a comfortable command-line arg) can be supplied by
// file.
let turnArgs = (rawArgs.count > 2 ? Array(rawArgs[2...]) : [])
    .filter { !$0.hasPrefix("--") }   // bare boolean flags are not chat turns
    .map { arg -> String in
        arg.hasPrefix("@")
            ? ((try? String(contentsOfFile: String(arg.dropFirst()),
                            encoding: .utf8)) ?? arg)
            : arg
    }

probeCompile()
if rawArgs.contains("--count") { try probeCount() }
await probeNet()

if arg1.hasSuffix(".gguf") { try await runGgufMain() }

let store = URL(fileURLWithPath: "models")
let direct = URL(fileURLWithPath: arg1)
let local = ModelCatalog.localSet(arg1, in: store)
let modelsDir: URL
if FileManager.default.fileExists(
    atPath: direct.appendingPathComponent("tokenizer.json").path) {
    modelsDir = direct
} else if let local, ModelCatalog.isComplete(local) {
    modelsDir = local
} else if let src = ModelCatalog.source(arg1) {
    err("model \(arg1) not in ./models; fetching \(src.repo)\n")
    // GADEON_PRIME=1: compile each finished .mlmodelc in-process while later
    // files still stream, so the post-download load takes e5 cache hits
    // (in-place download puts files at their final, cache-keyed paths).
    // cancel() stops the priming -- past that point the main load would
    // only contend with it for the serial ANECompilerService.
    let primer = ProcessInfo.processInfo.environment["GADEON_PRIME"] == "1"
        ? Primer() : nil
    let setDir = store.appendingPathComponent(arg1)
        .appendingPathComponent(src.revision)
    modelsDir = try await HubFetch.fetch(
        repo: src.repo, prefix: "", into: store.appendingPathComponent(arg1),
        revision: src.revision) { s in
        err("  [\(s.done)/\(s.total)] \(s.file)\n")
        Task { @MainActor in primer?.observe(s.file, set: setDir) }
    }
    primer?.cancel()
} else {
    modelsDir = direct
}

err("loading models...\n")
let chat = try AneChat(modelsDir: modelsDir, cpuOnly: cpuOnly)
// The engine loads the essential (decode) set at init, then the common heavy
// prefill set. The carry (>512) set compiles on demand in the app; a batch
// tool has no cold-launch budget, so block-load it up front so --longdoc +
// batched prefill work on the first turn. --no-carry skips both
// batched-prefill sets (the ~40s/prog compile), so decode-only profiling
// starts instantly.
if !noCarry {
    err("loading heavy prefill set...\n")
    try await chat.loadHeavy()
    err("loading on-demand carry set...\n")
    try await chat.loadCarry()
}
let eng = chat.engine
let tok = chat.tokenizer
err("ready (plain decode).\n")


if rawArgs.contains("--bench") { try await benchCoreML() }
if rawArgs.contains("--bench-mtp") { try await benchMTP() }
if rawArgs.contains("--verify-mtp") { try await verifyMTP() }
if rawArgs.contains("--probe") { try await probeCoreML() }

// Default chat routes through ChatSession (the shipping multi-turn path).
// --ingest keeps the raw token-by-token ChatML path (A/B reference); --longdoc
// keeps the block-carry probe. reasoning-effort none by default.
let fallbackTemplate = """
{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
"""
let templateURL = modelsDir.appendingPathComponent("chat_template.jinja")
let template = (try? String(contentsOf: templateURL, encoding: .utf8))
    ?? fallbackTemplate
// The set's own sampling matrix (the app reads the same file): thinking on/off
// x text/vision, selected per turn. Fall back to the Qwen3.5/QwenPaw card.
let genCfgURL = modelsDir.appendingPathComponent("generation_config.json")
let activePresets = (try? SamplingPresets.from(generationConfig: genCfgURL,
                                               fallback: .qwen35)) ?? .qwen35
let shownConfig = activePresets.select(thinking: enableThinking, vision: false)
err(String(format: "sampler[%@]: temp=%.2f top_p=%.2f top_k=%d " +
    "presence=%.1f repeat=%.2f\n", enableThinking ? "thinking" : "instruct",
    shownConfig.temperature, shownConfig.topP, shownConfig.topK,
    shownConfig.presencePenalty, shownConfig.repeatPenalty))
// The chat path autodetects the MTP self-spec drafter (a set without the
// tensors is a silent no-op); the bench/probe baselines above stay plain.
if !longDoc && !forceIngest {
    await chat.loadMTP()
    if await eng.mtpReady() {
        err("MTP self-speculative decode on (n=\(Engine.specDrafts))\n")
    }
}
let session: ChatSession? = (longDoc || forceIngest) ? nil : ChatSession(
    backend: EngineBackend(chat), template: template,
    system: systemPrompt, systemTail: systemTimeTail,
    vocabSize: tok.vocabCount,
    presets: activePresets, enableThinking: enableThinking,
    maxTokens: maxTokens,
    maxReasoning: maxReasoning, softReasoningCap: softReasoning,
    overthink: overthink, runner: toolRunner)
if let pkVal, let session {
    let url = URL(fileURLWithPath: pkVal)
    let t0 = Date()
    if await session.prime(from: url) {
        err(String(format: "[precook] primed in %.2fs\n",
                   Date().timeIntervalSince(t0)))
    } else {
        try await session.precook(to: url)
        err(String(format: "[precook] cooked + saved in %.1fs\n",
                   Date().timeIntervalSince(t0)))
    }
}

if let vlDir {
    try await runVLGate(vlDir)
} else if let vcPng {
    try await runVLChat(vcPng, turnArgs)
} else if let viPng {
    try await runVLImage(viPng, turnArgs.first ?? VLPrompt.defaultPrompt)
} else if longDoc {
    try await runLongDoc(turnArgs.first
        ?? "The ISDA Master Agreement is the most widely used")
} else if !turnArgs.isEmpty {
    var isFirst = true
    for turn in turnArgs {
        try await runTurn(turn, first: isFirst); isFirst = false
    }
} else {
    err("enter messages (Ctrl-D to end):\n")
    var isFirst = true
    while let line = readLine(strippingNewline: true) {
        if line.isEmpty { continue }
        if line == "/quit" { break }
        try await runTurn(line, first: isFirst); isFirst = false
    }
}

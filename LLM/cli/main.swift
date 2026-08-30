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
// The accelerator is the default; --cpu is the one opt-out. Read HERE because
// the flag sweep below strips --cpu out of rawArgs.
let useGPU = !cpuOnly
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
// --metal-golden DIR: dump each Metal kernel's output bytes on fixed inputs,
// or byte-compare against an earlier dump. The acceptance test for a kernel
// REFACTOR, where cosine is too weak to see a one-ulp change.
let metalGoldenDir = stripValue(&rawArgs, "--metal-golden")
// --reasoning-effort none|on|<level>, where <level> is a word the model's own
// template takes (Qwen3.8: low|medium|xhigh; Qwen3.5 takes none). Anything but
// `none` enables thinking; absent -> none (empty-think, direct answer).
let reVal = stripValue(&rawArgs, "--reasoning-effort")
let enableThinking = reVal.map { $0 != "none" } ?? false
let reasoningEffort = reVal.flatMap { v in
    ["none", "on"].contains(v) ? nil : v
}
// --seed N (or LLM_SEED), like llama.cpp's. Absent -> the Sampler's own
// default, which is FIXED: variety is the invoker's to supply.
let seedVal = stripValue(&rawArgs, "--seed", UInt64.init)
    ?? ProcessInfo.processInfo.environment["LLM_SEED"].flatMap(UInt64.init)
    ?? 0
// --overthink LAMBDA: bias the curated branch-opening tokens down while
// thinking to shorten chain-of-thought (arxiv 2606.00206). Absent / 0 -> off.
let overthink = stripValue(&rawArgs, "--overthink", Float.init) ?? 0
// --max-reasoning N: force </think> after N think tokens (0 = unbounded), so a
// no-EOS thinking runaway cannot hang. An n-gram loop breaker is always on.
let maxReasoning = stripValue(&rawArgs, "--max-reasoning", Int.init) ?? 0
// --soft-reasoning N: SOFT cap -- end <think> at the next paragraph break once
// it passes N tokens (0 = off), a cleaner cut than the hard --max-reasoning.
let softReasoning = stripValue(&rawArgs, "--soft-reasoning", Int.init) ?? 0
// --no-reason: leave the template's reasoning block open but spend none of it
// -- close the channel at its first token and sample as instruct. What a
// system-block template (gemma-4) takes mid-conversation, where the flag
// itself no longer reaches the model.
let suppressReasoning = rawArgs.contains("--no-reason")
// --vl-preprocess PNG REF.bin gates the Swift image preprocessor byte-vs-HF
let vpPair = stripPair(&rawArgs, "--vl-preprocess")
let vpPng = vpPair?.0
let vpBin = vpPair?.1
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
let greedyDecode = rawArgs.contains("--greedy")
rawArgs.removeAll {
    ["--longdoc", "--ingest", "--no-carry", "--cpu", "--fit",
     "--greedy"].contains($0)
}

try probeTTS()
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

await probeNet()

if rawArgs.contains("--meta") { runMeta(rawArgs) }
if rawArgs.contains("--graft") { runGraft(rawArgs) }
if rawArgs.contains("--drafter") { runDrafter(rawArgs) }
if rawArgs.contains("--assist") { runAssist(rawArgs) }
if arg1.hasSuffix(".gguf"), rawArgs.contains("--assist-probe") {
    try runAssistProbe(arg1, rawArgs)
}
if arg1.hasSuffix(".gguf"), rawArgs.contains("--assist-bench") {
    try runAssistBench(arg1, rawArgs)
}
if rawArgs.contains("--splice") { runSplice(rawArgs) }
if arg1.hasSuffix(".gguf"), rawArgs.contains("--replay-make") {
    try runReplayMake(arg1, rawArgs)
}
if arg1.hasSuffix(".gguf"), rawArgs.contains("--replay") {
    try runReplayScore(arg1, rawArgs)
}
if arg1.hasSuffix(".gguf"), rawArgs.contains("--kld")
    || rawArgs.contains("--kld-dump") { try runDivergence(arg1, rawArgs) }
if rawArgs.contains("--puzzle-rescore") { runPuzzleRescore(rawArgs) }
if rawArgs.contains("--puzzle-gate") { await runPuzzleGate(rawArgs) }
if arg1.hasSuffix(".gguf") { try await runGgufMain() }

err("\(arg1): not a .gguf -- this build runs GGUF models only\n")
exit(2)

import Foundation
import LLM

let args = CommandArgs(CommandLine.arguments)
let rawArgs = args.all
// >512 carry-prefill vs the token-serial reference
let longDoc = args.flag("--longdoc")
// continuation via token-by-token (A/B reference)
let forceIngest = args.flag("--ingest")
// skip the batched-prefill sets (decode-only profiling)
let noCarry = args.flag("--no-carry")
let useGPU = !args.flag("--cpu")
// Optional generation cap `-n N`: decode stops after N tokens; absent ->
// unlimited (runs to EOS). No memory reason to cap -- the paged KV grows
// lazily, so context is bounded only by the model's 256K training length, not
// a count.
let capVal = args.int("-n")
let maxTokens = capVal ?? Int.max
// --spec-n N: MTP draft count for the spec-decode bench / verify modes.
args.consume("--spec-n", hasValue: true)
let specNVal = Flags.int("spec-n")
// --ctx N: repeat the bench prompt to N tokens before decoding, so tg is
// measured at a REALISTIC context. The plain bench decodes at ~640, where the
// KV cache is ~1% of the bytes a token moves and any KV-side change is
// invisible; at 8K it is the dominant term on a dense model.
let benchCtxVal = args.int("--ctx")
// --metal-golden DIR: dump each Metal kernel's output bytes on fixed inputs,
// or byte-compare against an earlier dump. The acceptance test for a kernel
// REFACTOR, where cosine is too weak to see a one-ulp change.
let metalGoldenDir = args.value("--metal-golden")
// --reasoning-effort none|on|<level>, where <level> is a word the model's own
// template takes (Qwen3.8: low|medium|xhigh; Qwen3.5 takes none). Anything but
// `none` enables thinking; absent -> none (empty-think, direct answer).
let reVal = args.value("--reasoning-effort")
let enableThinking = (reVal.map { $0 != "none" } ?? false)
    || args.flag("--think")
let reasoningEffort = reVal.flatMap { v in
    ["none", "on"].contains(v) ? nil : v
}
args.consume("--seed", hasValue: true)
args.consume("--verbosity", hasValue: true)
args.consume("--diagnostics", hasValue: true)
let seedVal = Flags.uint64("seed") ?? 0
// --overthink LAMBDA: bias the curated branch-opening tokens down while
// thinking to shorten chain-of-thought (arxiv 2606.00206). Absent / 0 -> off.
let overthink = args.float("--overthink") ?? 0
// --max-reasoning N: force </think> after N think tokens (0 = unbounded), so a
// no-EOS thinking runaway cannot hang. An n-gram loop breaker is always on.
let maxReasoning = args.int("--max-reasoning") ?? 0
// --soft-reasoning N: SOFT cap -- end <think> at the next paragraph break once
// it passes N tokens (0 = off), a cleaner cut than the hard --max-reasoning.
let softReasoning = args.int("--soft-reasoning") ?? 0
// --no-reason: leave the template's reasoning block open but spend none of it
// -- close the channel at its first token and sample as instruct. What a
// system-block template (gemma-4) takes mid-conversation, where the flag
// itself no longer reaches the model.
let suppressReasoning = args.flag("--no-reason")
// --system PROMPT sets the system message (@path reads it from a file); absent
// -> the neutral default.
let systemPrompt = args.text("--system") ?? "You are a helpful assistant."
let wmVal = args.value("--wiki-model")
let toolRunner: (any ToolRunner)? =
    SafeToolRunner(slugsPath: wmVal, wikipedia: wmVal != nil,
                   network: wmVal != nil)
// --trace DIR writes one JSON line per turn to DIR/session.jsonl for later
// analysis of an agentic session: the per-channel messages (user / reasoning /
// content / tool-call names), the think/content token split, ctx, pp/tg, and
// wall time. The full engine log (raw tool calls, ctx trajectory) is the
// captured stderr; this is the structured companion.
let trVal = args.value("--trace")
let traceURL: URL? = trVal.map { p in
    let dir = URL(fileURLWithPath: p, isDirectory: true)
    try? FileManager.default.createDirectory(
        at: dir, withIntermediateDirectories: true)
    let file = dir.appendingPathComponent("session.jsonl")
    FileManager.default.createFile(atPath: file.path, contents: nil)
    return file
}
// --precook FILE: prime the session's system+tools prefix state from FILE,
// or cook + save it there on a miss -- the TTFT cache, per backend.
let pkVal = args.value("--precook")
let imageBudget = args.int("--image-budget") ?? 280
let greedyDecode = args.flag("--greedy")
let stopAtVal = args.int("--stop-at")
var stopArmed = true

if args.flag("--emit-iq-tables") {
    print(IQTablesEmit.header(), terminator: "")
    exit(0)
}

try probeTTS()
try probeVit()

let arg1 = args.rest.first ?? ""
var turnArgs: [String] { args.turns }

await probeNet()

if args.flag("--meta") { runMeta(args) }
if args.flag("--graft") { runGraft(args) }
if args.flag("--drafter") { runDrafter(args) }
if args.flag("--assist") { runAssist(args) }
if isModelFile(arg1), args.flag("--assist-probe") {
    try runAssistProbe(arg1, args)
}
if isModelFile(arg1), args.flag("--assist-bench") {
    try runAssistBench(arg1, args)
}
if args.flag("--splice") { runSplice(args) }
if isModelFile(arg1), args.flag("--replay-make") {
    try runReplayMake(arg1, args)
}
if isModelFile(arg1), args.flag("--replay") {
    try runReplayScore(arg1, args)
}
if isModelFile(arg1), rawArgs.contains("--kld")
    || rawArgs.contains("--kld-dump") { try runDivergence(arg1, args) }
if args.flag("--puzzle-rescore") { runPuzzleRescore(args) }
if args.flag("--puzzle-gate") { await runPuzzleGate(args) }
if isModelFile(arg1) { try await runGgufMain() }

err("\(arg1): not a .ggxf -- this build runs GGUF models only\n")
exit(2)

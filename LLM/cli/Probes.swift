import CoreML
import Foundation
import LLM

// Standalone probe / bench modes: each runs before (or instead
// of) the chat path and exits the process itself. Bodies are
// verbatim main.swift blocks; the dispatcher in main.swift
// keeps their original order.

// Minimal single-turn ChatML wrap (no system prompt, no tool schemas) for the
// greedy probe: the smallest templated prompt that still elicits a direct answer,
// so a slow set's correctness is checkable argmax-deterministically in ~1 block.
// The empty <think></think> is the reasoning-effort-none direct-answer form.
func probeWrap(_ user: String) -> String {
    "<|im_start|>user\n\(user)<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
}

// Bonsai-27B vision tower gate: run the Swift ViT (SIMD/Accelerate) over the
// exact pixels the numpy reference preprocessed and cosine-compare its merged
// embeddings, no LM load. Reference + pixels come from
// scripts/convert/qwen35/bonsai27b_vit_ref.py.
//   gadeon-cli --vit mmproj.gguf --vit-pixels pixels.bin --vit-ref ref.bin
@MainActor func probeVit() throws {
    if let vtIdx = rawArgs.firstIndex(of: "--vit") {
        let mmproj = vtIdx + 1 < rawArgs.count ? rawArgs[vtIdx + 1] : ""
        func pathArg(_ flag: String) -> String? {
            rawArgs.firstIndex(of: flag).flatMap { i in
                i + 1 < rawArgs.count ? rawArgs[i + 1] : nil
            }
        }
        let vit = try ViT(path: mmproj)
        let c = vit.cfg
        err("[vit] \(c.layers) blocks, \(c.embd) wide, \(c.imageSize)px, "
            + "\(c.mergedTokens) merged tokens -> \(c.projDim)\n")
        let pixPath = pathArg("--vit-pixels")
        var pixels = [Float](repeating: 0, count: c.imageSize * c.imageSize * 3)
        if let pixPath {
            let data = try Data(contentsOf: URL(fileURLWithPath: pixPath))
            precondition(data.count == pixels.count * 4,
                         "pixels.bin size mismatch: \(data.count)")
            _ = pixels.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        }
        let t0 = Date()
        let out = vit.forward(pixels: pixels)
        err(String(format: "[vit] forward %.2fs\n", Date().timeIntervalSince(t0)))
        // Cross-gate the GPU tower against the CPU forward (the oracle) on the
        // same pixels: f16 weights are the only delta, so cos must be ~1. The
        // second forward times the resident steady state (no load/dequant).
        if let mvit = try? MetalViT(path: mmproj) {
            let m0 = Date()
            let mout = mvit.forward(pixels: pixels)
            let m1 = Date()
            _ = mvit.forward(pixels: pixels)
            err(String(format: "[vit] metal forward %.2fs (warm %.2fs)\n",
                       m1.timeIntervalSince(m0),
                       Date().timeIntervalSince(m1)))
            var dot: Float = 0, nc: Float = 0, nm: Float = 0
            for i in 0 ..< out.count {
                dot += out[i] * mout[i]
                nc += out[i] * out[i]
                nm += mout[i] * mout[i]
            }
            let mcos = dot / (nc.squareRoot() * nm.squareRoot())
            print(String(format: "VIT metal-vs-cpu cos=%.7f %@", mcos,
                         mcos > 0.999 ? "MATCH" : "MISMATCH"))
            if mcos <= 0.999 { exit(1) }
        } else {
            err("[vit] no Metal device; GPU cross-gate skipped\n")
        }
        if let refPath = pathArg("--vit-ref") {
            let data = try Data(contentsOf: URL(fileURLWithPath: refPath))
            var ref = [Float](repeating: 0, count: data.count / 4)
            _ = ref.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            precondition(ref.count == out.count,
                         "ref count \(ref.count) != out \(out.count)")
            var dot: Float = 0, no: Float = 0, nr: Float = 0
            for i in 0 ..< out.count {
                dot += out[i] * ref[i]
                no += out[i] * out[i]
                nr += ref[i] * ref[i]
            }
            let cos = dot / (no.squareRoot() * nr.squareRoot())
            var worst: Float = 0
            for i in 0 ..< out.count { worst = max(worst, abs(out[i] - ref[i])) }
            print(String(format: "VIT cos=%.7f maxdiff=%.5f %@", cos, worst,
                         cos > 0.9999 ? "MATCH" : "MISMATCH"))
            exit(cos > 0.9999 ? 0 : 1)
        }
        print("VIT ok (no ref given): \(out.count) values")
        exit(0)
    }
}

// Byte-gate the Swift preprocessor against HF's pixel_values (fp16), no model
// load. A vertical flip or patch-order slip shows up on the non-uniform probe.
@MainActor func probeVLPreprocess() throws {
    if let png = vpPng, let bin = vpBin {
        let data = try Data(contentsOf: URL(fileURLWithPath: png))
        let patches = try VisionPreprocess.patches(data, VisionGrid.canonical)
        let ref = try Data(contentsOf: URL(fileURLWithPath: bin))
        var maxDiff: Float = 0
        patches.withUnsafeBytes { pb in
            let pp = pb.bindMemory(to: Float16.self)
            ref.withUnsafeBytes { rb in
                let rp = rb.bindMemory(to: UInt16.self)
                for i in 0 ..< min(patches.count, rp.count) {
                    let d = abs(Float(pp[i]) - Float(Float16(bitPattern: rp[i])))
                    maxDiff = max(maxDiff, d)
                }
            }
        }
        err("[vl-preprocess] \(patches.count) vals maxdiff=\(maxDiff) "
            + (maxDiff < 0.02 ? "MATCH\n" : "MISMATCH\n"))
        exit(maxDiff < 0.02 ? 0 : 1)
    }
}

// Compile/load ONLY the named programs (a comma list of name[:function]) then
// exit -- the WORKER mode for the multi-process cold-compile experiment: N
// copies get disjoint subsets, and whether ANECompilerService exceeds one core
// tells whether the compile serialization is client-side or in the daemon.
@MainActor func probeCompile() {
    if let ci = rawArgs.firstIndex(of: "--compile") {
        let spec = ci + 1 < rawArgs.count ? rawArgs[ci + 1] : ""
        let dir = URL(fileURLWithPath: arg1)
        let t0 = Date()
        for item in spec.split(separator: ",") {
            let parts = item.split(separator: ":", maxSplits: 1)
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndNeuralEngine
            if parts.count == 2 { cfg.functionName = String(parts[1]) }
            let t1 = Date()
            // try? so a candidate function a set does not carry (PrimePlan lists
            // every possible one) fails fast without killing the helper.
            let loaded = (try? MLModel(contentsOf:
                dir.appendingPathComponent(String(parts[0])),
                configuration: cfg)) != nil
            err(loaded
                ? String(format: "compiled %@ in %.1fs\n", String(item),
                         Date().timeIntervalSince(t1))
                : "skipped \(item)\n")
        }
        err(String(format: "[compile] total %.1fs\n",
                   Date().timeIntervalSince(t0)))
        exit(0)
    }
}

// Token-count probe: print how many tokens each turn (e.g. @prompt.txt) encodes
// to under this set's tokenizer.json -- the raw count the CoreML app prefills,
// for sizing a fixed-length benchmark prompt across models with different
// vocabs. Loads ONLY the tokenizer, no engine.
@MainActor func probeCount() throws {
    let store = URL(fileURLWithPath: "models")
    let direct = URL(fileURLWithPath: arg1)
    let dir = FileManager.default.fileExists(
        atPath: direct.appendingPathComponent("tokenizer.json").path)
        ? direct : (ModelCatalog.localSet(arg1, in: store) ?? direct)
    let tok = try Tokenizer(modelsDir: dir)
    for s in turnArgs {
        print("\(tok.encode(s, addSpecial: true).count)")
    }
    exit(0)
}

// Network-tool probes (no model): run one safe tool directly, for bring-up.
@MainActor func probeNet() async {
    if let wi = rawArgs.firstIndex(of: "--web") {
        let q = rawArgs[(wi + 1)...].first { !$0.hasPrefix("--") } ?? ""
        print(await Tools.websearch(q, count: 5))
        exit(0)
    }
    if let ni = rawArgs.firstIndex(of: "--news") {
        let topic = rawArgs[(ni + 1)...].first { !$0.hasPrefix("--") }
        print(await Tools.news(topic))
        exit(0)
    }
    if let fi = rawArgs.firstIndex(of: "--fetch") {
        let u = rawArgs[(fi + 1)...].first { !$0.hasPrefix("--") } ?? ""
        print(await Tools.fetch(u, limit: 1200, offset: 0))
        exit(0)
    }
    if let wi = rawArgs.firstIndex(of: "--weather") {
        let loc = rawArgs[(wi + 1)...].first { !$0.hasPrefix("--") } ?? ""
        print(await Tools.weather(loc))
        exit(0)
    }
}

// Raw prefill/decode throughput, apples-to-apples with `llama-bench -p N -n 128`:
// prefill the prompt's tokens (NO chat template) once for pp t/s, then decode
// 128 greedy tokens for tg t/s. `--bench @prompt512.txt` (or the first turn arg,
// else prompt512.txt). Greedy (no sampler), matching llama-bench; one warmup
// pass first so the ANE graphs are hot before timing.
@MainActor func benchCoreML() async throws {
    let text = turnArgs.first ?? benchPrompt
    let ids = tok.encode(text, addSpecial: true)
    // tg count = -n (default 128). Decode throughput is flat, so a small -n on a
    // slow set (e.g. the 27B at ~1 tok/min) gives the same t/s without a
    // multi-hour tg128.
    let gen = capVal ?? 128
    await eng.useSampler(nil)
    await eng.reset()
    var warm = try await eng.ingestBatched(ids)
    for _ in 0..<8 { warm = try await eng.decode(warm) }
    await eng.reset()
    let p0 = Date()
    var next = try await eng.ingestBatched(ids)
    let ppSec = Date().timeIntervalSince(p0)
    let g0 = Date()
    for _ in 0..<gen { next = try await eng.decode(next) }
    let tgSec = Date().timeIntervalSince(g0)
    print(String(format: "CoreML/ANE  pp%d %.1f t/s  |  tg%d %.1f t/s",
                 ids.count, Double(ids.count) / ppSec,
                 gen, Double(gen) / tgSec))
    exit(0)
}

// MTP self-speculative decode: draft `--spec-n` tokens (default 3) with the MTP
// head, verify + accept + host-rollback per cycle, decode `-n` tokens, report tg
// t/s + tokens-per-cycle (acceptance). Requires the set to carry mtp_front/back +
// the verify trunk (verify{i}of{n}), deployed alongside the base programs.
@MainActor func benchMTP() async throws {
    await eng.loadMTP()
    let text = turnArgs.first(where: { !$0.hasPrefix("-") }) ?? benchPrompt
    let ids = tok.encode(text, addSpecial: true)
    let gen = capVal ?? 128
    let specN = specNVal ?? 3
    await eng.useSampler(nil)
    if !(await eng.mtpReady()) {
        // Override: --bench-mtp on a set with no MTP tensors -> plain decode.
        err("this set ships no MTP tensors; --bench-mtp -> plain decode\n")
        await eng.reset()
        var warm = try await eng.ingestBatched(ids)
        for _ in 0 ..< 8 { warm = try await eng.decode(warm) }
        await eng.reset()
        let p0 = Date()
        var next = try await eng.ingestBatched(ids)
        let ppSec = Date().timeIntervalSince(p0)
        let g0 = Date()
        for _ in 0 ..< gen { next = try await eng.decode(next) }
        let tgSec = Date().timeIntervalSince(g0)
        print(String(format: "CoreML/ANE (no MTP)  pp%d %.1f t/s  |  tg%d %.1f t/s",
                     ids.count, Double(ids.count) / ppSec, gen, Double(gen) / tgSec))
        exit(0)
    }
    await eng.reset()
    var seed = try await eng.ingestBatched(ids)
    await eng.seedSpec(seed)
    for _ in 0 ..< 3 { _ = try await eng.decodeSpecStep(specN) }
    await eng.reset()
    seed = try await eng.ingestBatched(ids)
    await eng.seedSpec(seed)
    var produced = 0
    var cycles = 0
    let g0 = Date()
    while produced < gen {
        produced += try await eng.decodeSpecStep(specN).count
        cycles += 1
    }
    let tgSec = Date().timeIntervalSince(g0)
    let mPerCycle = Double(produced) / Double(max(cycles, 1))
    print(String(format: "CoreML/ANE MTP  tg%d %.1f t/s  |  %.2f tok/cycle "
        + "(n=%d, %d cycles)", produced, Double(produced) / tgSec,
        mPerCycle, specN, cycles))
    let pr = await eng.specProf()
    let cyc = pr.draft + pr.verify + pr.accept + pr.rest
    print(String(format: "  phases/cycle(ms): draft %.2f | verify %.2f | "
        + "accept(heads) %.2f | rest %.2f  (sum %.2f ms, %.1f t/s eff)",
        pr.draft, pr.verify, pr.accept, pr.rest, cyc, mPerCycle / cyc * 1000))
    // Batched-verify-head ceiling: replace the ~m early-stop verify heads with 1.
    let oneHead = pr.accept / max(mPerCycle, 1e-9)   // ~ per-head ms in accept
    let batched = cyc - pr.accept + oneHead
    print(String(format: "  est. batched-verify-head: ~%.2f ms/cycle -> %.1f t/s",
        batched, mPerCycle / batched * 1000))
    exit(0)
}

// Numerical correctness eval: from ONE prefill, decode the same length three ways
// and compare token-by-token -- base plain greedy (decode trunk), spec n=0 (verify
// trunk decoded sequentially: isolates the rollback mechanism), spec n=N. spec-N vs
// spec-0 = rollback/accept correctness; spec-N vs base = end-to-end (any gap here
// beyond the spec-0 gap is the verify-vs-decode pal6 graph difference, not a bug).
@MainActor func verifyMTP() async throws {
    await eng.loadMTP()
    if !(await eng.mtpReady()) { err("no MTP tensors in this set\n"); exit(1) }
    let text = turnArgs.first(where: { !$0.hasPrefix("-") }) ?? benchPrompt
    let ids = tok.encode(text, addSpecial: true)
    let gen = capVal ?? 48
    let specN = specNVal ?? 2
    await eng.useSampler(nil)

    func plainRef() async throws -> [Int32] {
        await eng.reset()
        _ = try await eng.ingestBatched(ids)
        var t = try await eng.plainNext()
        var out = [t]
        for _ in 0 ..< gen { t = try await eng.decodePlain(t); out.append(t) }
        return out
    }
    func specRun(_ n: Int) async throws -> [Int32] {
        await eng.reset()
        _ = try await eng.ingestBatched(ids)
        await eng.seedSpec(try await eng.plainNext())
        var out: [Int32] = []
        while out.count <= gen { out += try await eng.decodeSpecStep(n) }
        return Array(out.prefix(gen + 1))
    }
    func cmp(_ a: [Int32], _ b: [Int32]) -> (Int, Int, Int) {
        let n = min(a.count, b.count)
        var matches = 0, firstDiff = -1
        for i in 0 ..< n where a[i] == b[i] { matches += 1 }
        for i in 0 ..< n where a[i] != b[i] { firstDiff = i; break }
        return (matches, n, firstDiff)
    }

    // Verify needs L=n+1 >= convTaps (3), so the smallest valid draft count is
    // 2, and L <= S caps it at the loaded trunk's width - 1: an S=4 trunk
    // (enough for the n<=3 the decoder ever drafts) cannot verify n=4.
    let cap = await eng.verifyChunk() - 1
    let nA = min(max(2, specN), cap)
    let nB = min(nA + 2, cap)
    let base = try await plainRef()
    let specA = try await specRun(nA)
    let specB = try await specRun(nB)
    let (m0, n0, d0) = cmp(specA, specB)
    let (mb, nb, db) = cmp(specA, base)
    print("VERIFY-MTP (n=\(nA) vs n=\(nB), gen=\(gen)):")
    print(String(format: "  spec-n%d vs spec-n%d (rollback depth): %d/%d match, "
                 + "first diff @%d", nA, nB, m0, n0, d0))
    print(String(format: "  spec-n%d vs base plain-greedy (e2e) : %d/%d match, "
                 + "first diff @%d", nA, mb, nb, db))
    print("  BASE : \(tok.decode(base))")
    print("  SPEC : \(tok.decode(specA))")
    exit(0)
}

// Minimal GREEDY templated probe for fast correctness/iteration on a slow set:
// wrap the question in a minimal single-turn ChatML (no system, no tool schemas
// -> ~1 block, not the ChatSession's 500-tok tool-injected prompt), argmax-decode
// (temperature 0, no sampler) up to `-n` tokens or EOS, and print ids + text +
// wall timings. Greedy so the answer is deterministic -- the real correctness
// check. GADEON_PREFILL_DBG=1 adds per-block progress; GADEON_DECODE_PROF=1 the
// per-token phase split.
@MainActor func probeCoreML() async throws {
    let ids = tok.encode(
        probeWrap(turnArgs.first ?? "What is 2+2? Reply with just the number."),
        addSpecial: true)
    let gen = max(1, capVal ?? 4)
    await eng.useSampler(nil)
    await eng.reset()
    let p0 = Date()
    var next = try await eng.ingestBatched(ids)
    let ppSec = Date().timeIntervalSince(p0)
    var out: [Int32] = [next]
    let g0 = Date()
    while out.count < gen && next != tok.eosId {
        next = try await eng.decode(next); out.append(next)
    }
    let tgSec = Date().timeIntervalSince(g0)
    err(String(format: "[probe] prefill %d tok in %.2fs (%.1f t/s) | "
        + "gen %d tok in %.2fs (%.1f t/s)\n",
        ids.count, ppSec, ppSec > 0 ? Double(ids.count) / ppSec : 0,
        out.count, tgSec, tgSec > 0 ? Double(out.count) / tgSec : 0))
    print("PROBE ids: \(out)")
    print("PROBE txt: \(tok.decode(out))")
    exit(0)
}

// Per-op ANE placement audit of a whole model set (MLComputePlan) -- the Swift
// counterpart of scripts/convert/placement.py, run with no Python. For every
// .mlmodelc the engine loads, count where each op lands and FAIL if a planned
// function runs under 99% on the Neural Engine: a silent CPU/GPU fallback is
// invisible at runtime, since the tokens come out correct either way. Reads
// the compiled programs only -- no engine, no weights.
//   gadeon-cli <set> --place        (add --cpu to audit the CPU_ONLY plan)

@MainActor func placeProbe() async throws {
    let store = URL(fileURLWithPath: "models")
    let direct = URL(fileURLWithPath: arg1)
    let dir = FileManager.default.fileExists(
        atPath: direct.appendingPathComponent("tokenizer.json").path)
        ? direct : (ModelCatalog.localSet(arg1, in: store) ?? direct)
    let units: MLComputeUnits = cpuOnly ? .cpuOnly : .cpuAndNeuralEngine
    let entries = (try? FileManager.default
        .contentsOfDirectory(atPath: dir.path)) ?? []
    let files = entries.filter { n in n.hasSuffix(".mlmodelc") }.sorted()
    if files.isEmpty { err("no .mlmodelc in \(dir.path)\n"); exit(2) }
    err("placement audit of \(files.count) programs in "
        + "\(dir.lastPathComponent) "
        + "(\(cpuOnly ? "CPU_ONLY" : "CPU_AND_NE"))\n")
    var rows: [Placement.Result] = []
    for name in files {
        rows += try await placeProgram(dir.appendingPathComponent(name),
                                       name, units)
    }
    let overall = Placement.gate(rows)
    print("PLACE overall: \(overall ? "PASS" : "FAIL")")
    exit(overall ? 0 : 1)
}

// Audit every function one program carries, print a row each, and return those
// rows so the caller gates over the whole set. MLComputePlan plans only a
// package's DEFAULT function, so when every per-function audit comes back with
// the same fingerprint the functionName was ignored -- collapse to a single
// honest "default" row instead of repeating one plan under several names.

@MainActor func placeProgram(_ url: URL, _ name: String,
                             _ units: MLComputeUnits) async throws
    -> [Placement.Result] {
    var fns = PrimePlan.plan(at: url)
    if fns.isEmpty { fns = [nil] }
    var audited: [Placement.Result] = []
    for fn in fns {
        audited.append(try await Placement.audit(url, function: fn,
                                                 computeUnits: units))
    }
    let uniform = Set(audited.map { r in r.counts }).count == 1
    let collapsed = audited.count > 1 && uniform
    let labels = collapsed
        ? ["\(name):default (only the default fn is planned)"]
        : fns.map { fn in "\(name)\(fn.map { f in ":\(f)" } ?? "")" }
    let rows = collapsed ? [audited[0]] : audited
    for (label, r) in zip(labels, rows) { placeRow(label, r) }
    return rows
}

// One audit row: the counts, the NE percentage, the verdict, and the ops that
// landed off the engine (worst first) when there are any.

@MainActor func placeRow(_ label: String, _ r: Placement.Result) {
    if r.planned {
        let ok = r.pctNE >= Placement.minNEPercent
        print(String(format: "PLACE %@: NE=%d GPU=%d CPU=%d unknown=%d "
            + "placed=%d -> %.2f%% NE %@", label, r.ne, r.gpu, r.cpu,
            r.unknown, r.placed, r.pctNE, ok
                ? "PASS"
                : "FAIL (<\(Int(Placement.minNEPercent))% NE)"))
        if !r.offenders.isEmpty {
            let top = r.offenders
                .sorted { a, b in a.value > b.value }
                .prefix(8)
                .map { entry in "\(entry.key)x\(entry.value)" }
                .joined(separator: " ")
            print("  off the engine: \(top)")
        }
    } else {
        print("PLACE \(label): not planned (skip)")
    }
}

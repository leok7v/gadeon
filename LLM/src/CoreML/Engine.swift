import Accelerate
import CoreML
import Foundation
import os

private let log = Logger(subsystem: "io.github.leok7v.gadeon",
                          category: "engine")

enum EngineError: Error {
    case missingModel(String)
    case contextFull
    case stopped
}

// A stop signal the app can raise WHILE the engine actor is busy. The prefill
// block loops run synchronously inside one actor call (no await), so they hold
// the actor to completion -- an actor-isolated Bool set by a queued requestStop
// would never be observed until the prefill already finished. This lock-backed
// flag is nonisolated, so Stop flips it mid-prefill and the loops see it.

private final class StopSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var raised = false
    func raise() { lock.lock(); raised = true; lock.unlock() }
    func clear() { lock.lock(); raised = false; lock.unlock() }
    var raisedNow: Bool { lock.lock(); defer { lock.unlock() }; return raised }
}

// Host-driven PAGED-KV seq=1 decode for the 0.8B hybrid. The 24-layer GGGAx6
// trunk is cut into 7 programs (progNof7.mlmodelc) around each attention
// layer's flash point: a program ends by emitting one attn layer's FRONT
// projections (q/k/v/gate) and the next begins with that layer's BACK (folding
// the host's attention result back in). Between them the host runs paged
// attention (the flash tile + fp32 online-softmax merge in
// PagedAttention.swift) over a lazy PagePool per attention layer, so context
// is not capped by an in-graph cache. GDN layers are unchanged (conv_state
// [1,6144,3] + rec_state [1,16,128,128]), so one `stepHidden` still advances
// every layer's state by exactly one token and prefill and generation remain
// the same call.

public actor Engine {

    let dim: Int // hidden width; per-model (0.8B 1024, 2B 2048)
    // The lm_head output width, read per-model in init from head.mlmodelc's
    // output shape ([1,1,vocab]) -- the same model-I/O source `dim` comes from
    // (its INPUT shape) -- so it is NOT hardcoded and LLM stays multi-model.
    // The GGUF/ternary path reads the same width from
    // token_embd.weight.dims[1].
    let vocab: Int
    // Prefill block length S: tokens the mf "prefill" trunk does per execute,
    // read from the carry model's `hidden` input [S, dim] so it adapts to the
    // built S. A prompt > blockLen prefills in ceil(len/blockLen) blocks
    // (longDocPrefill).
    private var blockLen = 512 // default until the carry model loads
    // Per-model, NOT hardcoded: the SHAPES come from the model I/O in init --
    // rot from the front layers' `cos` input last dim, dhAttn from the decode
    // tile's `q` [KVH,G,DH] dim 2. ropeBase (rope_theta) is the one scalar not
    // in any tensor shape, so it is read from the set's config.json (emitted
    // from the source GGUF/HF config); a set without config.json is a legacy
    // family set, logged and bridged with the qwen3_5 value.
    let rot: Int
    let ropeBase: Double
    let dhAttn: Int
    // Paged-KV page length, from the decode tile's `K` [KVH,P,DH] dim 1.
    let pageP: Int
    // Derived at init from the decode tile's `q` input [KVH,G,DH]: kvHeads
    // (2/4) from dim 0, gqaGroup (query heads per kv head) from dim 1. nProg
    // (7/9/17) from the mf0of{N} name; attnLayers from GGGA (attn at 4p+3). The
    // 0.8-9B all have gqaGroup 4; the 27B is 6 (24 attn / 4 kv) -- reading it
    // (vs the old hardcoded 4) is what lets the 27B route through this same
    // engine.
    let kvHeads: Int
    let gqaGroup: Int // query heads per kv head (0.8-9B 4, 27B 6)
    let nProg: Int // GGGA -> fronts + norm tail (7/9/17)
    let attnLayers: [Int]
    // CONV_K-1 conv boundary positions, from a GDN layer's `conv0` input
    // [1,CONV_DIM,CONV_K-1] last dim (I/O, not hardcoded).
    let convTaps: Int
    // Greedy alone loops on this hybrid; a full-context repetition penalty
    // breaks the loop and masks the fp16 argmax tie bias (a chosen token else
    // wins ties forever). 1.3 = HF divide-if-positive / multiply-if-negative.
    static let repPenalty: Float = 1.3
    // loadNamed retries only the TRANSIENT aned XPC dropout ("Couldn't
    // communicate with a helper application"), which compiles on a retry once
    // the daemon recovers. A permanent graph rejection is not retried.
    static let aneCompileTries = 3
    // Concurrent mf-program loads. Cold compile is unaffected (the e5rt
    // client and the daemon both serialize internally), but WARM loads gain
    // ~20-25% wall (measured: 0.8B 2.12 -> 1.59s, 9B 10.5 -> 8.3s page-warm).
    // Default ON at >= 16 GiB physical memory (every Mac the 9B targets);
    // smaller devices stay sequential so the concurrent client-side compile
    // peak cannot worsen the iOS jetsam margin. GADEON_PARALLEL_LOAD=1/0
    // overrides either way.
    static let parallelLoad: Bool = {
        let env = ProcessInfo.processInfo.environment["GADEON_PARALLEL_LOAD"]
        let big = ProcessInfo.processInfo.physicalMemory >= (16 << 30)
        return env == nil ? big : env == "1"
    }()
    // Env-gated per-block prefill progress (GADEON_PREFILL_DBG=1): a wide 27B
    // prefill fires many serialized ANE tile predictions per block, so a long
    // prompt can take minutes; a per-block cumulative-timing line makes the run
    // legible. Off by default (the app never sets it; gadeon-cli long tests
    // do).
    static let prefillDbg =
        ProcessInfo.processInfo.environment["GADEON_PREFILL_DBG"] == "1"
    // GADEON_DECODE_PROF=1 dumps the per-token decode phase split (embed /
    // trunk [gdn, attn] / lm_head / argmax) EVERY token instead of the silent
    // 32-token window, so a slow set shows where the per-token time goes on the
    // first few.
    static let decodeProf =
        ProcessInfo.processInfo.environment["GADEON_DECODE_PROF"] == "1"
    // GADEON_SPEC_TRACE=1 logs each spec-dec cycle
    // (p0/drafts/accepted/m/committed/bonus) to stderr -- for hunting the m>=4
    // rollback divergence.
    static let specTrace =
        ProcessInfo.processInfo.environment["GADEON_SPEC_TRACE"] == "1"

    // embedding is nil when the fp16 mmap sidecar (embedTable) is present: the
    // per-token gather is then a memcpy from the mmap'd table, so the pal6
    // embedding model is neither loaded nor charged to the ANE compile budget.
    private var embedding: MLModel?
    private var trunk: [MLModel]           // mf0of7 ... mf6of7 "decode" fn
    private var tile: MLModel              // paged flash tile (per-page)
    // Optional unbounded long-doc prefill: the S=512 state-carry trunk + its
    // batched paged tile, plus a batched embedding gather (token_ids[1,512] ->
    // hidden[512,1024]) replacing the per-token seq=1 loop in 512-wide passes.
    private var carryProgs: [MLModel]?     // carry0of7 ... carry6of7
    private var tilePrefill: MLModel?      // batched paged flash tile (S=512)
    private var carryTile: MLModel?        // in-graph-merge flash tile (P=512)
    private var embeddingBatched: MLModel? // 512-wide embedding gather
    private var lmHead: MLModel
    private let lmHeadOut: String
    // Optional vision tower (vision.mlmodelc): patches[N,PATCH_DIM] -> merged
    // feats[MERGED,dim] in the LM embedding space. Compiled once at launch then
    // OFFLOADED (dropped to nil) so text-only serving never holds it; an image
    // turn reloads it warm (ensureVision) and prefill releases it again.
    private var visionModel: MLModel?
    // Whether a tower exists at all (a VL set), independent of current
    // residency. supportsVision reads this, not visionModel.
    private var visionAvailable = false
    // Tower wall time of the last vision prefill (tile encodes + the warm
    // tower load), for the session's .vision trace event.
    private var visionEncodeSec = 0.0
    // spatial_merge_size, family-constant like dhAttn/ropeBase: the vision grid
    // (t,h,w patches) collapses 2x2 -> one LM token, so MERGED = h*w/merge^2.
    static let visionMerge = 2
    private let modelsDir: URL
    // Reports each program's model.mil LoC as it finishes ANE compile, for the
    // UI progress bar (vs compileTotalLoC). Sendable, so the nonisolated
    // prepare* can read it.
    private let onCompiledLoC: (@Sendable (Int) -> Void)?
    // fp16 [vocab, dim] token-embedding table, mmap'd raw sidecar (tied to the
    // lm_head weight). When present it replaces the embedding model entirely.
    private let embedTable: EmbedTable?
    // ANE by default; CPU for CPU builds.
    public let computeUnits: MLComputeUnits
    // GDN recurrence state, keyed by the program's input feature name
    // (conv0/rec0/...); each step feeds gdnState[name] and stores
    // out[name+"_out"].
    private var gdnState: [String: MLMultiArray]
    // one lazy KV page pool per attention layer, keyed by layer index.
    private var pools: [Int: PagePool]
    private var pos = 0
    // Added to the sequence index to get the RoPE position for decode. 0 for
    // text (rope pos == token index). A vision prefill sets it so decode
    // continues from the image's compressed 3D M-RoPE position, not the raw
    // token count. Cleared on reset.
    private var ropeShift = 0
    // Raised by requestStop() (the Stop button). The prefill loops (tower
    // encode + block loops) poll it and throw EngineError.stopped so a cancel
    // during ingest rolls the turn back; decode cancels via Task.isCancelled on
    // the host loop. Cleared at the start of every prefill. Nonisolated so it
    // can be raised while a synchronous prefill holds the actor.
    private let stop = StopSignal()
    // Precomputed RoPE inverse-frequency table (constant across tokens): the
    // per-token cos/sin fill just multiplies the position in, no pow() per
    // call.
    private let ropeInvFreq: [Double]
    // Reused fp32 scratch for the vectorized vocab argmax: one [vocab] buffer
    // refilled per lm_head call, not a fresh ~1 MB allocation per token.
    private var logitsF32: [Float]
    // Tokens emitted since the last reset. argmax penalizes their logits
    // before the max so the decoder stops circling back to them. Cleared in
    // reset().
    private var genSeen: Set<Int32> = []
    // Agent path installs a Sampler (preset + grammar mask + penalties +
    // seeded RNG); nil -> the shipping app's greedy argmax. The sampler
    // carries its own loop-breaking penalties, so the genSeen rep hack is
    // greedy-only.
    private var sampler: Sampler?
    // MTP self-speculative decode (loadMTP installs these; nil -> plain
    // decode). The drafter (one attention layer, its own KV) + the verify
    // trunk that emits GDN rollback ingredients. specHidden carries the base
    // hidden that seeds the next draft.
    private var mtpFront: MLModel?
    private var mtpBack: MLModel?
    private var verifyProgs: [MLModel]?
    // Verify chunk width, read from the loaded verify trunk's own `hidden`
    // input; 0 until one loads. The verify pass and the host rollback's chunk
    // both read it, so a set baked narrower (4 positions is enough for the
    // n<=3 drafts spec decode ever asks for) runs a matching pass and rollback
    // with no constant to keep in step.
    private var verifyS = 0
    private var mtpPool: PagePool?
    private var specHidden: MLMultiArray?
    // Optional batched lm_head [S,dim]->[S,vocab]: one weight-read verifies all
    // rows' argmaxes vs the ~m sequential seq=1 head calls. nil -> per-row
    // path.
    private var lmHeadBatched: MLModel?
    // Decode profiling: per-phase ns over a 32-token window, dumped as
    // per-token averages via report() (embed / trunk+attn / lm_head / argmax).
    // Cheap: two clock reads per phase.
    private var profEmb = 0.0, profTrunk = 0.0, profHead = 0.0, profArg = 0.0
    // trunk split: mf executes vs attn tiles.
    private var profGDN = 0.0, profAttn = 0.0
    private var profTok = 0

    // ANE-only: every model set loads directly on the Neural Engine (no GPU
    // warm-start / hot-swap). The first (cold-cache) load blocks on the ANE
    // compile of the whole model (decode + prefill sets); the app gates chat
    // on that finishing.

    public init(modelsDir: URL, cpuOnly: Bool = false,
                onCompiledLoC: (@Sendable (Int) -> Void)? = nil) throws {
        self.modelsDir = modelsDir
        self.computeUnits = cpuOnly ? .cpuOnly : .cpuAndNeuralEngine
        self.onCompiledLoC = onCompiledLoC
        // rope_theta: the one geometry scalar not in a tensor shape, so it
        // comes from the set's config.json (a legacy family set with no
        // config.json is logged and bridged with the qwen3_5 value). The RoPE
        // inv-freq table is built below, once rot is read from the model I/O.
        self.ropeBase = Engine.ropeThetaFromConfig(modelsDir)
        // Per-model from the name: mf0of{N} gives the count; GGGA puts attn at
        // 4p+3, so 7- and 9-program sets share one loop, no config file.
        let nProg = Engine.programCount(modelsDir)
        self.nProg = nProg
        self.attnLayers = (0 ..< nProg - 1).map { 4 * $0 + 3 }
        // Load only the decode-ready set here (usable in seconds); the heavy
        // S=512 set loads later via prepareHeavy. Heavy refs start nil.
        let m = try Engine.loadEssential(modelsDir, computeUnits, nProg,
                                         onCompiledLoC)
        // EVERY geometry number below is force-read from the model I/O: a set
        // missing a shape is broken and must crash, not silently fake a
        // default -- a wrong silent default is the G=4 class of bug that
        // broke the 27B.
        // Hidden width from the lm_head input [.., dim] (1024/2048), no
        // config.
        self.dim = m.lmHead.modelDescription
            .inputDescriptionsByName["hidden_states"]!
            .multiArrayConstraint!.shape.last!.intValue
        // lm_head OUTPUT width = vocab, per-model (head.mlmodelc "logits"
        // [1,1,vocab]); the sole output feature, so take the first (==
        // lmHeadOut, set below).
        let headOut = m.lmHead.modelDescription
            .outputDescriptionsByName.keys.sorted().first!
        self.lmHeadOut = headOut
        self.vocab = m.lmHead.modelDescription.outputDescriptionsByName[headOut]!
            .multiArrayConstraint!.shape.last!.intValue
        self.logitsF32 = [Float](repeating: 0, count: self.vocab)
        // kv-head count from the decode tile's `q` input [KVH,G,DH] (2 for the
        // 0.8B/2B, 4 for the 4B/9B); the page pools + attn packers key off it.
        let qShape = m.tile.modelDescription
            .inputDescriptionsByName["q"]!.multiArrayConstraint!.shape
        self.kvHeads = qShape[0].intValue
        // query-heads-per-kv from the SAME tile `q` input [KVH,G,DH], dim 1.
        self.gqaGroup = qShape[1].intValue
        // head dim from the tile `q` [KVH,G,DH] dim 2; paged-KV page length
        // from the tile `K` [KVH,P,DH] dim 1.
        self.dhAttn = qShape[2].intValue
        self.pageP = m.tile.modelDescription.inputDescriptionsByName["K"]!
            .multiArrayConstraint!.shape[1].intValue
        // partial-rope dim from a front layer's `cos` [1,1,1,ROT] last dim;
        // conv taps from a GDN layer's `conv0` [1,CONV_DIM,CONV_K-1] last dim.
        // Program 0 = [G0,G1,G2,attn_front] carries both.
        self.rot = m.trunk[0].modelDescription.inputDescriptionsByName["cos"]!
            .multiArrayConstraint!.shape.last!.intValue
        self.convTaps = m.trunk[0].modelDescription
            .inputDescriptionsByName["conv0"]!.multiArrayConstraint!
            .shape.last!.intValue
        // RoPE inv-freq now that rot + ropeBase are known.
        var inv = [Double](repeating: 0, count: rot / 2)
        for j in 0 ..< rot / 2 {
            inv[j] = 1.0 / pow(ropeBase, Double(2 * j) / Double(rot))
        }
        self.ropeInvFreq = inv
        self.embedTable = Engine.openEmbedTable(modelsDir, self.dim, self.vocab)
        self.embedding = m.embedding
        self.trunk = m.trunk
        self.tile = m.tile
        self.carryProgs = nil
        self.tilePrefill = nil
        self.embeddingBatched = nil
        self.lmHead = m.lmHead
        self.visionModel = m.vision
        self.visionAvailable = m.vision != nil
        // lmHeadOut + vocab were derived above (before kvHeads) so vocab could
        // size the argmax scratch + the embed-table length.
        // zero-initialize the GDN conv/rec state from the programs' input
        // signatures; attention state lives in the page pools, not this dict.
        var gs: [String: MLMultiArray] = [:]
        for prog in m.trunk {
            for (name, desc) in prog.modelDescription.inputDescriptionsByName
            where name.hasPrefix("conv") || name.hasPrefix("rec") {
                if gs[name] == nil {
                    let shape = desc.multiArrayConstraint!.shape
                        .map { $0.intValue }
                    gs[name] = try Engine.zeros(shape)
                }
            }
        }
        self.gdnState = gs
        var pl: [Int: PagePool] = [:]
        for j in attnLayers {
            pl[j] = PagePool(P: pageP, kvHeads: kvHeads, dh: dhAttn)
        }
        self.pools = pl
        // The shipped config.json (origin HF form) witnesses the I/O-derived
        // geometry above: a disagreement is a mispaired or stale set and must
        // crash here, not serve garbage tokens.
        let cfgURL = modelsDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: cfgURL.path) {
            let bad = Engine.configMismatches(
                Engine.configJSON(cfgURL), dim: dim, vocab: vocab,
                kvHeads: kvHeads, gqaGroup: gqaGroup, dhAttn: dhAttn,
                rot: rot, convTaps: convTaps, nProg: nProg)
            if !bad.isEmpty {
                fatalError("config.json disagrees with the model I/O: "
                    + bad.joined(separator: "; "))
            }
        }
    }

    // The ESSENTIAL (decode-ready) set: plain seq=1 decode. Small + cached, so
    // it loads in seconds even on a cold ANE.

    private struct Essential {
        let embedding: MLModel?
        let trunk: [MLModel]
        let tile: MLModel
        let lmHead: MLModel
        let vision: MLModel?
    }

    // Load one required program, timing it. A non-nil `function` selects one
    // graph of a multifunction .mlmodelc: mfNof7 packs "prefill" and "decode"
    // over ONE shared pal6 blob, so both functions stream one mmap (single
    // residency). nil loads a plain model.

    private static func loadNamed(_ dir: URL, _ name: String,
                                  _ cfg: MLModelConfiguration,
                                  function: String? = nil,
                                  report: (@Sendable (Int) -> Void)? = nil)
        throws -> MLModel {
        let url = dir.appendingPathComponent(name)
        if !FileManager.default.fileExists(atPath: url.path) {
            throw EngineError.missingModel(name)
        }
        // Copy the config before setting functionName so it never leaks onto
        // the caller's shared config (reused across single-function loads
        // too).
        let use: MLModelConfiguration
        if let function = function {
            let c = cfg.copy() as! MLModelConfiguration
            c.functionName = function
            use = c
        } else {
            use = cfg
        }
        // MLModel(contentsOf:) triggers the ANE compile (seconds cold, ~ms
        // warm); log it so a cold launch is legible in `log stream`.
        let fn = function ?? "-"
        let mil = Engine.milStats(url)
        Diag.shared.report(
            "ANE compile START \(name):\(fn) "
            + "model.mil \(mil.loc) LoC \(mil.kb) KB")
        let t0 = Date()
        var model: MLModel? = nil
        var lastError: Error? = nil
        var attempt = 0
        while model == nil && attempt < Engine.aneCompileTries {
            do {
                model = try MLModel(contentsOf: url, configuration: use)
            } catch {
                lastError = error
                // Retry only the transient aned XPC dropout; a permanent
                // rejection (e.g. H12 "same dimension in view") repeats, so
                // stop at once rather than burn two more compiles.
                let transient = "\(error)".contains("helper application")
                attempt = transient ? attempt + 1 : Engine.aneCompileTries
                // Back off so aned is back up; blocking sleep is fine on this
                // detached background thread (never the actor or main).
                if attempt < Engine.aneCompileTries {
                    Diag.shared.report(
                        "ANE compile RETRY \(name):\(fn) attempt \(attempt)")
                    Thread.sleep(forTimeInterval: 0.25 * Double(attempt))
                }
            }
        }
        if model != nil {
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            Diag.shared.report("ANE compile END   \(name):\(fn) in \(ms)ms")
            report?(Engine.compileCost(name, function: function, loc: mil.loc))
        }
        return try Engine.require(model, lastError, name)
    }

    // Unwrap the loaded model or throw the last error (missing file ->
    // missingModel). Mirrors HubFetch.need: one throw, single exit.

    private static func require(_ model: MLModel?, _ cause: Error?,
                                _ name: String) throws -> MLModel {
        if model == nil {
            throw cause ?? EngineError.missingModel(name)
        }
        return model!
    }

    // Unwrap a geometry value the model was supposed to declare. Same shape as
    // require above, for the reads that are not models: a set that fails to
    // declare one is broken, and saying so beats substituting a plausible
    // number (see the G=4 class of bug the I/O-derived geometry exists to
    // prevent).

    private static func require<T>(_ value: T?, _ what: String) throws -> T {
        if value == nil {
            throw EngineError.missingModel(what)
        }
        return value!
    }

    // The leading dimension of a program's `hidden` input -- the chunk width it
    // was compiled for. The verify trunk and the batched verify head each carry
    // their own, and they must agree; nothing here reads a file name.

    private static func hiddenWidth(_ model: MLModel) -> Int? {
        model.modelDescription.inputDescriptionsByName["hidden"]?
            .multiArrayConstraint?.shape.first?.intValue
    }

    // model.mil size (lines + KB) for the compile log. A multifunction file
    // holds all its functions in one .mil, so this is the per-FILE size.
    private static func milStats(_ mlmodelc: URL) -> (loc: Int, kb: Int) {
        var loc = 0
        var kb = 0
        if let data = try? Data(
            contentsOf: mlmodelc.appendingPathComponent("model.mil")) {
            kb = (data.count + 512) / 1024
            loc = data.reduce(0) { $1 == 0x0A ? $0 + 1 : $0 }
        }
        return (loc, kb)
    }

    // Trunk program count from the mf0of{N}.mlmodelc name: 7 (0.8B/2B), 9
    // (4B/9B). Defaults to 7 for a set predating mf naming or mid-download.
    // Public: the app's compiled-once key derives the mf0 file name from it.

    public static func programCount(_ dir: URL) -> Int {
        let items = (try? FileManager.default
            .contentsOfDirectory(atPath: dir.path)) ?? []
        var n = 7
        for name in items
        where name.hasPrefix("mf0of") && name.hasSuffix(".mlmodelc") {
            let mid = name.dropFirst("mf0of".count).dropLast(".mlmodelc".count)
            if let parsed = Int(mid) { n = parsed }
        }
        return n
    }

    // Per-program compile-COST weight, in "decode model.mil LoC" units, so the
    // Optimizing progress fraction tracks TIME, not text -- raw LoC badly
    // mis-ranks the ANE compile. Measured on the 0.8B (iPhone): the S=512
    // "prefill" graph compiles ~7x slower per LoC than "decode"; a pal6 head
    // function costs a ~FIXED amount (dominated by the vocab blob, not its tiny
    // .mil); the vision tower ~11x; the flash tile a small fixed amount. These
    // are graph properties (device-independent ratios); the ETA's
    // elapsed/fraction extrapolation self-calibrates the absolute device speed,
    // so a cost-weighted fraction stops the early (decode) phase reading
    // optimistic while the heavy prefill phase still looms.
    static let headCost = 13000
    static let tileCost = 900
    static let prefillWeight = 7
    static let visionWeight = 11

    static func compileCost(_ name: String, function: String?,
                            loc: Int) -> Int {
        let result: Int
        if name == "head.mlmodelc" || name == "lm_head.mlmodelc"
            || name.hasPrefix("embedding") || name == "mtp_lmhead.mlmodelc" {
            result = headCost
        } else if name == "vision.mlmodelc" {
            result = loc * visionWeight
        } else if name.hasPrefix("tile") {
            result = tileCost
        } else if function == "prefill" {
            result = loc * prefillWeight
        } else {
            result = loc
        }
        return result
    }

    // Compile the smallest program (the tile) once and discard it, so the e5
    // daemon CREATES and initializes its cache directory before the cache
    // keeper relinks the shadow's restored entries into it. On a fresh install
    // the dir does not exist yet, and the daemon warm-hits on lookup only from
    // a dir it owns -- a hand-made one it ignores. The tile is tiny (~1s cold),
    // and the real load's tile compile then warm-hits this one.

    public static func primeCacheDir(_ dir: URL) {
        let tile = dir.appendingPathComponent("tile.mlmodelc")
        if FileManager.default.fileExists(atPath: tile.path) {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .all
            let t0 = Date()
            _ = try? MLModel(contentsOf: tile, configuration: cfg)
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            Diag.shared.report("primed cache dir via tile in \(ms)ms")
        }
    }

    // Total compile COST the load path will report, for the progress
    // denominator. MUST mirror loadEssential/loadHeavy/loadCarry/loadMTP: head
    // compiles 3 functions (embed + lm_head + embed_batched, the embeds skipped
    // with the fp16 sidecar), each mf twice (decode + prefill), tile once, the
    // vision tower once when the set is VL, and -- when the set ships the MTP
    // drafter -- each mf's "verify" function, mtp_front/back, and the batched
    // verify head. cpuOnly drops the batched-prefill compiles (embed_batched +
    // mf "prefill"): CPU never loads them (BNNS tensors OOM low-memory
    // devices).

    public static func compileTotalLoC(_ dir: URL, cpuOnly: Bool = false) -> Int {
        let fm = FileManager.default
        let nProg = programCount(dir)
        func exists(_ n: String) -> Bool {
            fm.fileExists(atPath: dir.appendingPathComponent(n).path)
        }
        func cost(_ n: String, _ fn: String?) -> Int {
            exists(n)
                ? compileCost(n, function: fn,
                              loc: milStats(dir.appendingPathComponent(n)).loc)
                : 0
        }
        let hasHead = exists("head.mlmodelc")
        let hasSidecar = exists("embed_weight.bin")
        let hasCarry = !cpuOnly && exists("mf0of\(nProg).mlmodelc")
            && exists("tile_prefill.mlmodelc")
        var total = 0
        if hasHead {
            total += cost("head.mlmodelc", "lm_head")
            // embed (essential) always; embed_batched (heavy) only off CPU.
            if !hasSidecar {
                total += (cpuOnly ? 1 : 2) * cost("head.mlmodelc", "embed")
            }
        } else {
            total += cost("lm_head.mlmodelc", nil)
            if !hasSidecar {
                total += cost("embedding.mlmodelc", nil)
                if !cpuOnly { total += cost("embedding_batched.mlmodelc", nil) }
            }
        }
        for i in 0 ..< nProg {
            total += cost("mf\(i)of\(nProg).mlmodelc", "decode")
            if hasCarry { total += cost("mf\(i)of\(nProg).mlmodelc", "prefill") }
        }
        total += cost("tile.mlmodelc", nil)
        total += cost("vision.mlmodelc", nil)
        if exists("mtp_front.mlmodelc") {
            total += cost("mtp_front.mlmodelc", nil)
                + cost("mtp_back.mlmodelc", nil)
            let standalone = exists("verify0of\(nProg).mlmodelc")
            for i in 0 ..< nProg {
                total += standalone
                    ? cost("verify\(i)of\(nProg).mlmodelc", "verify")
                    : cost("mf\(i)of\(nProg).mlmodelc", "verify")
            }
            if exists("mtp_lmhead.mlmodelc") {
                total += cost("mtp_lmhead.mlmodelc", nil)
            } else if hasHead {
                total += cost("head.mlmodelc", "lm_head")
            }
        }
        return total
    }

    // Raw fp16 [vocab, dim] token-embedding table, mmap'd once, read by row.
    // Row `id` is dim fp16 values at byte offset id*dim*2, so a gather is a
    // plain memcpy (the trunk consumes fp16 hidden).

    private struct EmbedTable {
        let base: UnsafeRawPointer
        let rowBytes: Int
        func copyRow(_ id: Int32, to dst: UnsafeMutableRawPointer,
                     rowOffsetBytes: Int) {
            memcpy(dst + rowOffsetBytes, base + Int(id) * rowBytes, rowBytes)
        }
    }

    // mmap embed_weight.bin read-only if it exists; MADV_RANDOM because token
    // rows are scattered. nil -> the embedding MLModel path is used instead.

    // rope_theta from the set's config.json (the one geometry scalar not in any
    // tensor shape). The set ships the ORIGIN HF config verbatim, which nests
    // it at text_config.rope_parameters.rope_theta; the retired minimal set
    // form held it top-level, so both read. A set with NO config.json is a
    // legacy family set: log loudly and bridge with the qwen3_5 value (1e7). A
    // config.json PRESENT but carrying no readable rope_theta anywhere is
    // malformed and crashes (no silent default).

    private static func ropeThetaFromConfig(_ dir: URL) -> Double {
        let url = dir.appendingPathComponent("config.json")
        let result: Double
        if !FileManager.default.fileExists(atPath: url.path) {
            log.warning("""
                no config.json in \(dir.lastPathComponent, privacy: .public); \
                legacy qwen3_5 rope_theta 1e7. Re-emit the set to embed config.json.
                """)
            result = 10_000_000.0
        } else if let theta = Engine.ropeTheta(Engine.configJSON(url)) {
            result = theta
        } else {
            fatalError("config.json has no readable rope_theta: \(url.path)")
        }
        return result
    }

    // The parsed config.json object; [:] when unreadable, so every lookup
    // misses and the callers complain loudly instead of faking values.

    static func configJSON(_ url: URL) -> [String: Any] {
        let obj = (try? Data(contentsOf: url)).flatMap { data in
            try? JSONSerialization.jsonObject(with: data)
        }
        return obj as? [String: Any] ?? [:]
    }

    static func ropeTheta(_ obj: [String: Any]) -> Double? {
        let text = obj["text_config"] as? [String: Any] ?? obj
        let rope = text["rope_parameters"] as? [String: Any] ?? [:]
        let n = (rope["rope_theta"] as? NSNumber)
            ?? (text["rope_theta"] as? NSNumber)
            ?? (obj["rope_theta"] as? NSNumber)
        return n?.doubleValue
    }

    // config.json is a WITNESS, never an authority: each geometry field the
    // conversion preserves is compared against the value derived from the
    // model I/O, and a disagreement means a mispaired or stale set (wrong
    // config beside right weights, a re-emit that drifted). Absent fields
    // verify nothing, so a legacy minimal config checks only what it has.
    // Returns the disagreements; the caller crashes on any.

    static func configMismatches(_ obj: [String: Any], dim: Int, vocab: Int,
                                 kvHeads: Int, gqaGroup: Int, dhAttn: Int,
                                 rot: Int, convTaps: Int,
                                 nProg: Int) -> [String] {
        let text = obj["text_config"] as? [String: Any] ?? obj
        let rope = text["rope_parameters"] as? [String: Any] ?? [:]
        var bad: [String] = []
        let ints: [(key: String, derived: Int)] = [
            ("hidden_size", dim),
            ("vocab_size", vocab),
            ("num_key_value_heads", kvHeads),
            ("num_attention_heads", kvHeads * gqaGroup),
            ("head_dim", dhAttn),
            ("linear_conv_kernel_dim", convTaps + 1),
        ]
        for c in ints {
            if let v = (text[c.key] as? NSNumber)?.intValue, v != c.derived {
                bad.append("\(c.key) \(v) != model \(c.derived)")
            }
        }
        if let layers = (text["num_hidden_layers"] as? NSNumber)?.intValue,
           let every = (text["full_attention_interval"] as? NSNumber)?
               .intValue,
           every > 0, layers / every + 1 != nProg {
            bad.append("num_hidden_layers \(layers) / "
                + "full_attention_interval \(every) + 1 != \(nProg) programs")
        }
        if let pf = (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue,
           Int((pf * Double(dhAttn)).rounded()) != rot {
            bad.append("partial_rotary_factor \(pf) * head_dim \(dhAttn) "
                + "!= rot \(rot)")
        }
        if let inter = rope["mrope_interleaved"] as? NSNumber,
           !inter.boolValue {
            bad.append("mrope_interleaved false; the engine ropes interleaved")
        }
        if let sec = (rope["mrope_section"] as? [NSNumber])?
            .map({ n in n.intValue }) {
            // mropeBatched assigns frequency j to axis j % 3; the config's
            // section sizes must be exactly that interleave's per-axis counts.
            var counts = [0, 0, 0]
            for j in 0 ..< rot / 2 { counts[j % 3] += 1 }
            if sec != counts {
                bad.append("mrope_section \(sec) != interleaved split \(counts)")
            }
        }
        if let vc = obj["vision_config"] as? [String: Any] {
            let vints: [(key: String, expect: Int)] = [
                ("spatial_merge_size", Engine.visionMerge),
                ("patch_size", VisionPreprocess.patch),
                ("temporal_patch_size", VisionPreprocess.temporal),
            ]
            for c in vints {
                if let v = (vc[c.key] as? NSNumber)?.intValue, v != c.expect {
                    bad.append("vision_config.\(c.key) \(v) != \(c.expect)")
                }
            }
        }
        return bad
    }

    private static func openEmbedTable(_ dir: URL, _ dim: Int, _ vocab: Int) -> EmbedTable? {
        let url = dir.appendingPathComponent("embed_weight.bin")
        let len = vocab * dim * 2
        var table: EmbedTable? = nil
        let fd = open(url.path, O_RDONLY)
        if fd >= 0 {
            let raw = mmap(nil, len, PROT_READ, MAP_PRIVATE, fd, 0)
            close(fd)
            if let raw, raw != MAP_FAILED {
                madvise(raw, len, MADV_RANDOM)
                table = EmbedTable(base: UnsafeRawPointer(raw),
                                   rowBytes: dim * 2)
            }
        }
        return table
    }

    // Ordered result slots for the concurrent program loads: models land by
    // index, the first error wins, take() rethrows it or returns the array.
    private final class LoadSlots: @unchecked Sendable {
        private let lock = NSLock()
        private var models: [MLModel?]
        private var firstError: Error?
        init(_ n: Int) { models = [MLModel?](repeating: nil, count: n) }
        func set(_ i: Int, _ m: MLModel) {
            lock.lock(); models[i] = m; lock.unlock()
        }
        func fail(_ e: Error) {
            lock.lock()
            if firstError == nil { firstError = e }
            lock.unlock()
        }
        func take() throws -> [MLModel] {
            lock.lock()
            defer { lock.unlock() }
            if let e = firstError { throw e }
            return models.map { $0! }
        }
    }

    // Load the trunk's nProg programs (one function each), sequentially or --
    // see parallelLoad -- on concurrent threads. Order is preserved either
    // way. Takes compute UNITS, not a shared MLModelConfiguration: the config
    // is not Sendable, so each worker builds its own and the closure is
    // genuinely @Sendable rather than checker-silenced.

    private static func loadPrograms(_ dir: URL, _ units: MLComputeUnits,
                                     _ nProg: Int, function: String,
                                     _ report: (@Sendable (Int) -> Void)?)
        throws -> [MLModel] {
        let slots = LoadSlots(nProg)
        let load: @Sendable (Int) -> Void = { i in
            do {
                let cfg = MLModelConfiguration()
                cfg.computeUnits = units
                slots.set(i, try Engine.loadNamed(
                    dir, "mf\(i)of\(nProg).mlmodelc", cfg,
                    function: function, report: report))
            } catch {
                slots.fail(error)
            }
        }
        if Engine.parallelLoad {
            DispatchQueue.concurrentPerform(iterations: nProg, execute: load)
        } else {
            for i in 0 ..< nProg { load(i) }
        }
        return try slots.take()
    }

    private static func loadEssential(_ dir: URL, _ units: MLComputeUnits,
                                      _ nProg: Int,
                                      _ report: (@Sendable (Int) -> Void)? = nil)
        throws -> Essential {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        // The tied vocab matrix ships once as pal6 inside head.mlmodelc
        // (lm_head/embed/embed_batched share one blob); older sets keep
        // lm_head + embedding separate. Prefer the merged head. seq=1 embed is
        // skipped entirely when the fp16 sidecar is present (host gather).
        let hasHead = FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("head.mlmodelc").path)
        let hasSidecar = FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("embed_weight.bin").path)
        let emb: MLModel? = hasSidecar ? nil
            : (hasHead
               ? try loadNamed(dir, "head.mlmodelc", cfg, function: "embed",
                               report: report)
               : try loadNamed(dir, "embedding.mlmodelc", cfg, report: report))
        // The decode trunk is the "decode" function of mfNof7 (the seq=1
        // paged-decode graph), sharing one pal6 blob with "prefill".
        let progs = try loadPrograms(dir, units, nProg, function: "decode",
                                     report)
        let flashTile = try loadNamed(dir, "tile.mlmodelc", cfg, report: report)
        let head = hasHead
            ? try loadNamed(dir, "head.mlmodelc", cfg, function: "lm_head",
                            report: report)
            : try loadNamed(dir, "lm_head.mlmodelc", cfg, report: report)
        // Optional vision tower: present only on a VL set, so try? and leave
        // nil on the text-only sets (every shipped 0.8B/2B set today).
        let vision = try? loadNamed(dir, "vision.mlmodelc", cfg, report: report)
        return Essential(embedding: emb, trunk: progs, tile: flashTile,
                         lmHead: head, vision: vision)
    }

    // The common heavy set: the batched embedding gather feeding the carry
    // prefill trunk, loaded in the background at launch. @unchecked Sendable
    // because MLModel crosses the Task boundary (load off-actor, use
    // on-actor).

    public struct Heavy: @unchecked Sendable {
        let embeddingBatched: MLModel?
        let units: MLComputeUnits
    }

    // The mf "prefill" function (S=512 state-carry trunk) + its batched tile,
    // used by every prefill (<=512 one-block, >512 multi-block). Loaded via
    // prepareCarry.

    public struct CarrySet: @unchecked Sendable {
        let carry: [MLModel]?
        let tilePrefill: MLModel?
        // In-graph online-softmax carry tile (P=512): threads m/l/o in-graph
        // so a 512-key flash runs with NO host merge. Weightless (outside the
        // 2fn bundle). nil -> fall back to the host-merged tilePrefill path.
        let carryTile: MLModel?
        let units: MLComputeUnits
    }

    private static func loadHeavy(_ dir: URL, _ units: MLComputeUnits,
                                  _ report: (@Sendable (Int) -> Void)? = nil)
        throws -> Heavy {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        // Batched embedding gather (token_ids[1,S] -> hidden[S,1024]). With
        // the fp16 sidecar it runs in Swift (no model); else it is
        // head.mlmodelc's "embed_batched" function, or standalone
        // embedding_batched on old sets.
        let hasSidecar = FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("embed_weight.bin").path)
        let hasHead = FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("head.mlmodelc").path)
        var embBatch: MLModel? = nil
        if !hasSidecar {
            if hasHead {
                embBatch = try? loadNamed(
                    dir, "head.mlmodelc", cfg, function: "embed_batched",
                    report: report)
            } else {
                let u = dir.appendingPathComponent("embedding_batched.mlmodelc")
                if FileManager.default.fileExists(atPath: u.path) {
                    embBatch = try? MLModel(contentsOf: u, configuration: cfg)
                }
            }
        }
        return Heavy(embeddingBatched: embBatch, units: units)
    }

    // Load the carry/prefill set: the mf "prefill" trunk + its batched tile.
    // Present -> loaded; absent -> nil, so the build falls back to token-by-
    // token ingest. Shares the mf weight blob with the decode trunk.

    private static func loadCarry(_ dir: URL, _ units: MLComputeUnits,
                                  _ nProg: Int,
                                  _ report: (@Sendable (Int) -> Void)? = nil)
        throws -> CarrySet {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        var carry: [MLModel]? = nil
        let mf0URL = dir.appendingPathComponent("mf0of\(nProg).mlmodelc")
        let tilePrefURL = dir.appendingPathComponent("tile_prefill.mlmodelc")
        if FileManager.default.fileExists(atPath: mf0URL.path),
           FileManager.default.fileExists(atPath: tilePrefURL.path) {
            carry = try loadPrograms(dir, units, nProg, function: "prefill",
                                     report)
        }
        let tilePrefill = carry == nil
            ? nil : try? MLModel(contentsOf: tilePrefURL, configuration: cfg)
        // Optional: present -> the in-graph-merge fast path; absent -> the
        // host-merged tilePrefill path.
        let carryURL = dir.appendingPathComponent("tile_carry.mlmodelc")
        let carryTile = FileManager.default.fileExists(atPath: carryURL.path)
            ? try? MLModel(contentsOf: carryURL, configuration: cfg) : nil
        return CarrySet(carry: carry, tilePrefill: tilePrefill,
                        carryTile: carryTile, units: units)
    }

    // Loads the common heavy set off the actor (each program a ~40s first-
    // launch compile) so decode stays live; the actor swaps it in via
    // applyHeavy. The carry set loads separately (prepareCarry).

    public nonisolated func prepareHeavy(_ units: MLComputeUnits) throws -> Heavy {
        try Engine.loadHeavy(modelsDir, units, onCompiledLoC)
    }

    // Swaps the common heavy refs in. Safe mid-conversation: every reader
    // gates on non-nil, so a swap only upgrades the next bounded prefill from
    // ingest to the batched program.

    public func applyHeavy(_ h: Heavy) {
        embeddingBatched = h.embeddingBatched
        log.info("loaded batched-embedding set units \(h.units.rawValue)")
    }

    // NONISOLATED: loads the batched-prefill (carry) set OFF the actor, like
    // prepareHeavy. Awaited by loadCarry during the one-time launch compile.

    public nonisolated func prepareCarry(_ units: MLComputeUnits) throws
        -> CarrySet {
        try Engine.loadCarry(modelsDir, units, nProg, onCompiledLoC)
    }

    // Swaps the carry refs in. Loaded once at launch (chat gated on it), so no
    // conversation is live and the pos == 0 warmup below is over an idle
    // engine.

    public func applyCarry(_ c: CarrySet) {
        carryProgs = c.carry
        tilePrefill = c.tilePrefill
        carryTile = c.carryTile
        // Adopt S from the model's `hidden` input [S, dim], so an S=128/256
        // trunk drives multi-block prefill without a rebuild.
        if let s = c.carry?.first?.modelDescription
            .inputDescriptionsByName["hidden"]?
            .multiArrayConstraint?.shape.first?.intValue {
            blockLen = s
        }
        // Pay the carry program's cold first-execution now: its first predict
        // after compile differs numerically (MultiTurn oracle caught it), so
        // the user's first turn must not be it. Only when idle (pos == 0) --
        // never over a live conversation's carried state.
        if pos == 0 {
            _ = try? prefill([0, 0, 0])
            reset()
        }
        log.info("loaded carry S=\(self.blockLen) set on units \(c.units.rawValue)")
    }

    // Install (or clear) the decode sampler: set -> sample through it, nil ->
    // greedy argmax. The agent path builds and installs it.

    public func useSampler(_ s: Sampler?) {
        sampler = s
    }

    // Test hook: raw fp32 logits of the last ingested position. fillLogitsF32
    // writes the raw model output here, but sample() then mutates logitsF32 IN
    // PLACE (penalties, grammar mask, softmax), so a raw readback holds ONLY
    // under a temperature-0 penalty-free, mask-free sampler -- under which
    // sample mutates nothing. That is the gate's contract: it compares
    // continuation paths without free-running (fp16 ANE would diverge).

    public func lastLogits() -> [Float] { logitsF32 }

    // A restorable checkpoint: paged-attention KV (pools) + GDN recurrence
    // (gdnState) + position, captured together (truncating one leaves the
    // other ahead). @unchecked: read back only on this actor.

    public struct Bookmark: @unchecked Sendable {
        let pos: Int
        let gdn: [String: MLMultiArray]
        let pools: [Int: PagePool]
    }

    // Snapshot the live conversation state (deep copies, so later decode does
    // not mutate the bookmark). Pairs with restore to bookmark / branch /
    // undo.

    public func bookmark() throws -> Bookmark {
        var gdn: [String: MLMultiArray] = [:]
        for (name, a) in gdnState {
            gdn[name] = try Engine.duplicate(a)
        }
        var pl: [Int: PagePool] = [:]
        for (j, p) in pools {
            pl[j] = try p.clone()
        }
        return Bookmark(pos: pos, gdn: gdn, pools: pl)
    }

    // Roll state back to a bookmark (deep copies, so it stays reusable).
    // Clears genSeen so the greedy penalty window restarts from the restored
    // point.

    public func restore(_ b: Bookmark) throws {
        pos = b.pos
        genSeen.removeAll()
        specFlush()
        var gdn: [String: MLMultiArray] = [:]
        for (name, a) in b.gdn {
            gdn[name] = try Engine.duplicate(a)
        }
        gdnState = gdn
        var pl: [Int: PagePool] = [:]
        for (j, p) in b.pools {
            pl[j] = try p.clone()
        }
        pools = pl
    }

    // Deep logical copy of a small fp16 gdnState array. The ANE sometimes
    // returns the source strided/padded (not C-contiguous), so a raw memcpy
    // would copy padding and corrupt the bookmark intermittently; copy
    // honoring strides (as toFloats does). A contiguous source takes the
    // memcpy path.

    private static func duplicate(_ a: MLMultiArray) throws -> MLMultiArray {
        let d = try MLMultiArray(shape: a.shape, dataType: a.dataType)
        let shape = a.shape.map { $0.intValue }
        let strides = a.strides.map { $0.intValue }
        var contiguous = true
        var expect = 1
        for r in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[r] != expect { contiguous = false }
            expect *= shape[r]
        }
        if contiguous {
            a.withUnsafeBytes { s in
                d.withUnsafeMutableBytes { dst, _ in
                    _ = memcpy(dst.baseAddress!, s.baseAddress!, s.count)
                }
            }
        } else {
            let rank = shape.count
            let count = shape.reduce(1, *)
            var idx = [Int](repeating: 0, count: rank)
            a.withUnsafeBytes { s in
                let sp = s.bindMemory(to: UInt16.self)
                d.withUnsafeMutableBytes { dst, _ in
                    let dp = dst.bindMemory(to: UInt16.self)
                    for flat in 0 ..< count {
                        var off = 0
                        for r in 0 ..< rank { off += idx[r] * strides[r] }
                        dp[flat] = sp[off]
                        var r = rank - 1
                        idx[r] += 1
                        while r > 0 && idx[r] == shape[r] {
                            idx[r] = 0
                            r -= 1
                            idx[r] += 1
                        }
                    }
                }
            }
        }
        return d
    }

    // Serialize a Bookmark to Data: pos, each gdn array (name, shape, fp16),
    // each pool (layer, length, compact K/V). Stride-aware for padded fp16.

    public func serialize(_ b: Bookmark) -> Data {
        var out = Data()
        Engine.putInt(&out, b.pos)
        let names = b.gdn.keys.sorted()
        Engine.putInt(&out, names.count)
        for name in names {
            let nameBytes = Array(name.utf8)
            Engine.putInt(&out, nameBytes.count)
            out.append(contentsOf: nameBytes)
            let arr = b.gdn[name]!
            let shape = arr.shape.map { $0.intValue }
            Engine.putInt(&out, shape.count)
            for s in shape { Engine.putInt(&out, s) }
            Engine.putU16(&out, Engine.fp16Flat(arr))
        }
        let layers = b.pools.keys.sorted()
        Engine.putInt(&out, layers.count)
        for layer in layers {
            let pool = b.pools[layer]!
            Engine.putInt(&out, layer)
            Engine.putInt(&out, pool.length)
            let (k, v) = pool.compact()
            Engine.putU16(&out, k)
            Engine.putU16(&out, v)
        }
        return out
    }

    // Reconstruct a Bookmark from serialize()'s Data (contiguous arrays; pools
    // rebuilt via the bulk seed path).

    public func deserialize(_ data: Data) throws -> Bookmark {
        let b = [UInt8](data)
        var p = 0
        let pos = Engine.getInt(b, &p)
        var gdn: [String: MLMultiArray] = [:]
        for _ in 0 ..< Engine.getInt(b, &p) {
            let nameLen = Engine.getInt(b, &p)
            let name = String(decoding: b[p ..< p + nameLen], as: UTF8.self)
            p += nameLen
            var shape: [Int] = []
            for _ in 0 ..< Engine.getInt(b, &p) {
                shape.append(Engine.getInt(b, &p))
            }
            gdn[name] = try Engine.denseFP16(shape, Engine.getU16(b, &p))
        }
        var pools: [Int: PagePool] = [:]
        for _ in 0 ..< Engine.getInt(b, &p) {
            let layer = Engine.getInt(b, &p)
            let length = Engine.getInt(b, &p)
            let k = Engine.getU16(b, &p)
            let v = Engine.getU16(b, &p)
            pools[layer] = try PagePool.fromCompact(k: k, v: v, length: length,
                                                    P: pageP, kvHeads: kvHeads,
                                                    dh: dhAttn)
        }
        return Bookmark(pos: pos, gdn: gdn, pools: pools)
    }

    private static func putInt(_ out: inout Data, _ v: Int) {
        var x = Int64(v).littleEndian
        withUnsafeBytes(of: &x) { out.append(contentsOf: $0) }
    }

    private static func putU16(_ out: inout Data, _ v: [UInt16]) {
        putInt(&out, v.count)
        v.withUnsafeBytes { out.append(contentsOf: $0) }
    }

    private static func getInt(_ b: [UInt8], _ p: inout Int) -> Int {
        var x: Int64 = 0
        withUnsafeMutableBytes(of: &x) { dst in
            for i in 0 ..< 8 { dst[i] = b[p + i] }
        }
        p += 8
        return Int(Int64(littleEndian: x))
    }

    private static func getU16(_ b: [UInt8], _ p: inout Int) -> [UInt16] {
        let n = getInt(b, &p)
        var out = [UInt16](repeating: 0, count: n)
        out.withUnsafeMutableBytes { dst in
            for i in 0 ..< n * 2 { dst[i] = b[p + i] }
        }
        p += n * 2
        return out
    }

    // fp16 MLMultiArray -> [UInt16] in logical (row-major) order, honouring
    // strides so a padded CoreML output is read correctly (as toFloats does).

    private static func fp16Flat(_ a: MLMultiArray) -> [UInt16] {
        let shape = a.shape.map { $0.intValue }
        let strides = a.strides.map { $0.intValue }
        let count = shape.reduce(1, *)
        var out = [UInt16](repeating: 0, count: count)
        var contiguous = true
        var expect = 1
        for r in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[r] != expect { contiguous = false }
            expect *= shape[r]
        }
        a.withUnsafeBytes { s in
            let sp = s.bindMemory(to: UInt16.self)
            if contiguous {
                for i in 0 ..< count { out[i] = sp[i] }
            } else {
                let rank = shape.count
                var idx = [Int](repeating: 0, count: rank)
                for flat in 0 ..< count {
                    var off = 0
                    for r in 0 ..< rank { off += idx[r] * strides[r] }
                    out[flat] = sp[off]
                    var r = rank - 1
                    idx[r] += 1
                    while r > 0 && idx[r] == shape[r] {
                        idx[r] = 0
                        r -= 1
                        idx[r] += 1
                    }
                }
            }
        }
        return out
    }

    private static func denseFP16(_ shape: [Int],
                                  _ vals: [UInt16]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) },
                                 dataType: .float16)
        a.withUnsafeMutableBytes { d, _ in
            let dp = d.bindMemory(to: UInt16.self)
            for i in 0 ..< vals.count { dp[i] = vals[i] }
        }
        return a
    }

    // Reset to a fresh conversation: clear GDN recurrence, drop every layer's
    // paged KV to an empty pool.

    public func reset() {
        pos = 0
        ropeShift = 0
        genSeen.removeAll()
        specFlush()
        for (_, a) in gdnState {
            a.withUnsafeMutableBytes { buf, _ in
                _ = buf.initializeMemory(as: UInt16.self, repeating: 0)
            }
        }
        for j in attnLayers {
            pools[j] = PagePool(P: pageP, kvHeads: kvHeads, dh: dhAttn)
        }
    }

    // Ingest tokens token-by-token WITHOUT resetting (carry conversation state
    // across turns); return the argmax after the last token -- the next token.

    public func ingest(_ ids: [Int32]) throws -> Int32 {
        var last: Int32 = 0
        for id in ids { last = try step(id) }
        return last
    }

    // Prefill a continuation delta onto the CURRENT state (no reset): same
    // position-agnostic routing as a fresh prefill, so a delta wider than one
    // block still gets the batched multi-block path.

    public func ingestBatched(_ ids: [Int32]) throws -> Int32 {
        stop.clear() // a prior turn's Stop must not kill this one
        return try prefillHere(ids)
    }

    private func report(_ s: String, file: String = #fileID, line: Int = #line) {
        Diag.shared.report(s, file: file, line: line)
    }

    // One GADEON_PREFILL_DBG progress line: block i of n (its real-token count
    // L) and the cumulative wall since the block loop started, so a long
    // multi-block 27B prefill reports forward motion instead of one silent
    // stall.
    private func reportBlock(_ i: Int, _ n: Int, _ len: Int, _ t0: UInt64) {
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        report(String(format: "  prefill block %d/%d (L=%d) cumulative %.0fms",
                      i, n, len, ms))
    }

    // Prefill `ids` at the CURRENT position: ONE position-agnostic routine for
    // both a fresh first prompt (via prefill, after reset) and a continuation
    // delta (via ingestBatched, at pos>0). >blockLen runs the batched multi-
    // block trunk, 3...blockLen one batched block, both at pos; anything
    // shorter -- or a build/CPU without the carry set -- is token-by-token.

    private func prefillHere(_ ids: [Int32]) throws -> Int32 {
        specFlush()
        report("prefill n=\(ids.count) pos=\(pos) block=\(blockLen) "
            + "carry=\(carryProgs != nil) tile=\(tilePrefill != nil)")
        let next: Int32
        if ids.count > blockLen, carryProgs != nil, tilePrefill != nil {
            report("path=longDoc (batched multi-block ANE)")
            next = try longDocPrefill(ids)
        } else if (convTaps ... blockLen).contains(ids.count),
                  carryProgs != nil, tilePrefill != nil {
            report("path=carryBlock (batched one-block ANE)")
            next = try carryBlock(ids, p0: pos)
        } else {
            report("path=ingest (token-by-token; CPU or no carry set)")
            next = try ingest(ids)
        }
        return next
    }

    // Fresh conversation: reset, then prefill the first prompt from pos 0.

    public func prefill(_ ids: [Int32]) throws -> Int32 {
        reset()
        stop.clear()
        return try prefillHere(ids)
    }

    // Embed up to blockLen tokens into a fresh [blockLen,dim] buffer (real rows
    // gathered, pad rows zero) via the mmap sidecar, one batched gather, or P
    // seq=1 calls. Pad token_ids are 0 and their rows zeroed, so the
    // downstream pmask GDN no-op sees the original numerics.

    private func embedBlock(_ ids: [Int32]) throws -> MLMultiArray {
        let dim = self.dim // local so the byte-copy closures don't capture self
        let P = ids.count
        let hidden = try Engine.zeros([blockLen, dim])
        if let embedTable {
            hidden.withUnsafeMutableBytes { d, _ in
                let base = d.baseAddress!
                for (row, id) in ids.enumerated() {
                    embedTable.copyRow(id, to: base,
                                       rowOffsetBytes: row * dim * 2)
                }
            }
        } else if let eb = embeddingBatched {
            let toks = try MLMultiArray(
                shape: [1, NSNumber(value: blockLen)], dataType: .int32)
            toks.withUnsafeMutableBytes { buf, _ in
                let d = buf.bindMemory(to: Int32.self)
                for i in 0 ..< blockLen { d[i] = 0 }
                for (i, id) in ids.enumerated() { d[i] = id }
            }
            let e = try run(eb, ["token_ids": toks])
                .featureValue(for: "hidden_states")!.multiArrayValue!
            let st = e.strides.map { $0.intValue }
            e.withUnsafeBytes { s in
                let sp = s.bindMemory(to: UInt16.self)
                hidden.withUnsafeMutableBytes { d, _ in
                    let dp = d.bindMemory(to: UInt16.self)
                    for r in 0 ..< P {
                        for c in 0 ..< dim {
                            dp[r * dim + c] = sp[r * st[0] + c * st[1]]
                        }
                    }
                }
            }
        } else if let embedding {
            for (row, id) in ids.enumerated() {
                let e = try run(embedding, ["token_id": try intScalar(id)])
                    .featureValue(for: "hidden_states")!.multiArrayValue!
                let s0 = e.strides.map { $0.intValue }.last! // last-axis stride
                e.withUnsafeBytes { s in
                    let sp = s.bindMemory(to: UInt16.self)
                    hidden.withUnsafeMutableBytes { d, _ in
                        let dp = d.bindMemory(to: UInt16.self)
                        for c in 0 ..< dim {
                            dp[row * dim + c] = sp[c * s0]
                        }
                    }
                }
            }
        }
        return hidden
    }

    // Prefill a > blockLen prompt as state-threaded blocks: each block runs
    // the carry trunk from the prior block's carried state (carryBlock), so
    // after the last block the GDN state + pools hold the full context and pos
    // = ids.count. Returns the next-token argmax after the last real token.

    private func longDocPrefill(_ ids: [Int32]) throws -> Int32 {
        // Prefill starting at the CURRENT position: 0 for a fresh prefill (the
        // caller reset first), pos>0 when extending a continuation delta wider
        // than one block. Each block's global offset is base + its local
        // start.
        let base = pos
        let n = ids.count
        let nBlocks = (n + blockLen - 1) / blockLen
        let t0 = DispatchTime.now().uptimeNanoseconds
        var arg = try carryBlock(Array(ids[0 ..< blockLen]), p0: base)
        var start = blockLen
        var bi = 1
        if Engine.prefillDbg { reportBlock(bi, nBlocks, blockLen, t0) }
        while start < n {
            if stop.raisedNow { throw EngineError.stopped }
            let end = min(start + blockLen, n)
            if end - start >= convTaps {
                arg = try carryBlock(Array(ids[start ..< end]),
                                     p0: base + start)
            } else {
                // A <3-token trailing remainder can't form a conv_sel chunk;
                // step it token-by-token -- the seq=1 trunk shares gdnState +
                // the pools, so the carried state stays exact.
                for t in ids[start ..< end] { arg = try step(t) }
            }
            bi += 1
            if Engine.prefillDbg { reportBlock(bi, nBlocks, end - start, t0) }
            start = end
        }
        return arg
    }

    // One S-wide carry block at global offset p0. MUTATES live state: threads
    // the real GDN state through the carry trunk and appends L real k/v to the
    // pools, running batched flash over the full history with the causal/pad
    // band (query row s sees keys up to p0+s). A partial block (L<S) marks its
    // L real rows via pmask/conv_sel; pads are no-ops. Advances pos by L,
    // returns the argmax over the last real row.

    private func carryBlock(_ ids: [Int32], p0: Int) throws -> Int32 {
        let hidden = try embedBlock(ids)
        // p0 is the RAW sequence offset -- the causal band keys off it. The
        // RoPE position is p0 + ropeShift: 0 for text, but a prior vision
        // prefill compresses the image's M-RoPE span and leaves ropeShift
        // holding that offset, so text continued past an image ropes at the
        // compressed position while the band stays raw-ordered.
        let cos = try ropeBatched(false, p0 + ropeShift, blockLen)
        let sin = try ropeBatched(true, p0 + ropeShift, blockLen)
        return try runCarryBlock(hidden, cos: cos, sin: sin, p0: p0,
                                 L: ids.count)
    }

    // Shared batched-block trunk loop for a text (carryBlock) or vision
    // (carryBlockVision) block. The two callers differ ONLY in how `hidden`
    // (text embed vs vision-spliced) and cos/sin (1D vs 3D M-RoPE) are built;
    // the GDN threading, k/v append, batched flash, and last-row argmax are
    // identical, so they live here once.

    private func runCarryBlock(_ hidden0: MLMultiArray, cos: MLMultiArray,
                               sin: MLMultiArray, p0: Int, L: Int)
        throws -> Int32 {
        let progs = carryProgs!
        let vtile = tilePrefill!
        let S = blockLen
        let pmask = try pmaskVerify(L, S)
        let convSel = try convSelVerify(L, S)
        var hidden = hidden0
        var attnOut: [Int: MLMultiArray] = [:]
        var attnGate: [Int: MLMultiArray] = [:]
        for p in 0 ..< nProg {
            let (back, gdn, front) = programLayout(p)
            var feed: [String: MLMultiArray] = ["hidden": hidden]
            if let back = back {
                feed["attn_out\(back)"] = attnOut[back]
                feed["gate\(back)"] = attnGate[back]
            }
            if !gdn.isEmpty { feed["pmask"] = pmask; feed["conv_sel"] = convSel }
            for i in gdn {
                feed["conv\(i)"] = gdnState["conv\(i)"]
                feed["rec\(i)"] = gdnState["rec\(i)"]
            }
            if front != nil { feed["cos"] = cos; feed["sin"] = sin }
            let out = try run(progs[p], feed)
            hidden = out.featureValue(for: "hidden_out")!.multiArrayValue!
            for i in gdn {
                gdnState["conv\(i)"] =
                    out.featureValue(for: "conv\(i)_out")!.multiArrayValue!
                gdnState["rec\(i)"] =
                    out.featureValue(for: "rec\(i)_out")!.multiArrayValue!
            }
            if let front = front {
                let (ao, gate) = try batchedAttn(
                    out, front, pools[front]!, vtile, p0, S, L)
                attnOut[front] = ao; attnGate[front] = gate
            }
        }
        pos += L
        let lmShape = lmHead.modelDescription
            .inputDescriptionsByName["hidden_states"]!
            .multiArrayConstraint!.shape.map { $0.intValue }
        let last = try lmRow(hidden, L - 1, lmShape)
        specHidden = last
        let logits = try run(lmHead, ["hidden_states": last])
        return pickToken(logits.featureValue(for: lmHeadOut)!.multiArrayValue!)
    }

    public func supportsVision() -> Bool { visionAvailable }

    public func visionEncodeSeconds() -> Double { visionEncodeSec }

    // Reload the tower on demand (an image turn): resident if already loaded,
    // else a warm load from the launch compile cache. Only called when
    // visionAvailable, so the file is present.

    private func ensureVision() throws -> MLModel {
        if visionModel == nil {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = computeUnits
            visionModel = try Engine.loadNamed(modelsDir, "vision.mlmodelc", cfg)
        }
        return visionModel!
    }

    // Drop the tower's residency (after an image turn, or after the launch
    // compile) so text-only serving does not hold it. The ANE compile stays
    // cached, so the next image reloads warm. No-op on a text-only set.

    public func offloadVision() { visionModel = nil }

    // The tower's baked grid, read from its `patches` input shape [N_PATCH,
    // PATCH_DIM]: N_PATCH is the square patch count, so perSide = sqrt(N_PATCH)
    // and side / merged tokens derive. The single source for the preprocessing
    // grid, so a 256 vs 512 tower needs no Swift change. nil on a text-only
    // set.

    public func visionGrid() -> VisionGrid? {
        var result: VisionGrid? = nil
        if visionAvailable,
           let vm = try? ensureVision(),
           let shape = vm.modelDescription
               .inputDescriptionsByName["patches"]?
               .multiArrayConstraint?.shape.map({ $0.intValue }),
           shape.count == 2 {
            let perSide = Int(Double(shape[0]).squareRoot().rounded())
            result = VisionGrid(perSide: perSide)
        }
        return result
    }

    // Prefill a prompt holding N images (tiling: a thumbnail + detail tiles),
    // MULTI-BLOCK: run the tower on each tile, then fuse-prefill in blockLen
    // blocks (as longDocPrefill does for text). Global 3D M-RoPE positions are
    // computed once over all image spans (each an independent grid, offsets
    // chained like HF get_rope_index); each block splices whichever image feats
    // fall inside it and ropes with its slice of the positions. Attention is
    // sequence-order, so an image may straddle a block boundary. ropeShift
    // carries the compressed next position into seq=1 decode.

    public func prefillVision(_ ids: [Int32], tiles: VisionTiles,
                              imageStarts: [Int], gridH: Int,
                              gridW: Int) throws -> Int32 {
        reset()
        return try prefillVisionCore(ids, tiles: tiles,
                                     imageStarts: imageStarts,
                                     gridH: gridH, gridW: gridW)
    }

    // Vision prefill onto the CURRENT state (NO reset): a later-turn image
    // whose feats splice at the running position, so prior turns stay in the
    // KV. The shared core bases the causal band at pos and the M-RoPE scalar at
    // pos + ropeShift, and accumulates ropeShift -- so this is prefillVision
    // with base = 0 (fresh, after reset) or base = pos (mid-conversation).

    public func prefillVisionExtend(_ ids: [Int32], tiles: VisionTiles,
                                    imageStarts: [Int], gridH: Int,
                                    gridW: Int) throws -> Int32 {
        try prefillVisionCore(ids, tiles: tiles, imageStarts: imageStarts,
                              gridH: gridH, gridW: gridW)
    }

    private func prefillVisionCore(_ ids: [Int32], tiles: VisionTiles,
                                   imageStarts: [Int], gridH: Int,
                                   gridW: Int) throws -> Int32 {
        if !visionAvailable || carryProgs == nil || tilePrefill == nil {
            throw EngineError.missingModel("vision/carry set")
        }
        stop.clear()
        specFlush()
        // base = the raw KV offset the delta appends at (0 fresh, pos when
        // extending); startScalar = the running M-RoPE scalar there (pos +
        // ropeShift, folding in any earlier image's compression).
        let base = pos
        let startScalar = base + ropeShift
        let vt0 = Date()
        var images: [(start: Int, feats: MLMultiArray)] = []
        for (start, tile) in zip(imageStarts, tiles.tiles) {
            if stop.raisedNow { throw EngineError.stopped }
            images.append((start, try encodeVision(tile)))
        }
        // Feats are extracted; the tower is idle for the prefill blocks and
        // decode, so drop its residency now (reloads warm on the next image).
        offloadVision()
        visionEncodeSec = Date().timeIntervalSince(vt0)
        let merged = images.first.map { $0.feats.shape[0].intValue } ?? 0
        let spans = imageStarts.map { (start: $0, gh: gridH, gw: gridW) }
        let n = ids.count
        let plan = Engine.visionPositionsMulti(n, spans, startScalar: startScalar)
        // Decode resumes at the compressed scalar; the delta is constant over
        // the text tail, so this one shift serves decode. Blocks rope
        // explicitly and ignore it. Accumulates: next scalar minus raw end.
        ropeShift = plan.next - (base + n)
        var arg: Int32 = 0
        var start = 0
        while start < n {
            if stop.raisedNow { throw EngineError.stopped }
            var end = min(start + blockLen, n)
            // Never leave a < convTaps trailing block (conv_sel needs >=
            // convTaps real rows); pull the boundary back so the last block
            // keeps enough.
            if end < n && n - end < convTaps { end = n - convTaps }
            arg = try carryBlockVision(ids, images: images, merged: merged,
                                       pos: plan.pos, start: start, end: end,
                                       base: base)
            start = end
        }
        return arg
    }

    // Run the vision tower: patches[N,PATCH_DIM] -> merged feats[MERGED,dim] in
    // the LM embedding space.

    private func encodeVision(_ patches: MLMultiArray) throws -> MLMultiArray {
        let out = try run(ensureVision(), ["patches": patches])
        return out.featureValue(for: "vision_feats")!.multiArrayValue!
    }

    // One fused vision block over the global range [start, end): embed its
    // tokens, splice whichever image feats fall inside it, rope it with its
    // slice of the global 3D positions, and thread it through the shared carry
    // trunk at causal offset p0 = start.

    private func carryBlockVision(_ ids: [Int32],
                                  images: [(start: Int, feats: MLMultiArray)],
                                  merged: Int, pos: [(Int32, Int32, Int32)],
                                  start: Int, end: Int, base: Int)
        throws -> Int32 {
        let L = end - start
        let hidden = try embedBlock(Array(ids[start ..< end]))
        for img in images {
            let lo = max(img.start, start)
            let hi = min(img.start + merged, end)
            if lo < hi {
                try spliceVision(hidden, img.feats, local: lo - start,
                                 feat: lo - img.start, count: hi - lo)
            }
        }
        let bpos = blockPositions(pos, start, end)
        let cos = try mropeBatched(false, bpos)
        let sin = try mropeBatched(true, bpos)
        // The causal band keys off the GLOBAL raw offset base + start, so an
        // extend attends prior turns' KV; the M-RoPE cos/sin already carry the
        // absolute scalar via bpos.
        return try runCarryBlock(hidden, cos: cos, sin: sin,
                                 p0: base + start, L: L)
    }

    // The block's per-row 3D positions padded to blockLen; pad rows are masked,
    // and (0,0,0) keeps their rope finite.

    private func blockPositions(_ global: [(Int32, Int32, Int32)],
                                _ start: Int, _ end: Int)
        -> [(Int32, Int32, Int32)] {
        var out = [(Int32, Int32, Int32)](repeating: (0, 0, 0),
                                          count: blockLen)
        for i in start ..< end { out[i - start] = global[i] }
        return out
    }

    // Overwrite `count` rows of the embed block from local row `local` with
    // vision feats from feature row `feat` (row-major, HF masked_scatter
    // order). feats is [MERGED, dim] fp16, possibly strided; hidden is the
    // contiguous [blockLen, dim] embed block.

    private func spliceVision(_ hidden: MLMultiArray, _ feats: MLMultiArray,
                              local: Int, feat: Int, count: Int) throws {
        let dim = self.dim
        let fs = feats.strides.map { $0.intValue }
        feats.withUnsafeBytes { s in
            let sp = s.bindMemory(to: UInt16.self)
            hidden.withUnsafeMutableBytes { d, _ in
                let dp = d.bindMemory(to: UInt16.self)
                for r in 0 ..< count {
                    for c in 0 ..< dim {
                        dp[(local + r) * dim + c] =
                            sp[(feat + r) * fs[0] + c * fs[1]]
                    }
                }
            }
        }
    }

    // Interleaved M-RoPE table [1,1,S,rot] from per-token 3D positions:
    // frequency j takes axis j % 3 (T/H/W), which reproduces HF's
    // apply_interleaved_mrope [11,11,10] split. Text tokens have t==h==w, so
    // this equals the 1D ropeBatched exactly.

    private func mropeBatched(_ sine: Bool,
                              _ pos: [(Int32, Int32, Int32)]) throws
        -> MLMultiArray {
        let a = try MLMultiArray(
            shape: [1, 1, NSNumber(value: pos.count),
                    NSNumber(value: rot)], dataType: .float16)
        a.withUnsafeMutableBytes { buf, _ in
            let d = buf.bindMemory(to: Float16.self)
            let half = rot / 2
            for s in 0 ..< pos.count {
                let comp = [pos[s].0, pos[s].1, pos[s].2]
                for j in 0 ..< half {
                    let f = Double(comp[j % 3]) * ropeInvFreq[j]
                    let val = Float16(sine ? Foundation.sin(f)
                                           : Foundation.cos(f))
                    d[s * rot + j] = val
                    d[s * rot + j + half] = val
                }
            }
        }
        return a
    }

    // Per-token 3D M-RoPE positions for a prompt with N image spans, porting
    // HF's get_rope_index over multiple images: text tokens advance a shared
    // scalar; each image's merged (gh/merge x gw/merge) grid maps token
    // (r,c) -> (s, s+r, s+c) at the running offset s, then text resumes at
    // s + max(gh,gw)/merge. Spans are ordered by start. `startScalar` seeds
    // the running scalar so a mid-conversation extend continues from
    // the current position (pos + ropeShift), not 0. `next` is the scalar for
    // the row after the last token -- the decode start, via ropeShift.

    static func visionPositionsMulti(_ n: Int,
        _ spans: [(start: Int, gh: Int, gw: Int)],
        startScalar: Int = 0)
        -> (pos: [(Int32, Int32, Int32)], next: Int) {
        let ordered = spans.sorted { $0.start < $1.start }
        var pos = [(Int32, Int32, Int32)](repeating: (0, 0, 0), count: n)
        var cur = startScalar
        var i = 0
        var si = 0
        while i < n {
            if si < ordered.count && i == ordered[si].start {
                let sp = ordered[si]
                let gm = sp.gh / visionMerge
                let gwm = sp.gw / visionMerge
                let nImg = gm * gwm
                let s = Int32(cur)
                for r in 0 ..< nImg {
                    pos[sp.start + r] = (s, s + Int32(r / gwm), s + Int32(r % gwm))
                }
                cur += max(sp.gh, sp.gw) / visionMerge
                i = sp.start + nImg
                si += 1
            } else {
                pos[i] = (Int32(cur), Int32(cur), Int32(cur))
                cur += 1
                i += 1
            }
        }
        return (pos, cur)
    }

    // Advance one token and return the next token. Routes through MTP
    // self-speculative decode when the set ships the drafter (loadMTP): a
    // cycle commits several tokens to the KV/GDN state at once and queues the
    // tail, and queued calls just pop, so the caller's one-token-at-a-time
    // contract holds -- whenever the queue is empty, the state holds every
    // consumed token except the one the caller is about to feed, exactly
    // plain decode's invariant, so the two paths interleave freely. A
    // grammar-masked sampler stays on the plain path (the mask cannot advance
    // mid-cycle). Tokens the caller never feeds back (a stop mid-queue)
    // remain in the KV AND the GDN recurrence as a decoded tail; that is the
    // same class as plain decode's uncommitted tail (which also advanced the
    // GDN through every raw token), and the same mechanism disposes of both:
    // ChatSession's rewind is a full Bookmark restore (deep-copied gdnState +
    // pools + pos), never a KV truncation.

    public func decode(_ token: Int32) throws -> Int32 {
        // The caller must feed back the token the last decode handed out
        // (specLastHanded resets to nil on every state jump, so a fresh seed
        // passes): with tokens queued, an arbitrary token would silently
        // desync the transcript from the already-committed KV.
        assert(specLastHanded == nil || specLastHanded == token,
               "decode fed \(token), expected \(specLastHanded ?? -1)")
        let specReady = mtpFront != nil && verifyProgs != nil
            && specHidden != nil && sampler?.logitMask == nil
        let result: Int32
        if !specQueue.isEmpty {
            result = specQueue.removeFirst()
        } else if specReady {
            specNext = token
            let committed = try decodeSpecStep(Engine.specDrafts)
            specQueue = Array(committed.dropFirst())
            specQueue.append(specNext)
            result = specQueue.removeFirst()
        } else {
            result = try step(token)
        }
        specLastHanded = result
        return result
    }

    // Advance one token, discarding the prediction (used to feed the closing
    // <|im_end|> so the next turn continues from clean state).

    public func feed(_ token: Int32) throws {
        specFlush()
        _ = try step(token)
    }

    // One throwaway prediction to trigger the one-time ANE compile now (e.g.
    // behind the disclaimer), so the user's first message doesn't pay it.

    public func warmup() throws { _ = try prefill([0]); reset() }

    public func position() -> Int { pos }

    // Last-run device for the status bar. pp/tg are no longer tracked here
    // (the ChatSession path surfaces ctx/think/gen); the tuple shape stays for
    // the app's lastStats().device call.
    public func lastStats() -> (device: String, pp: Double, tg: Double) {
        return (computeUnits == .cpuOnly ? "CPU" : "ANE", 0, 0)
    }

    // Stop the in-flight run. Nonisolated so it lands even while a synchronous
    // prefill holds the actor: the prefill loops poll shouldStop() and throw,
    // and the host decode loop cancels its Task. Cleared by the next prefill.

    public nonisolated func requestStop() { stop.raise() }

    public nonisolated func shouldStop() -> Bool { stop.raisedNow }

    // One front attention layer of a batched block: append this chunk's L real
    // k/v to the pool, run batched flash over history + this chunk for all S
    // query rows (pads unread), and pack the result as the next back attn_out.
    // Returns (attn_out [KVH,S,G,DH], gate [S,H_ATTN*DH]).

    private func batchedAttn(_ out: MLFeatureProvider, _ front: Int,
                             _ pool: PagePool, _ tile: MLModel,
                             _ base: Int, _ S: Int, _ L: Int) throws
        -> (MLMultiArray, MLMultiArray) {
        let q = out.featureValue(for: "q\(front)")!.multiArrayValue!
        let kRaw = out.featureValue(for: "k\(front)")!.multiArrayValue!
        let vRaw = out.featureValue(for: "v\(front)")!.multiArrayValue!
        let gate = out.featureValue(for: "gate\(front)")!.multiArrayValue!
        for s in 0 ..< L {
            try pool.append(k: try kvArrayAt(kRaw, s),
                            v: try kvArrayAt(vRaw, s))
        }
        // In-graph carry attention, right-sized 512-key tiles, no host merge:
        //   base==0 -> attend this block's own k/v directly (one call).
        //   base>0  -> chain 512-tiles over the full pool with the m/l/o carry
        //              threaded in-graph (any length).
        // Falls back to the host-merged pool path when carryTile is absent.
        let flat: [Float]
        if let ctile = carryTile {
            if base == 0 {
                flat = try pagedFlashCarrySingle(q: q, K: kRaw, V: vRaw,
                                              tile: ctile, s0GlobalPos: base,
                                            sCount: S, valid: L, G: gqaGroup)
            } else {
                flat = try pagedFlashCarry(q: q, pool: pool, tile: ctile,
                                           s0GlobalPos: base, sCount: S,
                                           tileP: blockLen, G: gqaGroup)
            }
        } else {
            flat = try pagedFlashBatched(q: q, pool: pool, tile: tile,
                                         s0GlobalPos: base, sCount: S,
                                         G: gqaGroup)
        }
        return (try attnOutArrayBatched(flat, S), gate)
    }
    // pmask [1,S,1] fp16: 1 on the L real rows, 0 on the (S-L) pads, so the
    // GDN state lands after exactly the L real tokens.
    private func pmaskVerify(_ L: Int, _ S: Int) throws
        -> MLMultiArray {
        let a = try Engine.zeros([1, S, 1])
        a.withUnsafeMutableBytes { buf, _ in
            let d = buf.bindMemory(to: Float16.self)
            for r in 0 ..< L { d[r] = Float16(1) }
        }
        return a
    }

    // conv_sel [convTaps,S] one-hot: row r picks conv position L-convTaps+r
    // (the trailing 3 real taps), so conv_out is after the last real token.
    // Requires L>=convTaps.

    private func convSelVerify(_ L: Int, _ S: Int) throws
        -> MLMultiArray {
        let a = try Engine.zeros([convTaps, S])
        a.withUnsafeMutableBytes { buf, _ in
            let d = buf.bindMemory(to: Float16.self)
            for r in 0 ..< convTaps {
                d[r * S + (L - convTaps + r)] = Float16(1)
            }
        }
        return a
    }

    private func step(_ token: Int32) throws -> Int32 {
        try stepHidden(token).0
    }

    // (back, gdn_layers, front) for program p: prog p opens with the previous
    // attn layer's back and closes with this one's front; the last prog is
    // the final attn back + norm (no gdn, no front).

    private func programLayout(_ p: Int) -> (Int?, [Int], Int?) {
        let last = attnLayers.count
        let back = p > 0 ? attnLayers[p - 1] : nil
        let front = p < last ? attnLayers[p] : nil
        let gdn = p < last ? [4 * p, 4 * p + 1, 4 * p + 2] : []
        return (back, gdn, front)
    }

    // Advance one token; return (next-token argmax, post-final-norm hidden).
    // Each program runs its GDN layers; a front attn layer emits q/k/v/gate,
    // the host runs paged flash and feeds attn_out to the next program's back.

    private func stepHidden(_ token: Int32) throws -> (Int32, MLMultiArray) {
        let dim = self.dim // local so the byte-copy closures don't capture self
        let t0 = DispatchTime.now().uptimeNanoseconds
        var hidden = try Engine.zeros([1, 1, dim])
        if let embedTable {
            hidden.withUnsafeMutableBytes { d, _ in
                embedTable.copyRow(token, to: d.baseAddress!, rowOffsetBytes: 0)
            }
        } else if let embedding {
            let emb = try run(embedding, ["token_id": intScalar(token)])
            hidden = emb.featureValue(for: "hidden_states")!.multiArrayValue!
        }
        let tEmb = DispatchTime.now().uptimeNanoseconds
        // ropeShift is 0 for text (rope pos == token index); a vision prefill
        // sets it so decode resumes at the image's compressed M-RoPE position.
        let cos = try ropeArray(false, pos + ropeShift)
        let sin = try ropeArray(true, pos + ropeShift)
        var attnOut: [Int: MLMultiArray] = [:]
        var attnGate: [Int: MLMultiArray] = [:]
        for p in 0 ..< nProg {
            let (back, gdn, front) = programLayout(p)
            var feed: [String: MLMultiArray] = ["hidden": hidden]
            if let back = back {
                feed["attn_out\(back)"] = attnOut[back]
                feed["gate\(back)"] = attnGate[back]
            }
            for i in gdn {
                feed["conv\(i)"] = gdnState["conv\(i)"]
                feed["rec\(i)"] = gdnState["rec\(i)"]
            }
            if front != nil { feed["cos"] = cos; feed["sin"] = sin }
            let tG0 = DispatchTime.now().uptimeNanoseconds
            let out = try run(trunk[p], feed)
            hidden = out.featureValue(for: "hidden_out")!.multiArrayValue!
            for i in gdn {
                gdnState["conv\(i)"] =
                    out.featureValue(for: "conv\(i)_out")!.multiArrayValue!
                gdnState["rec\(i)"] =
                    out.featureValue(for: "rec\(i)_out")!.multiArrayValue!
            }
            profGDN += Double(DispatchTime.now().uptimeNanoseconds - tG0)
            if let front = front {
                let tA0 = DispatchTime.now().uptimeNanoseconds
                let q = out.featureValue(for: "q\(front)")!.multiArrayValue!
                let kRaw = out.featureValue(for: "k\(front)")!.multiArrayValue!
                let vRaw = out.featureValue(for: "v\(front)")!.multiArrayValue!
                attnGate[front] =
                    out.featureValue(for: "gate\(front)")!.multiArrayValue!
                let pool = pools[front]!
                try pool.append(k: try kvArray(kRaw), v: try kvArray(vRaw))
                // seq=1 decode attention on the CPU (Accelerate), not the ANE
                // tile: at seq=1 each per-page attention is a tiny GEMV, so
                // this drops ~one ANE dispatch/layer/token (~half the budget),
                // fp32, no accuracy loss. The tile stays the >512 prefill
                // path.
                let flat = hostFlashDecode(q: q, pool: pool, G: gqaGroup)
                attnOut[front] = try attnOutArray(flat)
                profAttn += Double(DispatchTime.now().uptimeNanoseconds - tA0)
            }
        }
        pos += 1
        let tTrunk = DispatchTime.now().uptimeNanoseconds
        let logits = try run(lmHead, ["hidden_states": hidden])
        let tHead = DispatchTime.now().uptimeNanoseconds
        let arg = pickToken(
            logits.featureValue(for: lmHeadOut)!.multiArrayValue!)
        let tArg = DispatchTime.now().uptimeNanoseconds
        profEmb += Double(tEmb - t0); profTrunk += Double(tTrunk - tEmb)
        profHead += Double(tHead - tTrunk); profArg += Double(tArg - tHead)
        profTok += 1
        let profWindow = Engine.decodeProf ? 1 : 32
        if profTok == profWindow {
            if Engine.decodeProf {
                let w = Double(profWindow)
                let ms = { (ns: Double) in String(format: "%.1f", ns / w / 1e6) }
                report("decode/tok(ms) ctx=\(pos) emb=\(ms(profEmb)) "
                    + "trunk=\(ms(profTrunk)) [gdn=\(ms(profGDN)) "
                    + "attn=\(ms(profAttn))] head=\(ms(profHead)) "
                    + "arg=\(ms(profArg)) "
                    + "total=\(ms(profEmb + profTrunk + profHead + profArg))")
            }
            profEmb = 0; profTrunk = 0; profHead = 0; profArg = 0; profTok = 0
            profGDN = 0; profAttn = 0
        }
        specHidden = hidden
        return (arg, hidden)
    }

    // Copy a program's k/v output ([KVH,DH] fp16, possibly strided) into a
    // fresh contiguous [KVH,DH] array PagePool.append can memcpy row-by-row.

    private func kvArray(_ src: MLMultiArray) throws -> MLMultiArray {
        // locals so the byte-copy closure doesn't capture self
        let kvHeads = self.kvHeads
        let dhAttn = self.dhAttn
        let a = try MLMultiArray(
            shape: [NSNumber(value: kvHeads),
                    NSNumber(value: dhAttn)], dataType: .float16)
        let strides = src.strides.map { $0.intValue }
        src.withUnsafeBytes { s in
            let sp = s.bindMemory(to: UInt16.self)
            a.withUnsafeMutableBytes { d, _ in
                let dp = d.bindMemory(to: UInt16.self)
                for h in 0 ..< kvHeads {
                    for k in 0 ..< dhAttn {
                        dp[h * dhAttn + k] =
                            sp[h * strides[0] + k * strides[1]]
                    }
                }
            }
        }
        return a
    }

    // Extract row `r` of a prefill hidden_out (possibly strided) into a fresh
    // lm_head-shaped array, so the last real row's hidden drives the head.

    private func lmRow(_ src: MLMultiArray, _ r: Int, _ shape: [Int]) throws
        -> MLMultiArray {
        // local so the byte-copy closure doesn't capture self
        let dim = self.dim
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) },
                                 dataType: .float16)
        let strides = src.strides.map { $0.intValue }
        src.withUnsafeBytes { s in
            let sp = s.bindMemory(to: UInt16.self)
            a.withUnsafeMutableBytes { d, _ in
                let dp = d.bindMemory(to: UInt16.self)
                for c in 0 ..< dim {
                    dp[c] = sp[r * strides[0] + c * strides[1]]
                }
            }
        }
        return a
    }

    // Pack the flash result ([Float] KVH*G*DH, head kv*G+g) into a [KVH,G,DH]
    // fp16 attn_out; C order matches, so it copies straight through.

    private func attnOutArray(_ flat: [Float]) throws -> MLMultiArray {
        let a = try MLMultiArray(
            shape: [NSNumber(value: kvHeads),
                    NSNumber(value: gqaGroup),
                    NSNumber(value: dhAttn)], dataType: .float16)
        a.withUnsafeMutableBytes { buf, _ in
            let d = buf.bindMemory(to: Float16.self)
            for i in 0 ..< flat.count { d[i] = Float16(flat[i]) }
        }
        return a
    }

    // Per-position RoPE table [1,1,count,rot] for global positions
    // base..base+count-1 (row s = the seq=1 ropeArray at base+s).

    private func ropeBatched(_ sine: Bool, _ base: Int, _ count: Int) throws
        -> MLMultiArray {
        let a = try MLMultiArray(
            shape: [1, 1, NSNumber(value: count), NSNumber(value: rot)],
            dataType: .float16)
        a.withUnsafeMutableBytes { buf, _ in
            let d = buf.bindMemory(to: Float16.self)
            let half = rot / 2
            for s in 0 ..< count {
                for j in 0 ..< half {
                    let f = Double(base + s) * ropeInvFreq[j]
                    let val = Float16(sine ? Foundation.sin(f)
                                           : Foundation.cos(f))
                    d[s * rot + j] = val
                    d[s * rot + j + half] = val
                }
            }
        }
        return a
    }

    // Copy position `s` of a verify k/v output ([KVH,S,DH] fp16, possibly
    // strided) into a fresh contiguous [KVH,DH] array PagePool.append can
    // memcpy row-by-row.

    private func kvArrayAt(_ src: MLMultiArray, _ s: Int) throws -> MLMultiArray {
        // locals so the byte-copy closure doesn't capture self
        let kvHeads = self.kvHeads
        let dhAttn = self.dhAttn
        let a = try MLMultiArray(
            shape: [NSNumber(value: kvHeads),
                    NSNumber(value: dhAttn)], dataType: .float16)
        let strides = src.strides.map { $0.intValue }
        src.withUnsafeBytes { srcBuf in
            let sp = srcBuf.bindMemory(to: UInt16.self)
            a.withUnsafeMutableBytes { d, _ in
                let dp = d.bindMemory(to: UInt16.self)
                for h in 0 ..< kvHeads {
                    for k in 0 ..< dhAttn {
                        dp[h * dhAttn + k] =
                            sp[h * strides[0] + s * strides[1] + k * strides[2]]
                    }
                }
            }
        }
        return a
    }

    // Pack the batched flash result ([Float] KVH*S*G*DH, [KVH,S,G,DH] C order)
    // into an attn_out; the layout matches, so it copies straight through.

    private func attnOutArrayBatched(_ flat: [Float], _ S: Int) throws
        -> MLMultiArray {
        let a = try MLMultiArray(
            shape: [NSNumber(value: kvHeads), NSNumber(value: S),
                    NSNumber(value: gqaGroup),
                    NSNumber(value: dhAttn)], dataType: .float16)
        a.withUnsafeMutableBytes { buf, _ in
            let d = buf.bindMemory(to: Float16.self)
            for i in 0 ..< flat.count { d[i] = Float16(flat[i]) }
        }
        return a
    }

    private func run(_ model: MLModel,
                     _ inputs: [String: MLMultiArray]) throws
        -> MLFeatureProvider {
        var dict: [String: MLFeatureValue] = [:]
        for (k, v) in inputs { dict[k] = MLFeatureValue(multiArray: v) }
        let provider = try MLDictionaryFeatureProvider(dictionary: dict)
        return try model.prediction(from: provider)
    }

    // The next token: sampled through the injected Sampler when one is set,
    // else the greedy argmax (the shipping-app path).

    private func pickToken(_ logits: MLMultiArray) -> Int32 {
        var best: Int32 = 0
        if sampler == nil {
            best = argmax(logits)
        } else {
            fillLogitsF32(logits)
            if let picked = sampler?.sample(&logitsF32) {
                sampler?.accept(picked)
                best = picked
            }
        }
        return best
    }

    // fp16 [1,vocab] lm_head logits -> the reused fp32 scratch, one vImage
    // half->single pass. The [1,vocab] output is contiguous (stride 1).

    private func fillLogitsF32(_ logits: MLMultiArray) {
        logits.withUnsafeBytes { src in
            let raw = UnsafeMutableRawPointer(mutating: src.baseAddress!)
            var sBuf = vImage_Buffer(data: raw, height: 1,
                                     width: vImagePixelCount(vocab),
                                     rowBytes: vocab * 2)
            logitsF32.withUnsafeMutableBufferPointer { fb in
                var dBuf = vImage_Buffer(data: fb.baseAddress!, height: 1,
                                         width: vImagePixelCount(vocab),
                                         rowBytes: vocab * 4)
                _ = vImageConvert_Planar16FtoPlanarF(&sBuf, &dBuf, 0)
            }
        }
    }

    // Greedy vocab argmax with the full-context repetition penalty (else a
    // chosen token wins fp16 ties forever). vDSP_maxvi takes the lowest index
    // on ties, matching the first-occurrence tie-break.

    private func argmax(_ logits: MLMultiArray) -> Int32 {
        fillLogitsF32(logits)
        var best: Int32 = 0
        logitsF32.withUnsafeMutableBufferPointer { fb in
            for t in genSeen {
                let v = fb[Int(t)]
                fb[Int(t)] = v > 0 ? v / Engine.repPenalty
                                   : v * Engine.repPenalty
            }
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(fb.baseAddress!, 1, &maxVal, &maxIdx,
                       vDSP_Length(vocab))
            best = Int32(maxIdx)
        }
        genSeen.insert(best)
        return best
    }

    private func intScalar(_ v: Int32) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [1, 1], dataType: .int32)
        a.withUnsafeMutableBytes { buf, _ in
            buf.bindMemory(to: Int32.self)[0] = v
        }
        return a
    }

    // Per-position RoPE (computed, not table-indexed) so context is unbounded.
    // cos when `sine` is false, else sin.

    private func ropeArray(_ sine: Bool, _ p: Int) throws -> MLMultiArray {
        let a = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: rot)], dataType: .float16)
        a.withUnsafeMutableBytes { buf, _ in
            let d = buf.bindMemory(to: Float16.self)
            let half = rot / 2
            for j in 0 ..< half {
                let f = Double(p) * ropeInvFreq[j]
                let val = Float16(sine ? Foundation.sin(f) : Foundation.cos(f))
                d[j] = val; d[j + half] = val
            }
        }
        return a
    }

    private static func zeros(_ shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(
            shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        a.withUnsafeMutableBytes { buf, _ in
            _ = buf.initializeMemory(as: UInt16.self, repeating: 0)
        }
        return a
    }

    // ===================== MTP self-speculative decode =====================
    // Draft n tokens with the one-attention-layer MTP head, verify them in ONE
    // base pass (the verify trunk), accept the matching prefix + 1 bonus,
    // then roll the base GDN state back to the accepted length on the host
    // (no 2nd trunk pass).

    private var specNext: Int32 = 0
    // Committed-but-not-yet-handed-out tokens from the last spec cycle;
    // decode pops one per call so the per-token caller contract holds.
    private var specQueue: [Int32] = []
    // The token decode() handed out last, to assert the caller feeds it back
    // (nil after any state jump, when any next token is legitimate).
    private var specLastHanded: Int32? = nil
    // Draft count for the decode path: n=2 measured best (fewer unbatchable
    // draft heads, more full-accepts); --bench-mtp passes its own n.
    public static let specDrafts = 2
    // Per-phase spec-dec timing accumulators (ns),
    // for the batched-head decision.
    private var pDraft = 0.0, pVerify = 0.0, pAccept = 0.0, pRest = 0.0
    private var pCycles = 0

    // Averaged per-cycle ms of each spec phase (draft = MTP + draft heads;
    // verify = the trunk pass;
    // accept = the per-row verify heads (~m, early-stop);
    // rest = rollback + KV maintenance).

    public func specProf() -> (draft: Double, verify: Double, accept: Double,
                               rest: Double, cycles: Int) {
        let n = Double(max(pCycles, 1)) * 1e6
        return (pDraft / n, pVerify / n, pAccept / n, pRest / n, pCycles)
    }

    // AUTODETECT: load the drafter (mtp_front/back) + the verify trunk +
    // the batched verify head IF the set ships them. On current sets the
    // verify trunk is the mf programs' "verify" FUNCTION and the batched head
    // is head.mlmodelc's "lm_head_batched" FUNCTION -- both share their
    // package's one pal6 blob, so MTP adds only the ~0.2GB drafter, not a
    // second weight copy. Older test sets ship standalone verify{i}of{n} /
    // mtp_lmhead files instead; those load when present. A set without
    // mtp_front is a silent no-op -- mtpReady() stays false and the engine
    // decodes plainly. Graceful: a present-but-broken set logs and falls back
    // to plain rather than throwing.

    public func loadMTP() {
        let dir = modelsDir
        let fm = FileManager.default
        let present = fm.fileExists(
            atPath: dir.appendingPathComponent("mtp_front.mlmodelc").path)
        let standalone = fm.fileExists(atPath:
            dir.appendingPathComponent("verify0of\(nProg).mlmodelc").path)
        if present && mtpFront == nil {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = computeUnits
            // Report each compile so the app's Optimizing bar covers the MTP
            // set too (compileTotalLoC counts it in the denominator).
            let report = onCompiledLoC
            do {
                let front = try Engine.loadNamed(dir, "mtp_front.mlmodelc",
                                                 cfg, report: report)
                let back = try Engine.loadNamed(dir, "mtp_back.mlmodelc",
                                                cfg, report: report)
                var vp: [MLModel] = []
                for i in 0 ..< nProg {
                    vp.append(standalone
                        ? try Engine.loadNamed(
                            dir, "verify\(i)of\(nProg).mlmodelc", cfg,
                            report: report)
                        : try Engine.loadNamed(
                            dir, "mf\(i)of\(nProg).mlmodelc", cfg,
                            function: "verify", report: report))
                }
                mtpFront = front
                mtpBack = back
                verifyProgs = vp
                verifyS = try Engine.require(Engine.hiddenWidth(vp[0]),
                                             "verify hidden width")
                mtpPool = PagePool(P: pageP, kvHeads: kvHeads, dh: dhAttn)
                let hasLmhead = fm.fileExists(atPath:
                    dir.appendingPathComponent("mtp_lmhead.mlmodelc").path)
                lmHeadBatched = hasLmhead
                    ? try? Engine.loadNamed(dir, "mtp_lmhead.mlmodelc", cfg,
                                            report: report)
                    : try? Engine.loadNamed(dir, "head.mlmodelc", cfg,
                                            function: "lm_head_batched",
                                            report: report)
                // The batched head is fed the verify trunk's rows verbatim, so
                // a head baked to a different width cannot serve them; drop it
                // for the per-row path rather than fail at the first accept.
                // Only a head that DECLARES a conflicting width is dropped: one
                // that declares none is left alone, since dropping it would
                // cost the batched read on nothing more than a missing shape.
                if let head = lmHeadBatched, let wide = Engine.hiddenWidth(head),
                   wide != verifyS {
                    log.warning("""
                        batched verify head is \(wide) wide, verify trunk is \
                        \(self.verifyS); using per-row heads
                        """)
                    lmHeadBatched = nil
                }
                log.info("""
                    MTP drafter + verify trunk loaded (\(self.nProg) progs, \
                    standalone \(standalone), S \(self.verifyS), \
                    batched-head \(self.lmHeadBatched != nil))
                    """)
            } catch {
                log.warning("""
                    MTP tensors present but failed to load; plain decode: \
                    \(error.localizedDescription, privacy: .public)
                    """)
            }
        }
    }

    public func mtpReady() -> Bool { mtpFront != nil && verifyProgs != nil }

    // The loaded verify trunk's chunk width. A cycle verifies n+1 rows, so it
    // caps the draft count at verifyChunk()-1; a caller that asks for more is
    // asking the trunk to hold rows it was not compiled for.

    public func verifyChunk() -> Int { verifyS }

    // Committed-but-undelivered spec tokens: already in the KV/GDN state,
    // not yet handed out by decode(). The turn loop checks this before
    // injecting </think>, so markup never lands after invisible tokens.
    public func specQueueCount() -> Int { specQueue.count }

    // Plain (no repetition-penalty) next-token argmax of the current
    // post-prefill/post-decode hidden -- the deterministic base function
    // the MTP verify checks against. For the correctness eval.
    public func plainNext() throws -> Int32 { try argmaxLogits(specHidden!) }

    // Advance one token via the plain seq=1 decode trunk, returning the PLAIN
    // argmax (no penalty), to build a base-greedy reference sequence.
    public func decodePlain(_ token: Int32) throws -> Int32 {
        let (_, hidden) = try stepHidden(token)
        return try argmaxLogits(hidden)
    }

    // The next token after a prefill seeds the first draft; the drafter's KV
    // starts empty and is maintained (from base hiddens) as tokens commit.
    public func seedSpec(_ tNext: Int32) {
        specNext = tNext
        specQueue.removeAll()
        specLastHanded = nil
        mtpPool = PagePool(P: pageP, kvHeads: kvHeads, dh: dhAttn)
        pDraft = 0; pVerify = 0; pAccept = 0; pRest = 0; pCycles = 0
    }

    // Drop the spec pipeline across any state jump (reset / restore / prefill
    // / feed): queued tokens belong to the pre-jump KV, and the drafter KV no
    // longer matches the base position, so both restart at the new position.
    // specHidden goes too -- after a jump it points at a hidden from the wrong
    // position; the next prefill/step refills it, and until then spec decode
    // stays off (specReady requires it non-nil) and plainNext crashes loudly
    // instead of silently mis-verifying.

    private func specFlush() {
        specQueue.removeAll()
        specHidden = nil
        specLastHanded = nil
        if mtpPool != nil {
            mtpPool = PagePool(P: pageP, kvHeads: kvHeads, dh: dhAttn)
        }
    }

    // fp16 MLMultiArray -> row-major [Float] (contiguous fast path, strided
    // fallback), so the host rollback reads the verify ingredients.

    private func floats(_ a: MLMultiArray) -> [Float] {
        let shape = a.shape.map { $0.intValue }
        let strides = a.strides.map { $0.intValue }
        let count = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        var contiguous = true
        var expect = 1
        for r in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[r] != expect { contiguous = false }
            expect *= shape[r]
        }
        a.withUnsafeBytes { raw in
            let sp = raw.bindMemory(to: UInt16.self)
            if contiguous {
                out.withUnsafeMutableBufferPointer { ob in
                    var s = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: sp.baseAddress!),
                        height: 1, width: vImagePixelCount(count),
                        rowBytes: count * 2)
                    var d = vImage_Buffer(data: ob.baseAddress!, height: 1,
                                          width: vImagePixelCount(count),
                                          rowBytes: count * 4)
                    _ = vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
                }
            } else {
                let rank = shape.count
                var idx = [Int](repeating: 0, count: rank)
                for flat in 0 ..< count {
                    var off = 0
                    for r in 0 ..< rank { off += idx[r] * strides[r] }
                    out[flat] = Float(Float16(bitPattern: sp[off]))
                    var r = rank - 1
                    idx[r] += 1
                    while r > 0 && idx[r] == shape[r] {
                        idx[r] = 0; r -= 1; idx[r] += 1
                    }
                }
            }
        }
        return out
    }

    private func fp16Array(_ shape: [Int], _ v: [Float]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) },
                                 dataType: .float16)
        a.withUnsafeMutableBytes { d, _ in
            let dp = d.bindMemory(to: Float16.self)
            for i in 0 ..< v.count { dp[i] = Float16(v[i]) }
        }
        return a
    }

    // Write one token's fp16 embedding into `dst` at `rowOffsetBytes`, via the
    // mmap sidecar or (no sidecar) the embedding MLModel -- matching
    // embedBlock.

    private func writeEmbed(_ id: Int32, into dst: MLMultiArray,
                            rowOffsetBytes: Int) throws {
        let dim = self.dim
        if let embedTable {
            dst.withUnsafeMutableBytes { d, _ in
                embedTable.copyRow(id, to: d.baseAddress!,
                                   rowOffsetBytes: rowOffsetBytes)
            }
        } else if let embedding {
            let e = try run(embedding, ["token_id": try intScalar(id)])
                .featureValue(for: "hidden_states")!.multiArrayValue!
            let s0 = e.strides.map { $0.intValue }.last!
            e.withUnsafeBytes { s in
                let sp = s.bindMemory(to: UInt16.self)
                dst.withUnsafeMutableBytes { d, _ in
                    let dp = (d.baseAddress! + rowOffsetBytes)
                        .bindMemory(to: UInt16.self, capacity: dim)
                    for c in 0 ..< dim { dp[c] = sp[c * s0] }
                }
            }
        }
    }

    // Embed `ids` into a fresh [S, dim] block (real rows gathered, pads zero).

    private func embedRows(_ ids: [Int32], _ S: Int) throws -> MLMultiArray {
        let h = try Engine.zeros([S, dim])
        for (row, id) in ids.enumerated() {
            try writeEmbed(id, into: h, rowOffsetBytes: row * dim * 2)
        }
        return h
    }

    // One drafter step: MTP front -> host flash over mtpPool -> back -> h_nextn
    // [1,1,dim]. Appends this token's k/v to mtpPool; rope at position `pos`.

    private func mtpForward(_ token: Int32, _ hidden: MLMultiArray,
                            pos: Int) throws -> MLMultiArray {
        let embed = try Engine.zeros([1, 1, dim])
        try writeEmbed(token, into: embed, rowOffsetBytes: 0)
        let cos = try ropeArray(false, pos)
        let sin = try ropeArray(true, pos)
        let f = try run(mtpFront!, ["embed": embed, "hidden": hidden,
                                    "cos": cos, "sin": sin])
        let q = f.featureValue(for: "q")!.multiArrayValue!
        try mtpPool!.append(
            k: try kvArray(f.featureValue(for: "k")!.multiArrayValue!),
            v: try kvArray(f.featureValue(for: "v")!.multiArrayValue!))
        let attnOut = try attnOutArray(hostFlashDecode(q: q, pool: mtpPool!,
                                                       G: gqaGroup))
        let b = try run(mtpBack!, [
            "resid": f.featureValue(for: "resid")!.multiArrayValue!,
            "attn_out": attnOut,
            "gate": f.featureValue(for: "gate")!.multiArrayValue!])
        return b.featureValue(for: "h_nextn")!.multiArrayValue!
    }

    // Plain (no repetition-penalty) vocab argmax over an lm_head-shaped hidden
    // ([1,1,dim]) -- the deterministic pick the greedy spec paths and the
    // correctness eval compare against.

    private func argmaxLogits(_ hidden: MLMultiArray) throws -> Int32 {
        var none: Sampler? = nil
        return try pickLogits(hidden, using: &none)
    }

    // lm_head over an lm-shaped hidden, then a pick from the filled scratch.

    private func pickLogits(_ hidden: MLMultiArray,
                            using via: inout Sampler?) throws -> Int32 {
        let logits = try run(lmHead, ["hidden_states": hidden])
        fillLogitsF32(logits.featureValue(for: lmHeadOut)!.multiArrayValue!)
        return pickF32(using: &via)
    }

    // Pick from the filled logitsF32: sample + accept through `via` when
    // present (penalties and RNG advance exactly as sequential decode), else
    // the raw vocab argmax.

    private func pickF32(using via: inout Sampler?) -> Int32 {
        var best: Int32 = 0
        if let picked = via?.sample(&logitsF32) {
            via?.accept(picked)
            best = picked
        } else {
            logitsF32.withUnsafeMutableBufferPointer { fb in
                var mv: Float = 0
                var mi: vDSP_Length = 0
                vDSP_maxvi(fb.baseAddress!, 1, &mv, &mi, vDSP_Length(vocab))
                best = Int32(mi)
            }
        }
        return best
    }

    // Fill the fp32 scratch from row r of a batched-head logits [S,vocab]
    // fp16 (row-contiguous).

    private func fillRowF32(_ lg: MLMultiArray, _ r: Int) {
        let vocab = self.vocab
        let rowStride = lg.strides[0].intValue
        lg.withUnsafeBytes { raw in
            let sp = raw.bindMemory(to: UInt16.self)
            var sBuf = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: sp.baseAddress! + r * rowStride),
                height: 1, width: vImagePixelCount(vocab), rowBytes: vocab * 2)
            logitsF32.withUnsafeMutableBufferPointer { fb in
                var dBuf = vImage_Buffer(data: fb.baseAddress!, height: 1,
                                         width: vImagePixelCount(vocab),
                                         rowBytes: vocab * 4)
                _ = vImageConvert_Planar16FtoPlanarF(&sBuf, &dBuf, 0)
            }
        }
    }

    // Verify-row r pick: batched-head logits when present (no extra ANE
    // call), else a per-row seq=1 head call. Picked through the TURN SAMPLER,
    // so every committed token -- accepted draft or bonus -- is that
    // sampler's own pick with its penalties and RNG advanced in commit order;
    // the spec stream is therefore distributed identically to plain sampled
    // decode. Raw argmax when no sampler is installed (the bench / eval
    // config).

    private func specRowPick(_ rows: MLMultiArray, _ r: Int, _ lmShape: [Int],
                             _ batched: MLMultiArray?) throws -> Int32 {
        let out: Int32
        if let batched = batched {
            fillRowF32(batched, r)
            out = pickF32(using: &sampler)
        } else {
            out = try pickLogits(try lmRow(rows, r, lmShape), using: &sampler)
        }
        return out
    }

    // Run the verify trunk over `ids` (real rows, pmask), returning the
    // per-row hiddens [S,dim] and, per GDN layer, the rollback ingredients.
    // Appends the L real k/v to the base attention pools; does NOT mutate
    // gdnState (the host rolls it back after acceptance).

    private func verifyPass(_ ids: [Int32], p0: Int) throws
        -> (MLMultiArray, [Int: (delta: [Float], k: [Float], gcum: [Float],
                                 conv: [Float])],
            [Int: (rec: MLMultiArray, conv: MLMultiArray)]) {
        let S = verifyS
        let L = ids.count
        // embedRows writes L rows into an S-row block, so a chunk wider than
        // the trunk was compiled for would run off the end of it.
        precondition(L <= S, "verify chunk holds \(S) rows, got \(L)")
        var hidden = try embedRows(ids, S)
        let cos = try ropeBatched(false, p0 + ropeShift, S)
        let sin = try ropeBatched(true, p0 + ropeShift, S)
        let pmask = try pmaskVerify(L, S)
        let convSel = try convSelVerify(L, S)
        var ingr: [Int: (delta: [Float], k: [Float], gcum: [Float],
                         conv: [Float])] = [:]
        var stateOut: [Int: (rec: MLMultiArray, conv: MLMultiArray)] = [:]
        var attnOut: [Int: MLMultiArray] = [:]
        var attnGate: [Int: MLMultiArray] = [:]
        let progs = verifyProgs!
        for p in 0 ..< nProg {
            let (back, gdn, front) = programLayout(p)
            var feed: [String: MLMultiArray] = ["hidden": hidden]
            if let back = back {
                feed["attn_out\(back)"] = attnOut[back]
                feed["gate\(back)"] = attnGate[back]
            }
            if !gdn.isEmpty { feed["pmask"] = pmask; feed["conv_sel"] = convSel }
            for i in gdn {
                feed["conv\(i)"] = gdnState["conv\(i)"]
                feed["rec\(i)"] = gdnState["rec\(i)"]
            }
            if front != nil { feed["cos"] = cos; feed["sin"] = sin }
            let out = try run(progs[p], feed)
            hidden = out.featureValue(for: "hidden_out")!.multiArrayValue!
            for i in gdn {
                func vec(_ name: String) -> [Float] {
                    floats(out.featureValue(for: "\(name)\(i)")!
                        .multiArrayValue!)
                }
                func arr(_ name: String) -> MLMultiArray {
                    out.featureValue(for: "\(name)\(i)_out")!.multiArrayValue!
                }
                ingr[i] = (vec("delta"), vec("k"), vec("gcum"),
                           vec("convproj"))
                // The verify's own end-state after all L real tokens: EXACT
                // for a full accept (m == L), skipping the host reconstruction.
                stateOut[i] = (arr("rec"), arr("conv"))
            }
            if let front = front {
                let (ao, gate) = try verifyAttn(out, front, pools[front]!, S, L)
                attnOut[front] = ao
                attnGate[front] = gate
            }
        }
        return (hidden, ingr, stateOut)
    }

    // Verify-chunk attention on the host: append each of the L real rows' k/v
    // to the pool and flash that row's query over the pool so far (causal),
    // so row s attends history + verify rows 0..s. Tile-free (the batched
    // ANE tiles are sized for the S=128 carry, not the verify chunk).
    // Returns attn_out [KVH,S,G,DH] + gate.

    private func verifyAttn(_ out: MLFeatureProvider, _ front: Int,
                            _ pool: PagePool, _ S: Int, _ L: Int) throws
        -> (MLMultiArray, MLMultiArray) {
        let q = out.featureValue(for: "q\(front)")!.multiArrayValue!
        let kRaw = out.featureValue(for: "k\(front)")!.multiArrayValue!
        let vRaw = out.featureValue(for: "v\(front)")!.multiArrayValue!
        let gate = out.featureValue(for: "gate\(front)")!.multiArrayValue!
        let kvHeads = self.kvHeads, gqaGroup = self.gqaGroup, dhAttn = self.dhAttn
        let ao = try Engine.zeros([kvHeads, S, gqaGroup, dhAttn])
        for s in 0 ..< L {
            try pool.append(k: try kvArrayAt(kRaw, s), v: try kvArrayAt(vRaw, s))
            let flat = hostFlashDecode(q: try qRow(q, s), pool: pool, G: gqaGroup)
            ao.withUnsafeMutableBytes { d, _ in
                let dp = d.bindMemory(to: Float16.self)
                for kvh in 0 ..< kvHeads {
                    for g in 0 ..< gqaGroup {
                        let base = ((kvh * S + s) * gqaGroup + g) * dhAttn
                        let src = (kvh * gqaGroup + g) * dhAttn
                        for dd in 0 ..< dhAttn {
                            dp[base + dd] = Float16(flat[src + dd])
                        }
                    }
                }
            }
        }
        return (ao, gate)
    }

    // Extract query row s of a verify front's q [KVH,S,G,DH] into a seq=1
    // [KVH,G,DH] for hostFlashDecode.

    private func qRow(_ q: MLMultiArray, _ s: Int) throws -> MLMultiArray {
        let kvHeads = self.kvHeads, gqaGroup = self.gqaGroup, dhAttn = self.dhAttn
        let a = try MLMultiArray(
            shape: [kvHeads, gqaGroup, dhAttn].map { NSNumber(value: $0) },
            dataType: .float16)
        let st = q.strides.map { $0.intValue }
        q.withUnsafeBytes { src in
            let sp = src.bindMemory(to: UInt16.self)
            a.withUnsafeMutableBytes { d, _ in
                let dp = d.bindMemory(to: UInt16.self)
                for kvh in 0 ..< kvHeads {
                    for g in 0 ..< gqaGroup {
                        for dd in 0 ..< dhAttn {
                            let src = kvh * st[0] + s * st[1] +
                                      g * st[2] + dd * st[3]
                            dp[(kvh * gqaGroup + g) * dhAttn + dd] = sp[src]
                        }
                    }
                }
            }
        }
        return a
    }

    // Host GDN rollback to the accepted length m: rec via MtpRollback, conv
    // from the committed convproj rows. Writes gdnState in place.

    private func rollbackGDN(_ ingr: [Int: (delta: [Float], k: [Float],
                                            gcum: [Float], conv: [Float])],
                             _ stateOut: [Int: (rec: MLMultiArray,
                                                conv: MLMultiArray)],
                             m: Int, L: Int) throws {
        // The verify trunk is single-chunk, so its chunk width IS its S: the
        // rollback ingredients arrive C-per-head and reading them at any other
        // stride walks off the end of the array.
        let C = verifyS
        for (i, ing) in ingr {
            if m == L, let so = stateOut[i] {
                // Full accept: the verify already advanced the state to exactly
                // m==L tokens, so use its own rec/conv (no host
                // reconstruction).
                gdnState["rec\(i)"] = so.rec
                gdnState["conv\(i)"] = so.conv
            } else {
                // rs [1,H,DK,DV]
                let rs = gdnState["rec\(i)"]!.shape.map { $0.intValue }
                let (hh, dk, dv) = (rs[1], rs[2], rs[3])
                let sm = MtpRollback.stateAfter(
                    m: m, H: hh, DK: dk, DV: dv, C: C,
                    s0: floats(gdnState["rec\(i)"]!),
                    delta: ing.delta, k: ing.k, gcum: ing.gcum)
                // cs [1,CD,taps]
                let cs = gdnState["conv\(i)"]!.shape.map { $0.intValue }
                let preConv = floats(gdnState["conv\(i)"]!)
                gdnState["rec\(i)"] = try fp16Array([1, hh, dk, dv], sm)
                gdnState["conv\(i)"] = try convFromProj(ing.conv,
                                            preConv: preConv, m: m,
                                            convDim: cs[1], taps: cs[2])
            }
        }
    }

    // conv_state after m committed tokens, laid [1,CD,taps] ([0,c,t] holds
    // position m-taps+t). Positions >= 0 come from this chunk's convproj rows;
    // negative positions (m < taps) come from the pre-verify conv tail
    // preConv ([1,CD,taps], tap index oldest..newest), so a 1-token commit
    // keeps the two carried taps.

    private func convFromProj(_ proj: [Float], preConv: [Float], m: Int,
                              convDim: Int, taps: Int) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [1, NSNumber(value: convDim),
                                         NSNumber(value: taps)],
                                 dataType: .float16)
        a.withUnsafeMutableBytes { d, _ in
            let dp = d.bindMemory(to: Float16.self)
            for t in 0 ..< taps {
                let pos = m - taps + t
                for c in 0 ..< convDim {
                    let val = pos >= 0 ? proj[pos * convDim + c]
                                       : preConv[c * taps + (m + t)]
                    dp[c * taps + t] = Float16(val)
                }
            }
        }
        return a
    }

    // One spec-dec cycle: draft n, verify, accept prefix + bonus,
    // host-rollback, commit.
    // Returns the committed tokens (m = accepted + 1, >= 1).

    public func decodeSpecStep(_ n: Int) throws -> [Int32] {
        let p0 = pos
        let hCur = specHidden!
        let tok = specNext
        // Draft n against the PERSISTENT drafter KV (committed history so far),
        // so each draft attends the generated context, not just hCur.
        let mtpBase = mtpPool!.length
        let t0 = DispatchTime.now().uptimeNanoseconds
        // Draft through a VALUE COPY of the turn sampler: the same RNG stream
        // and penalty windows the verify picks below will consume, so at
        // matched randomness the drafter's pick tends to equal the verify
        // row's, keeping acceptance high under temperature. The copy is
        // discarded -- only verify picks advance the real sampler.
        var draftSampler = sampler
        var drafts: [Int32] = []
        var h = hCur
        var dt = tok
        for i in 0 ..< n {
            let hn = try mtpForward(dt, h, pos: p0 + i)
            drafts.append(try pickLogits(hn, using: &draftSampler))
            h = hn
            dt = drafts[i]
        }
        let t1 = DispatchTime.now().uptimeNanoseconds
        let (rows, ingr, stateOut) = try verifyPass([tok] + drafts, p0: p0)
        let t2 = DispatchTime.now().uptimeNanoseconds
        let lmShape = lmHead.modelDescription
            .inputDescriptionsByName["hidden_states"]!
            .multiArrayConstraint!.shape.map { $0.intValue }
        // ONE batched head reads the vocab weight once for all rows; the accept
        // loop then early-stops over host picks (vs ~m sequential head calls).
        let batched = try lmHeadBatched.map {
            try run($0, ["hidden": rows]).featureValue(for: "logits")!.multiArrayValue!
        }
        var r = 0
        var a = try specRowPick(rows, 0, lmShape, batched)
        while r < n && a == drafts[r] {
            r += 1
            a = try specRowPick(rows, r, lmShape, batched)
        }
        let t3 = DispatchTime.now().uptimeNanoseconds
        let accepted = r
        let m = accepted + 1
        try rollbackGDN(ingr, stateOut, m: m, L: n + 1)
        pos = p0 + m
        for j in attnLayers { pools[j]?.truncate(to: p0 + m) }
        // Maintain the drafter KV: drop the n speculative draft rows,
        // re-append the m committed positions from the BASE hiddens
        // (position j gets h_{j-1} + embed(x_j)) -- the catch-up decode
        // that makes drafts match training.
        mtpPool!.truncate(to: mtpBase)
        for j in 0 ..< m {
            let ctok = j == 0 ? tok : drafts[j - 1]
            let bh = j == 0 ? hCur : try rowTo3D(rows, j - 1)
            _ = try mtpForward(ctok, bh, pos: p0 + j)
        }
        specHidden = try lmRow(rows, accepted, lmShape)
        specNext = a
        if Engine.specTrace {
            let committed = [tok] + drafts.prefix(accepted)
            report("[spec] p0=\(p0) L=\(n + 1) drafts=\(drafts) "
                + "acc=\(accepted) m=\(m) committed=\(committed) bonus=\(a)")
        }
        let t4 = DispatchTime.now().uptimeNanoseconds
        pDraft += Double(t1 - t0); pVerify += Double(t2 - t1)
        pAccept += Double(t3 - t2); pRest += Double(t4 - t3); pCycles += 1
        var committed = [tok]
        committed.append(contentsOf: drafts.prefix(accepted))
        return committed
    }

    // Extract verify-chunk row r ([S,dim]) into an lm/mtp hidden [1,1,dim].

    private func rowTo3D(_ rows: MLMultiArray, _ r: Int) throws -> MLMultiArray {
        let dim = self.dim
        let a = try Engine.zeros([1, 1, dim])
        let st = rows.strides.map { $0.intValue }
        rows.withUnsafeBytes { s in
            let sp = s.bindMemory(to: UInt16.self)
            a.withUnsafeMutableBytes { d, _ in
                let dp = d.bindMemory(to: UInt16.self)
                for c in 0 ..< dim { dp[c] = sp[r * st[0] + c * st[1]] }
            }
        }
        return a
    }
}

// Deferred optimization: the ANE compile cache (Library/Caches e5bundlecache)
// is dropped on an Xcode re-deploy and can be purged by iOS under storage
// pressure, forcing a full recompile. If that ever bites real users, snapshot
// the e5bundlecache subtree to Application Support after a successful warmup()
// (APFS clonefile, COW so ~free) and clone it back at launch when Caches is
// cold. Not built: verify a restored cache actually loads warm first, and note
// the .mlmodelc must stay -- MLModel(contentsOf:) needs it every load, the
// cache only skips compilation.

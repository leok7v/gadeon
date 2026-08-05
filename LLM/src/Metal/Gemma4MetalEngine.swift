// The gemma-4-E2B text forward on the GPU, mirroring Gemma4Engine op for op:
// the SIMD engine is the oracle, and it is itself gated against the Python
// reference dump. Every kernel for one token is encoded onto ONE command
// buffer and committed once, so a decoded token costs one dispatch stream and
// one GPU sync.
//
// Three things differ from MetalEngine and are load-bearing. The weights are
// MIXED (Q4_0 attention and low MLP, Q2_0 wide MLP and lm_head, BF16 norms),
// so every dispatch names a tensor and lets MetalEnc pick the kernel. The KV
// is not one geometry: 11 sliding layers window at 512, L13 is sliding but
// must stay unwindowed because layers 15+ read past its own window, the full
// layers are 512 dims, and 20 layers own no pool at all. And the window is
// applied at READ time as an absolute-position bound rather than by evicting,
// so a shared layer windows its source's full history exactly as HF does --
// the attended range is identical either way.
import Foundation
import Metal

public final class Gemma4MetalEngine {
    let model: Gemma4Model
    public let cfg: Gemma4Config
    // The model's ONE weight mapping. The vision and audio towers read the
    // same file, so they bind through this rather than each cutting their own
    // windows over it.
    public let ctx: MetalContext
    private let map: UnsafeRawPointer
    let pageP: Int

    public private(set) var pos = 0
    var sampler: Sampler?
    // Lock-backed stop flag, the same lever MetalEngine carries for the
    // ternary models. The forward is synchronous and passes no suspension
    // point, so Task cancellation does not reach it -- without this a Stop
    // pressed during a long prefill did NOTHING here, and the turn ran to
    // completion before anything noticed. Cleared at the start of each extend
    // so a prior turn's Stop cannot kill this one.
    private let stopSignal = MetalStopSignal()
    public func requestStop() { stopSignal.raise() }
    public func shouldStop() -> Bool { stopSignal.raisedNow }

    // One pool per NON-shared layer; a shared layer reads its source's.
    private var kv: [Int: MetalKVPool] = [:]

    private let bx, bNormed, bContrib: MTLBuffer
    private let bQ, bK, bV, bAttnOut, bGateNull: MTLBuffer
    private let bFfnGate, bFfnUp: MTLBuffer
    private let bPle, bPleProj, bPleGate: MTLBuffer
    private let bLogits: MTLBuffer
    private let bClamp: MTLBuffer
    // Chunk-sized twins of the scratch above. Prefill runs N tokens through
    // ONE command buffer; the per-token buffers stay for decode.
    private let bXN, bNormedN, bContribN: MTLBuffer
    private let bQN, bKN, bVN, bAttnOutN, bGateNullN: MTLBuffer
    private let bFfnGateN, bFfnUpN: MTLBuffer
    private let bPleN, bPleProjN, bPleGateN, bClampN: MTLBuffer
    // The vision block each chunk ROW belongs to, two absolute uints per row.
    // Sized 8 rows past the batch because the matrix attention masks a partial
    // tail tile rather than branching around it. [vision-block]
    private let bBlocks: MTLBuffer

    // Tokens per prefill chunk. ON by default: 7.6x prefill (pp512 58.0 ->
    // 441.7 t/s on an M4 Pro, decode untouched).
    //
    // It was opt-in while the batched path scored only 0.988 hidden against
    // the per-token one, which looked like a defect. It is not: that number
    // asked the wrong question. Every quantized linear rint()s its input and
    // output to a trained SRQ scale, and rint is DISCONTINUOUS -- so any
    // reordering of a reduction (which is all a GEMM is against a GEMV) lands
    // some element on the far side of a tie and moves it a whole quantization
    // step. The two paths therefore cannot agree to 1.0, and writing f32-tile
    // q4_0/q8_0 GEMMs to remove the fp16 delta did NOT move the number
    // (0.9885 -> 0.9895), which is what proves the mechanism.
    //
    // The question that decides shipping is whether batching costs ACCURACY,
    // and `--gemma-batch-gate` now measures that directly against HF's own
    // logits for the reference ids: per-token 0.9990146, batched 0.9987516,
    // both picking HF's own argmax. Equally close to the truth, differing
    // only in which way they round.
    //
    // 128 rather than more: the curve is 372 (32) / 428 (64) / 442 (128) /
    // 445 (256) / 451 (512) t/s, so 128 buys 98% of the win for a quarter of
    // 512's scratch (~285 KB per token of chunk buffers, ~36 MB here).
    public static let defaultBatch: Int = {
        let raw = ProcessInfo.processInfo.environment["LLM_GEMMA_BATCH"]
        return raw.flatMap { v in Int(v) } ?? 128
    }()
    // The chunk scratch is sized ONCE, at init, from defaultBatch. Raising
    // `batch` past that would write N tokens into buffers holding `capacity`
    // -- off the end of every chunk buffer, into whatever else the GPU has
    // mapped. That is not a wrong answer, it is memory corruption: it took
    // WindowServer down. So the capacity is remembered and enforced.
    //
    // ONE on a GPU without matrix units, which is also what sizes the scratch
    // away to nothing there. The batched forward is built on the simdgroup
    // GEMM, so it does not exist on an A13 (Apple6) -- and asking for a
    // pipeline it cannot build does not fail, it takes the shader compiler
    // service down. MetalEngine draws the same line for the ternary models
    // (`batched`); this is gemma's, and it makes `extend` degrade to the
    // token-by-token path rather than have no path at all.
    private let capacity: Int
    // Per instance so a gate can run the SAME ids both ways and diff them,
    // bounded by what was actually allocated.
    public var batch: Int {
        get { batchStore }
        set {
            precondition(newValue >= 1 && newValue <= capacity,
                         "batch \(newValue) exceeds the \(capacity) this "
                         + "engine allocated for; set LLM_GEMMA_BATCH before "
                         + "constructing it")
            batchStore = newValue
        }
    }
    private var batchStore: Int
    // How many decoder layers ride one command buffer during prefill.
    //
    // A bytesNoCopy weight window is made resident when a command buffer
    // references it, so what ONE buffer asks for is what a split can bound.
    // Against MetalContext.minWindows a layer reads 26-102 MB windows, so one
    // layer per buffer references 78 MB where the whole chunk on one buffer
    // references 714. See [[gpu-weights-are-wired]].
    //
    // WHAT IT ACTUALLY BUYS, macOS, and it is not what that arithmetic says:
    // sampling vm_stat's wired count around a prefill from outside the
    // process, 4 reps each, the delta is 1008 MB at one layer per buffer
    // against 1090 MB for the whole chunk -- both sitting at the ~993 MB
    // every bindable window sums to. The driver does NOT hand residency back
    // between queued buffers, so the peak barely moves. It is ONE anyway
    // because splitting is MEASURED FREE and a device under real memory
    // pressure is a different regime from a 24 GB Mac with none: pp521 t/s
    // ABBA with a cooldown, 4 reps each, 1 buffer 422.3 / 12 buffers 422.2 /
    // 35 (one per layer) 421.7, a 0.15% spread against a 1% run-to-run one.
    // The queue runs them back to back and nothing drains.
    //
    // Env-readable for the same reason LLM_F16_TILES is: it is the A/B
    // instrument, and the iOS half of that measurement is still owed.
    public static let prefillLayers: Int = {
        let raw = ProcessInfo.processInfo
            .environment["LLM_GEMMA_PREFILL_LAYERS"]
        return max(1, raw.flatMap { v in Int(v) } ?? 1)
    }()
    // The bisection gate's readback, on the BATCHED path -- the only one that
    // can express a vision block, since a per-token prefill has not appended
    // the block's forward keys when it computes a query inside it. Absolute
    // positions to sample, and the sink for (layer, position, hidden).
    // DEBUG only: setting it drains the queue after every layer group, where
    // the shipping path lets those buffers run back to back.
    public var probe: Set<Int> = []
    public var onLayer: ((Int, Int, [Float]) -> Void)?

    // Diagnostic: LLM_SKIP=attn,ffn omits a kernel group from the forward, so
    // a METAL_TIMING drop attributes that group's cost. The forward is
    // numerically WRONG while skipping. Decode needs it as much as prefill:
    // past a few hundred cached positions attention is the term that grows
    // with context, and nothing else says by how much.
    static let skip: Set<String> = Set(
        (ProcessInfo.processInfo.environment["LLM_SKIP"] ?? "")
            .split(separator: ",").map(String.init))
    // Per-commit GPU-vs-wall milliseconds on stderr. Wall-clock t/s swings far
    // too much on a shared GPU to A/B against; the GPU interval does not.
    static let timing =
        ProcessInfo.processInfo.environment["METAL_TIMING"] != nil

    // The trained activation clamps, keyed by weight name.
    private var scales: [String: SRQ] = [:]

    // Gemma4Layer holds every norm already dequantized for the SIMD path; the
    // GPU needs their byte offsets into the mapped file instead, resolved once
    // here rather than looked up per token.
    private struct LayerNorms {
        let attn, postAttn, ffn, postFfn, qNorm: WeightRef
        let kNorm: WeightRef?
        let perLayerPost: WeightRef?
    }
    private var norms: [LayerNorms] = []
    private let outputNormOff: WeightRef
    private let perLayerProjNormOff: WeightRef?

    public init(_ model: Gemma4Model, pageP: Int = 512) throws {
        self.model = model
        cfg = model.cfg
        map = model.gguf.map
        self.pageP = pageP
        ctx = try MetalContext(model.gguf)
        try ctx.prewarm()
        // The two tables a token GATHERS a row from rather than streams. The
        // lm_head is untied on this checkpoint, so token_embd is gather-only
        // -- a TIED one is walked end to end every token, where suppressing
        // readahead would be exactly wrong.
        if let ple = model.perLayerEmbd { model.gguf.gathered(ple) }
        if model.output.name != model.tokEmbd.name {
            model.gguf.gathered(model.tokEmbd)
        }
        let c = cfg
        // Scratch is sized to the WIDEST layer: the head dim, the kv width and
        // the MLP width all vary by layer, and a short layer uses a prefix.
        // The kv width is a PRODUCT of two things that vary in opposite
        // directions -- a full layer is twice as wide per head and has a
        // fraction of the heads -- so it is maximised over the layers rather
        // than from the two maxima, which would over-allocate by 8x here.
        let maxHd = max(c.headDimSliding, c.headDimFull)
        let maxFF = (0..<c.nLayer)
            .map { il in model.layers[il].nFF }.max() ?? 0
        let maxKV = (0..<c.nLayer)
            .map { il in c.headDim(il) * model.layers[il].nHeadKV }.max() ?? 0
        // A model with no per-layer inputs has no PLE scratch at all, and a
        // zero-length MTLBuffer is nil rather than empty.
        let pleWidth = max(1, c.nLayer * c.perLayerDim)
        let pleDim = max(1, c.perLayerDim)
        bx = ctx.makeF32(c.nEmbd)
        bNormed = ctx.makeF32(c.nEmbd)
        bContrib = ctx.makeF32(c.nEmbd)
        bQ = ctx.makeF32(maxHd * c.nHead)
        bK = ctx.makeF32(maxKV)
        bV = ctx.makeF32(maxKV)
        bAttnOut = ctx.makeF32(maxHd * c.nHead)
        bGateNull = ctx.makeF32(maxHd * c.nHead)
        bFfnGate = ctx.makeF32(maxFF)
        bFfnUp = ctx.makeF32(maxFF)
        bPle = ctx.makeF32(pleWidth)
        bPleProj = ctx.makeF32(pleWidth)
        bPleGate = ctx.makeF32(pleDim)
        bLogits = ctx.makeF32(c.nVocab)
        bClamp = ctx.makeF32(max(maxFF, max(pleWidth, c.nEmbd)))
        // A blockwise-vision checkpoint must fit its widest image in ONE
        // chunk, so the scratch is sized to the budget the FILE offers rather
        // than to the batch default -- a block that does not fit cannot be
        // attended correctly at all. [vision-block]
        let widest = cfg.blockwiseVision
            ? (model.gguf.int("gemma4.vision.max_soft_tokens") ?? 0) : 0
        let B = ctx.matrixUnits
            ? max(Gemma4MetalEngine.defaultBatch, widest) : 1
        capacity = B
        batchStore = B
        bXN = ctx.makeF32(B * c.nEmbd)
        bNormedN = ctx.makeF32(B * c.nEmbd)
        bContribN = ctx.makeF32(B * c.nEmbd)
        bQN = ctx.makeF32(B * maxHd * c.nHead)
        bKN = ctx.makeF32(B * maxKV)
        bVN = ctx.makeF32(B * maxKV)
        bAttnOutN = ctx.makeF32(B * maxHd * c.nHead)
        bGateNullN = ctx.makeF32(B * maxHd * c.nHead)
        bFfnGateN = ctx.makeF32(B * maxFF)
        bFfnUpN = ctx.makeF32(B * maxFF)
        bPleN = ctx.makeF32(B * pleWidth)
        bPleProjN = ctx.makeF32(B * pleWidth)
        bPleGateN = ctx.makeF32(B * pleDim)
        bClampN = ctx.makeF32(B * max(maxFF, max(pleWidth, c.nEmbd)))
        bBlocks = ctx.makeU32(2 * (B + 8))
        for il in 0..<c.nLayer where !c.isShared(il) {
            kv[il] = MetalKVPool(
                device: ctx.device, P: pageP,
                kvDim: c.headDim(il) * model.layers[il].nHeadKV)
        }
        let g = model.gguf
        for L in model.layers {
            for w in [L.wq, L.wo, L.wk, L.wv, L.ffnGate, L.ffnUp, L.ffnDown,
                      L.perLayerGate, L.perLayerProj] {
                if let w { scales[w.name] = SRQ(g, w.name) }
            }
        }
        scales[model.output.name] = SRQ(g, model.output.name)
        // The BF16 norm kernels read two bytes per weight, so a re-emit that
        // moved a norm to f32 would fuse two weights into one garbage float.
        // Fail here rather than in a hidden state.
        // Over a LOCAL handle, not self.ctx: two of the stored properties are
        // what this builds, so `self` is not callable yet.
        let context = ctx
        func normOff(_ name: String) -> WeightRef {
            let t = g.tensor(name)
            precondition(t.type == .bf16,
                         "gemma norm \(name) is \(t.type), expected bf16")
            return context.window(UInt64(t.base - g.map))
        }
        func normOffIfPresent(_ name: String) -> WeightRef? {
            g.maybe(name).map { _ in normOff(name) }
        }
        outputNormOff = normOff("output_norm.weight")
        perLayerProjNormOff = normOffIfPresent("per_layer_proj_norm.weight")
        for il in 0..<c.nLayer {
            norms.append(LayerNorms(
                attn: normOff("blk.\(il).attn_norm.weight"),
                postAttn: normOff("blk.\(il).post_attn_norm.weight"),
                ffn: normOff("blk.\(il).ffn_norm.weight"),
                postFfn: normOff("blk.\(il).post_ffn_norm.weight"),
                qNorm: normOff("blk.\(il).attn_q_norm.weight"),
                kNorm: c.isShared(il)
                    ? nil : normOff("blk.\(il).attn_k_norm.weight"),
                perLayerPost: normOffIfPresent(
                    "blk.\(il).per_layer_post_norm.weight")))
        }
    }

    public func reset() {
        pos = 0
        stopSignal.clear()
        for (_, pool) in kv { pool.truncate(to: 0) }
    }

    private func off(_ t: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(t.base - map))
    }

    private func srq(_ w: GGUFTensor) -> SRQ { scales[w.name] ?? SRQ.none }

    public func extend(_ ids: [Int32]) -> Int32 {
        extend(ids, softAt: { _ in nil })
    }

    // Prefill a turn that may carry soft tokens, in chunks. `softAt` is asked
    // EXACTLY ONCE per id and in order, because a caller's feed is a cursor.
    public func extend(_ ids: [Int32],
                       softAt: (Int32) -> [Float]?) -> Int32 {
        let B = batch
        stopSignal.clear()
        Diag.memory?("prefill start \(ids.count) ids at pos \(pos)")
        // Where each id may attend across, absolute. Empty everywhere unless
        // this checkpoint asks for blockwise vision.
        let blocks = cfg.blockwiseVision
            ? cfg.visionBlocks(ids, from: pos)
            : [(Int, Int)](repeating: (0, 0), count: ids.count)
        var i = 0
        // The stop is the loop PREDICATE, so a Stop lands between chunks and
        // leaves `pos` and the KV consistent -- the caller then throws
        // EngineError.stopped and ChatSession rolls the turn back.
        while i < ids.count && !stopSignal.raisedNow {
            let n = B > 1 ? chunk(blocks, at: i, want: min(B, ids.count - i))
                          : 1
            var soft: [Int: [Float]] = [:]
            for j in 0..<n {
                if let feature = softAt(ids[i + j]) { soft[j] = feature }
            }
            if n == 1 {
                if let feature = soft[0] {
                    forward(embedding: feature, pos: pos)
                } else {
                    forward(token: Int(ids[i]), pos: pos)
                }
            } else {
                forwardChunk(Array(ids[i ..< (i + n)]), soft,
                             Array(blocks[i ..< (i + n)]))
            }
            pos += n
            i += n
        }
        Diag.memory?("prefill done at pos \(pos)")
        return pick(logits())
    }

    // How many ids the next chunk takes: whole vision blocks and text, packed
    // greedily up to `want`. Without blockwise vision every range is empty and
    // this is `want`. [vision-block]

    private func chunk(_ blocks: [(Int, Int)], at i: Int, want: Int) -> Int {
        let out = Gemma4Config.chunkLength(blocks, at: i, want: want)
        precondition(out <= capacity,
                     "a \(out)-token vision block must ride one chunk and "
                     + "this engine allocated \(capacity); raise "
                     + "LLM_GEMMA_BATCH or lower the image budget")
        return out
    }

    public func decode(_ token: Int32) -> Int32 {
        forward(token: Int(token), pos: pos)
        pos += 1
        return pick(logits())
    }

    func pick(_ values: [Float]) -> Int32 {
        var out: Int32
        if sampler != nil {
            var work = values
            out = sampler!.sample(&work)
            sampler!.accept(out)
        } else {
            out = Int32(argmax(values))
        }
        return out
    }

    public static func argmaxOf(_ v: [Float]) -> Int {
        var bi = 0
        var bv = -Float.greatestFiniteMagnitude
        for i in 0..<v.count where v[i] > bv { bv = v[i]; bi = i }
        return bi
    }

    public func argmax(_ v: [Float]) -> Int {
        var bi = 0
        var bv = -Float.greatestFiniteMagnitude
        for i in 0..<v.count where v[i] > bv { bv = v[i]; bi = i }
        return bi
    }

    // One token at absolute position `pos`, leaving the post-final-norm hidden
    // in bNormed. The KV appends are encoded on the same buffer, so the pools
    // advance with the forward rather than in a separate pass.
    public func forward(token: Int, pos: Int) {
        seedToken(token)
        encode("forward pos=\(pos)") { f in
            if cfg.hasPerLayerInputs { buildPLE(f) }
            layers(f, pos: pos)
        }
    }

    // The CPU-side seed: the embedding row, and the token-identity half of the
    // per-layer input. Split from the encode so the bisection tap can reuse it
    // without duplicating the gather rules.
    private func seedToken(_ token: Int) {
        let c = cfg
        gatherEmbed(token, into: bx.f32(c.nEmbd).baseAddress!)
        if c.hasPerLayerInputs {
            gatherPLE(token,
                      into: bPle.f32(c.nLayer * c.perLayerDim).baseAddress!)
        }
    }

    // One command buffer around one encoded step, committed and waited on.
    private func encode(_ tag: String, _ body: (MetalEnc) -> Void) {
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        body(MetalEnc(ctx: ctx, e: e))
        e.endEncoding()
        commit(cb, tag)
    }

    // The two embedding tables are ROW-GATHERED -- a token reads one row and
    // nothing else in the forward touches them -- so the CPU reads them
    // straight off the mmap and they never enter a GPU buffer.
    //
    // That is the point, not the three dispatches per token it also deletes:
    // a bytesNoCopy buffer gets its pages WIRED into kernel_task when a
    // command buffer references it, wired memory is charged to no process,
    // and per_layer_token_embd is 1260 of this model's 2541 MB with a window
    // to itself. Never binding that window is what keeps it out of the wired
    // set. See [[gpu-weights-are-wired]].
    //
    // The write is safe before the encoder for the reason the soft-token seam
    // already relies on: shared storage, and the previous commit waited on.
    private func gatherEmbed(_ token: Int,
                             into dst: UnsafeMutablePointer<Float>) {
        let c = cfg
        GQ.gather(model.tokEmbd, row: token, from: 0, count: c.nEmbd,
                  into: dst)
        for i in 0..<c.nEmbd { dst[i] *= c.embedScale }
    }

    private func gatherPLE(_ token: Int,
                           into dst: UnsafeMutablePointer<Float>) {
        let c = cfg
        let width = c.nLayer * c.perLayerDim
        GQ.gather(model.perLayerEmbd!, row: token, from: 0, count: width,
                  into: dst)
        for i in 0..<width { dst[i] *= c.perLayerEmbedScale }
    }

    // One SOFT token: a tower's projected feature stands in for an embedding.
    // The per-layer input gathers the PAD row rather than the placeholder's --
    // HF rewrites every multimodal position to pad_token_id before the gather
    // -- while the CONTEXT half still projects the feature itself. And the
    // feature enters UNSCALED: embed_scale multiplies text embeddings only.
    public func forward(embedding: [Float], pos: Int) {
        seedEmbedding(embedding)
        encode("soft forward pos=\(pos)") { f in
            if cfg.hasPerLayerInputs { buildPLE(f) }
            layers(f, pos: pos)
        }
    }

    private func seedEmbedding(_ embedding: [Float]) {
        let c = cfg
        precondition(embedding.count == c.nEmbd,
                     "soft token is \(embedding.count) wide, expected "
                     + "\(c.nEmbd)")
        // Shared storage and the previous forward has been waited on, so the
        // CPU may seed the hidden directly.
        let dst = bx.f32(c.nEmbd)
        for i in 0..<c.nEmbd { dst[i] = embedding[i] }
        if c.hasPerLayerInputs {
            gatherPLE(c.padTokenId,
                      into: bPle.f32(c.nLayer * c.perLayerDim).baseAddress!)
        }
    }

    // The per-layer readback the offline bisection gate reads, with the same
    // tap names Gemma4Engine emits so ONE gate can drive either engine.
    //
    // DEBUG ONLY. It commits a command buffer per LAYER where the shipping
    // path commits one per token, so it is far slower and must never sit on a
    // hot path -- it exists because a hidden state between two layers cannot
    // be read out of a buffer that has not been committed yet.

    public func forward(token: Int, pos: Int,
                        tap: (String, Int, [Float]) -> Void) {
        seedToken(token)
        tapped(pos: pos, tap: tap)
    }

    public func forward(embedding: [Float], pos: Int,
                        tap: (String, Int, [Float]) -> Void) {
        seedEmbedding(embedding)
        tapped(pos: pos, tap: tap)
    }

    private func tapped(pos: Int, tap: (String, Int, [Float]) -> Void) {
        let c = cfg
        tap("embed", -1, Array(bx.f32(c.nEmbd)))
        if c.hasPerLayerInputs { encode("tap ple") { f in buildPLE(f) } }
        for il in 0..<c.nLayer {
            encode("tap attn \(il)") { f in layerAttn(f, il, pos: pos) }
            tap("attn", il, Array(bContrib.f32(c.nEmbd)))
            encode("tap ffn \(il)") { f in layerFfn(f, il) }
            tap("mlp", il, Array(bContrib.f32(c.nEmbd)))
            encode("tap tail \(il)") { f in layerTail(f, il) }
            tap("l_out", il, Array(bx.f32(c.nEmbd)))
        }
        encode("tap final") { f in
            f.rmsnormBF16(x: bx, weightOff: outputNormOff, out: bNormed,
                          n: c.nEmbd, eps: c.eps)
        }
        tap("final", -1, hidden())
    }


    // ---- batched prefill --------------------------------------------------
    // WHY: forward(token:pos:) commits ONE command buffer per token and blocks
    // on it, so a 512-token prefill was 512 GPU round-trips and cost the same
    // per token as decoding -- measured 55 t/s where llama.cpp reaches 629 on
    // this same model and GPU. Here a whole chunk rides a handful of them and
    // is waited on ONCE (see encodeChunk for why it is not exactly one).
    //
    // `pos` is the engine's LIVE position, never zero: a continuation turn
    // prefills onto an existing KV, so every absolute index below (the causal
    // length, the window start, the page slot) is basePos-relative.
    //
    // `soft` maps a row in this chunk to a tower feature. Those rows take the
    // feature UNSCALED as their hidden and gather the PAD row of the per-layer
    // table, exactly as forward(embedding:pos:) does one at a time.
    private func forwardChunk(_ ids: [Int32], _ soft: [Int: [Float]],
                              _ blocks: [(Int, Int)]) {
        let c = cfg
        let n = ids.count
        // The last line of defence: every kernel below indexes by row, so an
        // oversized chunk writes past every chunk buffer at once.
        precondition(n <= capacity,
                     "chunk of \(n) exceeds the \(capacity) allocated")
        let basePos = pos
        // Soft rows are written by the CPU BEFORE anything is encoded; the
        // buffers are shared storage and the previous commit has been waited
        // on, so this is the same seam forward(embedding:) already uses.
        let hidden = bXN.f32(n * c.nEmbd)
        for (row, feature) in soft {
            precondition(feature.count == c.nEmbd,
                         "soft token is \(feature.count) wide, expected "
                         + "\(c.nEmbd)")
            for i in 0..<c.nEmbd { hidden[row * c.nEmbd + i] = feature[i] }
        }
        // Every non-shared layer must reserve the same N positions BEFORE the
        // kernels run: appendBatch allocates the pages and refreshes the
        // bindless table the batched append and attention both read through.
        embedChunk(ids, soft, n)
        blockChunk(blocks, n)
        if c.hasPerLayerInputs { gatherPLEChunk(ids, soft, n) }
        for (_, pool) in kv { pool.appendBatch(n) }
        encodeChunk(basePos: basePos, n: n)
        // logits() reads bNormed, so hand it the LAST row -- the only one a
        // prefill needs a prediction from.
        let src = bNormedN.f32(n * c.nEmbd)
        let dst = bNormed.f32(c.nEmbd)
        for i in 0..<c.nEmbd { dst[i] = src[(n - 1) * c.nEmbd + i] }
    }

    // Text rows take their embedding row and embed_scale; soft rows already
    // hold a tower feature and must NOT be scaled.
    private func embedChunk(_ ids: [Int32], _ soft: [Int: [Float]],
                            _ n: Int) {
        let c = cfg
        let dst = bXN.f32(n * c.nEmbd).baseAddress!
        for j in 0..<n where soft[j] == nil {
            gatherEmbed(Int(ids[j]), into: dst + j * c.nEmbd)
        }
    }

    // The chunk's block table. The padding rows the matrix attention reads
    // past n are cleared to an empty range, so a tail tile's dead rows cannot
    // widen the key sweep. [vision-block]
    private func blockChunk(_ blocks: [(Int, Int)], _ n: Int) {
        let dst = bBlocks.u32(2 * (n + 8))
        for j in 0..<n {
            dst[2 * j] = UInt32(blocks[j].0)
            dst[2 * j + 1] = UInt32(blocks[j].1)
        }
        for j in n..<(n + 8) {
            dst[2 * j] = 0
            dst[2 * j + 1] = 0
        }
    }

    // The token-identity half for the chunk. A soft row gathers the PAD row
    // rather than its placeholder's.
    private func gatherPLEChunk(_ ids: [Int32], _ soft: [Int: [Float]],
                                _ n: Int) {
        let c = cfg
        let width = c.nLayer * c.perLayerDim
        let dst = bPleN.f32(n * width).baseAddress!
        for j in 0..<n {
            gatherPLE(soft[j] == nil ? Int(ids[j]) : c.padTokenId,
                      into: dst + j * width)
        }
    }

    // The CONTEXT half, and the average: bPleN already holds the gathered
    // table rows, scaled.
    private func buildPLEChunk(_ f: MetalEnc, _ n: Int) {
        let c = cfg
        let width = c.nLayer * c.perLayerDim
        // The only unquantized weight in the forward, so the only one with
        // no batched kernel; a row loop still shares the chunk's command
        // buffer.
        f.gemmRows(model.perLayerModelProj!, X: bXN, out: bPleProjN,
                   off: off(model.perLayerModelProj!), N: n)
        f.scaleInPlace(x: bPleProjN, n: n * width,
                       s: 1 / Float(c.nEmbd).squareRoot())
        f.rmsnormRowsBF16(x: bPleProjN, xoff: 0, d: c.perLayerDim,
                          rows: n * c.nLayer,
                          weightOff: perLayerProjNormOff!, eps: c.eps)
        f.addScaled(a: bPleN, b: bPleProjN, n: n * width,
                    s: 1 / Float(2).squareRoot())
    }

    // The chunk's whole forward, cut into command buffers of `prefillLayers`
    // layers each so fewer weight windows are resident at once. They are
    // committed WITHOUT waiting between groups: one queue runs its buffers in
    // commit order, so the scratch every group shares is ordered by
    // construction and only the last has to be waited on. An error is read off
    // every buffer, since a fault in an earlier group leaves the last one
    // reporting success.
    private func encodeChunk(basePos: Int, n: Int) {
        let c = cfg
        let per = min(Gemma4MetalEngine.prefillLayers, c.nLayer)
        BackgroundGate.shared.waitForForeground()
        var queued: [MTLCommandBuffer] = []
        var il = 0
        while il < c.nLayer {
            let end = min(il + per, c.nLayer)
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            let f = MetalEnc(ctx: ctx, e: e)
            if il == 0 && c.hasPerLayerInputs { buildPLEChunk(f, n) }
            layersChunk(f, basePos: basePos, n: n, layers: il..<end)
            if end == c.nLayer {
                f.rmsnormBatchBF16(x: bXN, weightOff: outputNormOff,
                                   y: bNormedN, n: c.nEmbd, rows: n,
                                   eps: c.eps)
            }
            e.endEncoding()
            cb.commit()
            queued.append(cb)
            if let onLayer {
                cb.waitUntilCompleted()
                emit(onLayer, end - 1, bXN, basePos: basePos, n: n)
                if end == c.nLayer {
                    emit(onLayer, c.nLayer, bNormedN, basePos: basePos, n: n)
                }
            }
            il = end
        }
        queued[queued.count - 1].waitUntilCompleted()
        let fault = queued.compactMap { cb in cb.error }.first
        if let fault {
            fatalError("metal prefill chunk n=\(n) pos=\(basePos): \(fault)")
        }
        if Gemma4MetalEngine.timing {
            // Summed across the group, since the buffers run back to back on
            // one queue and only the last was waited on.
            let gpu = queued.reduce(0.0) { total, cb in
                total + (cb.gpuEndTime - cb.gpuStartTime) * 1000
            }
            FileHandle.standardError.write(Data(
                "chunk n=\(n) pos=\(basePos) gpu=\(Int(gpu))ms\n".utf8))
        }
        Diag.memory?("prefill chunk n=\(n) pos=\(basePos)")
    }

    // The probed rows of one chunk buffer, for whoever set `onLayer`.
    private func emit(_ tap: (Int, Int, [Float]) -> Void, _ il: Int,
                      _ buffer: MTLBuffer, basePos: Int, n: Int) {
        let c = cfg
        let rows = buffer.f32(n * c.nEmbd)
        for j in 0..<n where probe.contains(basePos + j) {
            let lo = j * c.nEmbd
            tap(il, basePos + j, Array(rows[lo..<(lo + c.nEmbd)]))
        }
    }

    private func layersChunk(_ f: MetalEnc, basePos: Int, n: Int,
                             layers: Range<Int>) {
        let c = cfg
        let width = c.nLayer * c.perLayerDim
        for il in layers {
            let L = model.layers[il]
            let nm = norms[il]
            f.rmsnormBatchBF16(x: bXN, weightOff: nm.attn, y: bNormedN,
                               n: c.nEmbd, rows: n, eps: c.eps)
            attentionChunk(f, L, il, basePos: basePos, n: n)
            f.rmsnormBatchBF16(x: bContribN, weightOff: nm.postAttn,
                               y: bNormedN, n: c.nEmbd, rows: n, eps: c.eps)
            f.add(x: bXN, y: bNormedN, n: n * c.nEmbd)

            f.rmsnormBatchBF16(x: bXN, weightOff: nm.ffn, y: bNormedN,
                               n: c.nEmbd, rows: n, eps: c.eps)
            if !Gemma4MetalEngine.skip.contains("ffn") {
                f.linear(L.ffnGate, X: bNormedN, out: bFfnGateN,
                         off: off(L.ffnGate), N: n, srq: srq(L.ffnGate),
                         scratch: bClampN)
                f.linear(L.ffnUp, X: bNormedN, out: bFfnUpN, off: off(L.ffnUp),
                         N: n, srq: srq(L.ffnUp), scratch: bClampN)
                f.activateMul(c.activation, a: bFfnGateN, b: bFfnUpN,
                              n: n * L.nFF)
                f.linear(L.ffnDown, X: bFfnGateN, out: bContribN,
                         off: off(L.ffnDown), N: n, srq: srq(L.ffnDown),
                         scratch: bClampN)
            }
            f.rmsnormBatchBF16(x: bContribN, weightOff: nm.postFfn,
                               y: bNormedN, n: c.nEmbd, rows: n, eps: c.eps)
            f.add(x: bXN, y: bNormedN, n: n * c.nEmbd)

            if c.hasPerLayerInputs {
                f.linear(L.perLayerGate!, X: bXN, out: bPleGateN,
                         off: off(L.perLayerGate!), N: n,
                         srq: srq(L.perLayerGate!), scratch: bClampN)
                // Each row multiplies its OWN table row at this layer's
                // offset, so the two operands have different strides.
                f.activateMulRows(c.activation, a: bPleGateN, b: bPleN,
                                  n: c.perLayerDim, rows: n,
                                  aStride: c.perLayerDim, bStride: width,
                                  bOff: il * c.perLayerDim)
                f.linear(L.perLayerProj!, X: bPleGateN, out: bContribN,
                         off: off(L.perLayerProj!), N: n,
                         srq: srq(L.perLayerProj!), scratch: bClampN)
                f.rmsnormBatchBF16(x: bContribN, weightOff: nm.perLayerPost!,
                                   y: bNormedN, n: c.nEmbd, rows: n,
                                   eps: c.eps)
                f.add(x: bXN, y: bNormedN, n: n * c.nEmbd)
            }
            f.scaleInPlace(x: bXN, n: n * c.nEmbd, s: L.layerScalar)
        }
    }

    private func attentionChunk(_ f: MetalEnc, _ L: Gemma4Layer, _ il: Int,
                                basePos: Int, n: Int) {
        let c = cfg
        let hd = c.headDim(il)
        let full = c.isFull(il)
        let base = full ? c.ropeBaseFull : c.ropeBaseSliding
        let rot = full ? c.rotatedPairsFull : c.rotatedPairsSliding
        f.linear(L.wq, X: bNormedN, out: bQN, off: off(L.wq), N: n,
                 srq: srq(L.wq), scratch: bClampN)
        f.rmsnormRowsBF16(x: bQN, xoff: 0, d: hd, rows: n * c.nHead,
                          weightOff: norms[il].qNorm, eps: c.eps)
        f.ropeGemmaBatch(x: bQN, headDim: hd, nHead: c.nHead, rotated: rot,
                         base: base, basePos: basePos, N: n)
        let nKV = L.nHeadKV
        let pool = kv[c.isShared(il) ? c.sharedSource(il) : il]!
        if !c.isShared(il) {
            f.linear(L.wk!, X: bNormedN, out: bKN, off: off(L.wk!), N: n,
                     srq: srq(L.wk!), scratch: bClampN)
            f.rmsnormRowsBF16(x: bKN, xoff: 0, d: hd, rows: n * nKV,
                              weightOff: norms[il].kNorm!, eps: c.eps)
            f.ropeGemmaBatch(x: bKN, headDim: hd, nHead: nKV,
                             rotated: rot, base: base, basePos: basePos, N: n)
            // A layer with no value projection takes the key's, which is why
            // this reads bNormedN again rather than bKN: by now that holds a
            // normed and rotated key, and the value wants neither.
            let wv = L.wv ?? L.wk!
            f.linear(wv, X: bNormedN, out: bVN, off: off(wv), N: n,
                     srq: srq(wv), scratch: bClampN)
            f.rmsnormRowsNoWeight(x: bVN, xoff: 0, d: hd,
                                  rows: n * nKV, eps: c.eps)
            // The pages were reserved before the encoder opened, so this
            // writes absolute positions basePos..basePos+n-1.
            f.kvAppendBatch(kCurN: bKN, vCurN: bVN, kAddr: pool.kAddr,
                            vAddr: pool.vAddr, pages: pool.residentPages,
                            kvDim: hd * nKV, basePos: basePos,
                            P: pool.P, N: n)
        }
        // scaling 1.0 -- q_norm is the query's only normalization. The window
        // is per ROW inside the kernel, since each query in the chunk sits at
        // its own absolute position.
        if !Gemma4MetalEngine.skip.contains("attn") {
            f.attnBatch(qN: bQN, kAddr: pool.kAddr, vAddr: pool.vAddr,
                        pages: pool.residentPages, gateN: bGateNullN,
                        outN: bAttnOutN, hd: hd, nH: c.nHead, nKV: nKV,
                        kvDim: hd * nKV, P: pool.P, scale: 1,
                        basePos: basePos, N: n, gated: 0,
                        window: full ? 0 : c.slidingWindow,
                        // BOTH layer types carry the blocks. The docstring on
                        // create_masks_for_vision_model says the global ones
                        // are causal only, and that function is not the one
                        // the forward runs: it puts block_sequence_ids into
                        // mask_kwargs and lets create_masks_for_generate
                        // build both masks from it. Gated, not guessed --
                        // sliding-only leaves layer 5 at 0.95 where both
                        // reach six nines. [vision-block]
                        blocks: bBlocks)
        }
        f.linear(L.wo, X: bAttnOutN, out: bContribN, off: off(L.wo), N: n,
                 srq: srq(L.wo), scratch: bClampN)
    }

    private func layers(_ f: MetalEnc, pos: Int) {
        let c = cfg
        for il in 0..<c.nLayer { layer(f, il, pos: pos) }
        f.rmsnormBF16(x: bx, weightOff: outputNormOff, out: bNormed,
                      n: c.nEmbd, eps: c.eps)
    }

    // One decoder layer, leaving its output in bx. Split out because a
    // per-layer tap has to commit between layers to read the residual, where
    // the shipping path encodes all of them onto one command buffer.
    private func layer(_ f: MetalEnc, _ il: Int, pos: Int) {
        layerAttn(f, il, pos: pos)
        layerFfn(f, il)
        layerTail(f, il)
    }

    // The layer in the three pieces a bisection can score against HF's own
    // modules: self_attn's output, mlp's output, and what the layer leaves
    // behind. Cut here rather than inside the tap so the shipping path runs
    // the identical sequence either way. Each leaves its result in bContrib
    // except the last, which leaves the layer's output in bx.
    private func layerAttn(_ f: MetalEnc, _ il: Int, pos: Int) {
        let c = cfg
        f.rmsnormBF16(x: bx, weightOff: norms[il].attn, out: bNormed,
                      n: c.nEmbd, eps: c.eps)
        attention(f, model.layers[il], il, pos: pos)
    }

    private func layerFfn(_ f: MetalEnc, _ il: Int) {
        let c = cfg
        let L = model.layers[il]
        let nm = norms[il]
        f.rmsnormBF16(x: bContrib, weightOff: nm.postAttn,
                      out: bNormed, n: c.nEmbd, eps: c.eps)
        f.add(x: bx, y: bNormed, n: c.nEmbd)

        f.rmsnormBF16(x: bx, weightOff: nm.ffn, out: bNormed,
                      n: c.nEmbd, eps: c.eps)
        if !Gemma4MetalEngine.skip.contains("ffn") {
            f.linear(L.ffnGate, x: bNormed, out: bFfnGate,
                     off: off(L.ffnGate), srq: srq(L.ffnGate),
                     scratch: bClamp)
            f.linear(L.ffnUp, x: bNormed, out: bFfnUp, off: off(L.ffnUp),
                     srq: srq(L.ffnUp), scratch: bClamp)
            f.activateMul(c.activation, a: bFfnGate, b: bFfnUp, n: L.nFF)
            f.linear(L.ffnDown, x: bFfnGate, out: bContrib,
                     off: off(L.ffnDown), srq: srq(L.ffnDown),
                     scratch: bClamp)
        }
    }

    private func layerTail(_ f: MetalEnc, _ il: Int) {
        let c = cfg
        let L = model.layers[il]
        let nm = norms[il]
        f.rmsnormBF16(x: bContrib, weightOff: nm.postFfn,
                      out: bNormed, n: c.nEmbd, eps: c.eps)
        f.add(x: bx, y: bNormed, n: c.nEmbd)

        if c.hasPerLayerInputs {
            f.linear(L.perLayerGate!, x: bx, out: bPleGate,
                     off: off(L.perLayerGate!),
                     srq: srq(L.perLayerGate!), scratch: bClamp)
            f.activateMul(c.activation, a: bPleGate, b: bPle,
                          n: c.perLayerDim, bOff: il * c.perLayerDim)
            f.linear(L.perLayerProj!, x: bPleGate, out: bContrib,
                     off: off(L.perLayerProj!),
                     srq: srq(L.perLayerProj!), scratch: bClamp)
            f.rmsnormBF16(x: bContrib, weightOff: nm.perLayerPost!,
                          out: bNormed, n: c.nEmbd, eps: c.eps)
            f.add(x: bx, y: bNormed, n: c.nEmbd)
        }
        f.scaleInPlace(x: bx, n: c.nEmbd, s: L.layerScalar)
    }

    // The CONTEXT half of the per-layer input, and the average. bPle already
    // holds the gathered table row -- that is only the token-identity half --
    // and this projects the scaled embedding for the other, averaging the two
    // by 1/sqrt(2). Using the gather alone still produces plausible numbers
    // and is wrong everywhere.
    private func buildPLE(_ f: MetalEnc) {
        let c = cfg
        let n = c.nLayer * c.perLayerDim
        f.gemv(model.perLayerModelProj!, x: bx, out: bPleProj,
               off: off(model.perLayerModelProj!))
        f.scaleInPlace(x: bPleProj, n: n,
                       s: 1 / Float(c.nEmbd).squareRoot())
        f.rmsnormRowsBF16(x: bPleProj, xoff: 0, d: c.perLayerDim,
                          rows: c.nLayer,
                          weightOff: perLayerProjNormOff!, eps: c.eps)
        f.addScaled(a: bPle, b: bPleProj, n: n,
                    s: 1 / Float(2).squareRoot())
    }

    private func attention(_ f: MetalEnc, _ L: Gemma4Layer, _ il: Int,
                           pos: Int) {
        let c = cfg
        let hd = c.headDim(il)
        let full = c.isFull(il)
        let base = full ? c.ropeBaseFull : c.ropeBaseSliding
        let rot = full ? c.rotatedPairsFull : c.rotatedPairsSliding
        f.linear(L.wq, x: bNormed, out: bQ, off: off(L.wq),
                 srq: srq(L.wq), scratch: bClamp)
        f.rmsnormRowsBF16(x: bQ, xoff: 0, d: hd, rows: c.nHead,
                          weightOff: norms[il].qNorm, eps: c.eps)
        f.ropeGemma(x: bQ, headDim: hd, nHead: c.nHead, rotated: rot,
                    base: base, pos: pos)
        let nKV = L.nHeadKV
        let pool = kv[c.isShared(il) ? c.sharedSource(il) : il]!
        if !c.isShared(il) {
            f.linear(L.wk!, x: bNormed, out: bK, off: off(L.wk!),
                     srq: srq(L.wk!), scratch: bClamp)
            f.rmsnormRowsBF16(x: bK, xoff: 0, d: hd, rows: nKV,
                              weightOff: norms[il].kNorm!, eps: c.eps)
            f.ropeGemma(x: bK, headDim: hd, nHead: nKV, rotated: rot,
                        base: base, pos: pos)
            let wv = L.wv ?? L.wk!
            f.linear(wv, x: bNormed, out: bV, off: off(wv),
                     srq: srq(wv), scratch: bClamp)
            f.rmsnormRowsNoWeight(x: bV, xoff: 0, d: hd, rows: nKV,
                                  eps: c.eps)
            let tail = pool.tailForAppend()
            f.kvAppend(kCur: bK, vCur: bV, K: tail.k, V: tail.v,
                       kvDim: hd * nKV, pos: tail.slot)
            pool.commitAppend()
            pool.refreshTable()
        }
        // The window bounds the READ, so no eviction is needed for parity: the
        // attended range is the same either way.
        let lo = full ? 0 : max(0, pos - c.slidingWindow + 1)
        // self.scaling = 1.0 -- q_norm is the query's only normalization.
        if !Gemma4MetalEngine.skip.contains("attn") {
            f.attnPaged(q: bQ, kAddr: pool.kAddr, vAddr: pool.vAddr,
                        pages: pool.residentPages, gate: bGateNull,
                        out: bAttnOut, hd: hd, nH: c.nHead, nKV: nKV,
                        T: pos + 1, kvDim: hd * nKV, P: pool.P, scale: 1,
                        gated: 0, lo: lo)
        }
        f.linear(L.wo, x: bAttnOut, out: bContrib, off: off(L.wo),
                 srq: srq(L.wo), scratch: bClamp)
    }

    // The final softcap is why nothing in the reference dump exceeds +/-30.
    public func logits() -> [Float] {
        let c = cfg
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        let f = MetalEnc(ctx: ctx, e: e)
        f.linear(model.output, x: bNormed, out: bLogits,
                 off: off(model.output), srq: srq(model.output),
                 scratch: bClamp)
        if c.logitSoftcap > 0 {
            f.softcap(x: bLogits, n: c.nVocab, cap: c.logitSoftcap)
        }
        e.endEncoding()
        commit(cb, "lm_head")
        return Array(bLogits.f32(c.nVocab))
    }

    // Rollback is INDEX-based. The pools are pos-major and append-only, so a
    // mark is the position plus each pool's length and a rewind truncates back
    // to it -- nothing is copied, which is what lets a turn be undone at any
    // context length for the cost of a few integers.
    public struct Bookmark: Sendable {
        let pos: Int
        let lens: [Int: Int]
    }

    public func bookmark() -> Bookmark {
        var lens: [Int: Int] = [:]
        for (il, pool) in kv { lens[il] = pool.len }
        return Bookmark(pos: pos, lens: lens)
    }

    public func restore(_ b: Bookmark) {
        pos = b.pos
        for (il, n) in b.lens { kv[il]!.truncate(to: n) }
    }

    // A parked conversation. The bookmark is INDEX-only on purpose (rollback
    // must stay free at any context length), so the bytes come from the LIVE
    // pools truncated to what it recorded -- which is exact, because a caller
    // serializes the state it just took.

    public func serialize(_ b: Bookmark) -> Data {
        var out = Data()
        StateBytes.putHeader(&out)
        StateBytes.putInt(&out, b.pos)
        StateBytes.putInt(&out, b.lens.count)
        for il in b.lens.keys.sorted() {
            let len = b.lens[il]!
            let pool = kv[il]!
            StateBytes.putInt(&out, il)
            StateBytes.putInt(&out, len)
            StateBytes.putFloats(&out, read(pool.kPages, len, pool))
            StateBytes.putFloats(&out, read(pool.vPages, len, pool))
        }
        return out
    }

    // Pages are HALF; the file stays f32 so it means one thing whichever
    // engine wrote it. One pass per park, never per token.
    private func read(_ pages: [MTLBuffer], _ len: Int,
                      _ pool: MetalKVPool) -> [Float] {
        var out = [Float](repeating: 0, count: len * pool.kvDim)
        for i in 0..<len {
            let src = pages[i / pool.P].contents()
                .assumingMemoryBound(to: Float16.self)
                + (i % pool.P) * pool.kvDim
            for c in 0..<pool.kvDim { out[i * pool.kvDim + c] = Float(src[c]) }
        }
        return out
    }

    // What deserialize hands back: whole pools rather than the lengths a
    // rollback carries, since the engine it lands in holds nothing yet.
    public struct Parked: @unchecked Sendable, BackendState {
        let pos: Int
        let kv: [Int: [(k: [Float], v: [Float])]]
    }

    public func deserialize(_ data: Data) -> Parked? {
        let b = [UInt8](data)
        var p = 0
        guard StateBytes.readHeader(b, &p) else { return nil }
        let pos = StateBytes.getInt(b, &p)
        var kv: [Int: [(k: [Float], v: [Float])]] = [:]
        for _ in 0..<StateBytes.getInt(b, &p) {
            let il = StateBytes.getInt(b, &p)
            _ = StateBytes.getInt(b, &p)
            let k = StateBytes.getFloats(b, &p)
            let v = StateBytes.getFloats(b, &p)
            kv[il] = [(k: k, v: v)]
        }
        return Parked(pos: pos, kv: kv)
    }

    // Install a parked conversation: refill the pools position by position
    // through the ordinary append, so the page table is built the one way.
    public func adopt(_ parked: Parked) {
        reset()
        for (il, blobs) in parked.kv {
            guard let pool = kv[il], let blob = blobs.first else { continue }
            let n = blob.k.count / pool.kvDim
            for i in 0..<n {
                let tail = pool.tailForAppend()
                let kd = tail.k.contents()
                    .assumingMemoryBound(to: Float16.self)
                    + tail.slot * pool.kvDim
                let vd = tail.v.contents()
                    .assumingMemoryBound(to: Float16.self)
                    + tail.slot * pool.kvDim
                for c in 0..<pool.kvDim {
                    kd[c] = Float16(blob.k[i * pool.kvDim + c])
                    vd[c] = Float16(blob.v[i * pool.kvDim + c])
                }
                pool.commitAppend()
            }
            pool.refreshTable()
        }
        pos = parked.pos
    }

    // The post-final-norm hidden, for the op-by-op gate against SIMD.
    public func hidden() -> [Float] { Array(bNormed.f32(cfg.nEmbd)) }

    // The GPU/CPU boundary: everything between commit and waitUntilCompleted
    // happens where no Swift runs.
    //
    // NO memory report here, deliberately. This is the DECODE path, one call
    // per token plus one per lm_head, and the hook behind Diag.memory writes
    // a formatted line to stderr AND to the log file synchronously -- ~86
    // blocking writes a second at 43 t/s, for a curve the 250 ms sampler in
    // Footprint.watch already draws. The reports that earn their place are on
    // the PREFILL path (encodeChunk, and the brackets in extend), where the
    // peak actually lands and no sampler tick is guaranteed to catch it.
    private func commit(_ cb: MTLCommandBuffer, _ tag: String) {
        BackgroundGate.shared.waitForForeground()
        let t0 = Date()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fatalError("metal \(tag): \(err)") }
        if Gemma4MetalEngine.timing {
            let wall = Date().timeIntervalSince(t0) * 1000
            let gpu = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            FileHandle.standardError.write(Data(
                "\(tag) gpu=\(Int(gpu))ms wall=\(Int(wall))ms\n".utf8))
        }
    }
}


// Adapts Gemma4MetalEngine to the AgentBackend seam, so ChatSession drives the
// GPU exactly as it drives the SIMD engine. The shape mirrors Gemma4Backend;
// only the engine type differs.
public final class Gemma4MetalBackend: AgentBackend, @unchecked Sendable {
    let engine: Gemma4MetalEngine
    let tokenizer: GemmaTokenizer
    private var savedMark: Gemma4MetalEngine.Bookmark?

    init(engine: Gemma4MetalEngine, tokenizer: GemmaTokenizer) {
        self.engine = engine
        self.tokenizer = tokenizer
    }

    public var eos: Int32 { tokenizer.eosId }
    public var bosToken: String { tokenizer.bosToken }
    public var eosIds: Set<Int32> { tokenizer.eosIds }
    public var position: Int { get async { engine.pos } }

    // The engine's weight mapping, for whatever builds this model's towers.
    public var ctx: MetalContext { engine.ctx }

    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text, addSpecial: true)
    }

    public func tokenBytes(_ id: Int32) -> [UInt8] { tokenizer.tokenBytes(id) }

    public func text(_ ids: [Int32]) -> String { tokenizer.decode(ids) }

    public func reset() async { engine.reset() }

    public func useSampler(_ s: Sampler?) async { engine.sampler = s }

    // A prefill-phase Stop leaves the chunk loop early; surface it as
    // EngineError.stopped so ChatSession rolls the turn back, which is the
    // parity MetalBackend already has. Without it a Stop during a long gemma
    // prefill was silently ignored and the turn ran to completion.
    public func extend(_ ids: [Int32]) async throws -> Int32 {
        let out = engine.extend(ids)
        if engine.shouldStop() { throw EngineError.stopped }
        return out
    }

    public func requestStop() { engine.requestStop() }
    public func shouldStop() -> Bool { engine.shouldStop() }

    public func supportsSoftTokens() async -> Bool { true }

    public func extendSoft(_ ids: [Int32],
                           spans: [SoftSpan]) async throws -> Int32 {
        let feed = SoftFeed(spans)
        let out = engine.extend(ids, softAt: { id in feed.row(id) })
        if engine.shouldStop() { throw EngineError.stopped }
        return out
    }

    public func decode(_ token: Int32) async throws -> Int32 {
        engine.decode(token)
    }

    public func mark() async throws { savedMark = engine.bookmark() }

    public func rewind() async throws {
        if let m = savedMark { engine.restore(m) }
    }

    struct State: BackendState {
        let bookmark: Gemma4MetalEngine.Bookmark
    }

    public func saveState() async throws -> any BackendState {
        State(bookmark: engine.bookmark())
    }

    public func loadState(_ state: any BackendState) async throws {
        if let s = state as? State {
            engine.restore(s.bookmark)
        } else if let p = state as? Gemma4MetalEngine.Parked {
            engine.adopt(p)
        }
    }

    // A parked conversation restores from BYTES, never from a forward pass.
    // The protocol default returns empty Data, which reads as a warm cache
    // and is a cold one every time. [[never-reprefill]]

    public func serializeState(_ state: any BackendState) async -> Data {
        var out = Data()
        if let s = state as? State { out = engine.serialize(s.bookmark) }
        return out
    }

    public func deserializeState(_ data: Data) async throws
        -> any BackendState {
        guard let p = engine.deserialize(data) else {
            throw GGUFErr.parse("parked state is not this build's format")
        }
        return p
    }

    struct Turn: BackendState {
        let bookmark: Gemma4MetalEngine.Bookmark
        let mark: Gemma4MetalEngine.Bookmark?
    }

    public func checkpoint() async throws -> any BackendState {
        Turn(bookmark: engine.bookmark(), mark: savedMark)
    }

    public func rollback(_ state: any BackendState) async throws {
        if let t = state as? Turn {
            engine.restore(t.bookmark)
            savedMark = t.mark
        }
    }
}

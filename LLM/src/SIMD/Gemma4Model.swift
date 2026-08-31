// gemma-4-E2B geometry and per-layer tensor handles, read straight from the
// GGUF that scripts/convert/gemma4gguf/repack_qat.py writes. Nothing here is
// hardcoded per model: every scalar comes from a `gemma4.*` metadata key and
// every width from a tensor shape, so a re-emit with different dimensions
// loads unchanged.
//
// This is NOT a BonsaiConfig variant. The key set genuinely differs (two rope
// bases, two head dims, a per-layer type array, KV sharing, a logit softcap,
// the per-layer-embedding width), so it is its own type -- see Findings F28.
import Foundation

// A key the runtime cannot invent a value for. Absence throws so a caller can
// say WHICH file is too old, instead of defaulting into a wrong answer.

func requireInt(_ g: GGUF, _ key: String, _ why: String) throws -> Int {
    let value = g.int(key)
    if value == nil { throw GGUFErr.parse("\(key) is absent; \(why)") }
    return value!
}

public struct Gemma4Config {
    public let nEmbd: Int            // 1536
    public let nLayer: Int           // 35
    let nHead: Int                   // 8
    let nHeadKV: Int                 // 1 (MQA)
    // A full-attention layer may carry its own kv-head count as well as its
    // own head width; neither follows from the sliding ones.
    let nHeadKVFull: Int
    // The full layers of a unified checkpoint have no value projection at
    // all: the value is the key projection's output, taken BEFORE k_norm and
    // rope, through the scale-free v_norm.
    let kEqV: Bool
    let headDimSliding: Int          // 256
    let headDimFull: Int             // 512  (global_head_dim)
    let eps: Float                   // 1e-6
    public let nVocab: Int           // 262144
    let perLayerDim: Int             // 256
    public let slidingWindow: Int    // 512
    let kvSharedLayers: Int          // 20
    let logitSoftcap: Float          // 30
    let embedScale: Float            // sqrt(1536)
    let perLayerEmbedScale: Float    // sqrt(256) = 16
    let ropeBaseSliding: Float       // 1e4
    let ropeBaseFull: Float          // 1e6
    // `proportional` rope builds rope_angles = int(pf * head_dim // 2) real
    // inverse frequencies and ZERO-PADS the rest, so a full-attention head
    // rotates its first 64 pairs and leaves the other 192 as identity. The
    // pairing stays (j, j + head_dim/2) either way. See F5.
    let rotatedPairsFull: Int        // 64
    let rotatedPairsSliding: Int     // 128 (all of them)
    // 0 = sliding_attention, 1 = full_attention, per layer.
    let layerFull: [Bool]
    // The tokens of ONE image see each other in BOTH directions, and only on
    // the SLIDING layers -- the global ones stay causal, which is where gemma
    // 4 parts company with gemma 3. Off on the mobile checkpoints, and a
    // model that wants it reads nothing but nonsense from an image without
    // it: the receptive field is wrong on 40 of the 12B's 48 layers.
    let blockwiseVision: Bool
    // Which placeholders form such a block. Audio is deliberately NOT one:
    // HF's block ids come from mm_token_type_ids 1 and 2, image and video.
    let visionTokens: Set<Int32>
    let activation: GemmaActivation
    // Every multimodal position gathers THIS row of the per-layer table,
    // not the modality placeholder's.
    let padTokenId: Int

    // The first layer index that owns no k/v weights and reads a shared
    // history instead (35 - 20 = 15).
    var firstShared: Int { nLayer - kvSharedLayers }
    // Read-only geometry for callers outside the module (the CLI banner).
    public var fullLayers: Int {
        layerFull.filter { full in full }.count
    }
    public var slidingLayers: Int {
        layerFull.filter { full in !full }.count
    }
    func isFull(_ il: Int) -> Bool { layerFull[il] }
    func headDim(_ il: Int) -> Int {
        isFull(il) ? headDimFull : headDimSliding
    }
    func headCountKV(_ il: Int) -> Int {
        isFull(il) ? nHeadKVFull : nHeadKV
    }
    func isShared(_ il: Int) -> Bool { il >= firstShared }
    // A checkpoint with no per-layer embeddings records the width as zero,
    // which is the file saying it has none rather than being too old to say.
    var hasPerLayerInputs: Bool { perLayerDim > 0 }

    // The layer whose K/V a shared layer reads: the LAST non-shared layer of
    // the same type. HF marks it with store_full_length_kv; for E2B that is
    // L13 (sliding) and L14 (full), and L13 must therefore keep an unwindowed
    // history even though it is a sliding layer (F9).
    // The vision blocks in a turn's ids: each CONTIGUOUS run of image or
    // video placeholders is one, given as the absolute [start, end) every
    // position inside it attends across. A position outside every run gets an
    // empty range, which reduces its attention to the plain causal one.
    //
    // Only the QUERY's own block matters, which is what keeps this out of the
    // KV: a key in another block that is already behind the query is reached
    // causally, and one ahead of it is not reachable at all.

    func visionBlocks(_ ids: [Int32], from base: Int) -> [(Int, Int)] {
        var out = [(Int, Int)](repeating: (0, 0), count: ids.count)
        var i = 0
        while i < ids.count {
            var j = i
            while j < ids.count && visionTokens.contains(ids[j]) { j += 1 }
            if j > i {
                for k in i..<j { out[k] = (base + i, base + j) }
                i = j
            } else {
                i += 1
            }
        }
        return out
    }

    // One step of the walk below: a vision block is indivisible, text advances
    // by one id.

    private static func span(_ blocks: [(Int, Int)], at i: Int) -> Int {
        let here = blocks[i]
        return here.1 > here.0 ? here.1 - here.0 : 1
    }

    // How many ids the next prefill chunk may take. A vision block reads
    // FORWARD across itself, and a key in a later chunk has not been appended
    // yet, so a block must ride ONE chunk WHOLE. Beyond that the walk is
    // greedy: whole blocks and single text ids until the next step would pass
    // `want`, so a video's 32 blocks of 63 ride 128-token chunks rather than
    // forcing one chunk per frame. The FIRST step is taken unconditionally --
    // a block wider than `want` still goes alone, since a partial one cannot
    // be attended at all, and the engine's capacity precondition is what says
    // so. Without blockwise vision every range is empty and this is `want`.
    // Both engines chunk through here. STATIC because it reads nothing but
    // its arguments, which is what lets a test drive it without a checkpoint.

    static func chunkLength(_ blocks: [(Int, Int)], at i: Int,
                            want: Int) -> Int {
        let limit = min(want, blocks.count - i)
        var out = span(blocks, at: i)
        while out < limit && out + span(blocks, at: i + out) <= limit {
            out += span(blocks, at: i + out)
        }
        return out
    }

    private static func sourceBidirectional(_ g: GGUF) -> String? {
        var out: String? = nil
        if let raw = g.string("gemma4.source.config_json"),
           let data = raw.data(using: .utf8),
           let root = (try? JSONSerialization.jsonObject(with: data))
               as? [String: Any],
           let text = root["text_config"] as? [String: Any] {
            out = text["use_bidirectional_attention"] as? String
        }
        return out
    }

    func sharedSource(_ il: Int) -> Int {
        var src = -1
        var i = 0
        while i < firstShared {
            if layerFull[i] == layerFull[il] { src = i }
            i += 1
        }
        return src
    }

    init(_ g: GGUF) throws {
        func i(_ k: String) -> Int { g.int("gemma4." + k)! }
        func f(_ k: String) -> Float { Float(g.double("gemma4." + k)!) }
        nEmbd = i("embedding_length")
        nLayer = i("block_count")
        nHead = i("attention.head_count")
        nHeadKV = i("attention.head_count_kv")
        // Both are absent from the mobile checkpoints, where the full layers
        // share the sliding kv-head count and every layer owns a value.
        nHeadKVFull = g.int("gemma4.attention.global_head_count_kv")
            ?? i("attention.head_count_kv")
        kEqV = g.bool("gemma4.attention.k_eq_v") ?? false
        // NOT READABLE BY UPSTREAM llama.cpp, and these two lines are most
        // of why. Deliberate, not an oversight -- we do not publish these
        // files to it. What it would take, recorded so a future reader does
        // not have to re-derive it:
        //
        //   * `attention.key_length` means the GLOBAL head width upstream and
        //     the SLIDING one here. A file cannot satisfy both readings, so
        //     the emit would write the global width there and the sliding one
        //     as `attention.key_length_swa`; a reader takes
        //     `key_length_swa ?? key_length` and stays right for old files.
        //   * `attention.sliding_window_pattern` is what upstream expects in
        //     place of our `gemma4.layer_types` array, and it is what its
        //     loader trips on first.
        //   * Tensor NAMES are ours (`v.patch_embd`, `mm.vision`, the
        //     `per_layer_*` set); upstream's gemma path expects its own
        //     spelling, and an encoder-free vision block has no counterpart
        //     there at all.
        //
        // The block TYPES are already standard (Q4_0 / Q8_0 / BF16), unlike
        // the ternary Q2_0 lineage -- so this is a metadata and naming job,
        // not a format one, and it costs no re-quantization.
        headDimSliding = i("attention.key_length")
        headDimFull = i("attention.global_key_length")
        eps = f("attention.layer_norm_rms_epsilon")
        perLayerDim = i("per_layer_dim")
        slidingWindow = i("attention.sliding_window")
        kvSharedLayers = i("attention.kv_shared_layers")
        logitSoftcap = f("logit_softcap")
        embedScale = f("embed_scale")
        perLayerEmbedScale = f("per_layer_embed_scale")
        ropeBaseSliding = f("rope.freq_base_sliding")
        ropeBaseFull = f("rope.freq_base_full")
        let pf = Double(f("rope.partial_factor_full"))
        rotatedPairsFull = Int(pf * Double(headDimFull) / 2)
        rotatedPairsSliding = headDimSliding / 2
        layerFull = g.ints("gemma4.layer_types")!.map { t in t == 1 }
        activation = try GemmaActivation.read(g, "gemma4.activation")
        // The dedicated key wins; a file emitted before it existed still
        // carries its origin config.json verbatim and answers from there.
        let mode = g.string("gemma4.attention.bidirectional")
            ?? Gemma4Config.sourceBidirectional(g)
        blockwiseVision = mode == "vision"
        visionTokens = Set([g.int("gemma4.image_token_id"),
                            g.int("gemma4.video_token_id")]
            .compactMap { id in id.map { v in Int32(v) } })
        padTokenId = try requireInt(
            g, "tokenizer.ggml.padding_token_id",
            "a soft token has no per-layer row to gather without it")
        nVocab = g.tensor("token_embd.weight").dims[1]
    }
}

// One decoder block's weights. The MLP width is read from the tensor itself,
// not from the config: layers 0-14 are 6144 wide and 15-34 are 12288
// (use_double_wide_mlp), and layer 15 is simultaneously where KV sharing
// starts and where the QAT drops to 2 bits (F8).
struct Gemma4Layer {
    let attnNorm: [Float]        // input_layernorm
    let postAttnNorm: [Float]    // post_attention_layernorm (BEFORE the add)
    let ffnNorm: [Float]         // pre_feedforward_layernorm
    let postFfnNorm: [Float]     // post_feedforward_layernorm
    let perLayerPostNorm: [Float]
    let layerScalar: Float

    let wq: GGUFTensor
    let wo: GGUFTensor
    let qNorm: [Float]
    // Absent on the shared layers (15-34): HF builds no k/v there, and the
    // repack drops the checkpoint's dead copies (F7). `wv` is additionally
    // absent wherever K IS V, which is a property of the layer rather than of
    // the model -- the same file mixes both.
    let wk: GGUFTensor?
    let wv: GGUFTensor?
    let kNorm: [Float]?
    // How many kv heads this layer's history holds, off the projection it
    // actually multiplies rather than off a config scalar.
    let nHeadKV: Int

    let ffnGate: GGUFTensor
    let ffnUp: GGUFTensor
    let ffnDown: GGUFTensor
    let nFF: Int

    let perLayerGate: GGUFTensor?  // [nEmbd -> perLayerDim]
    let perLayerProj: GGUFTensor?  // [perLayerDim -> nEmbd]

    init(_ g: GGUF, _ il: Int, _ cfg: Gemma4Config) {
        func t(_ s: String) -> GGUFTensor { g.tensor("blk.\(il).\(s)") }
        func v(_ s: String) -> [Float] { Dense.floats(t(s)) }
        func m(_ s: String) -> GGUFTensor? { g.maybe("blk.\(il).\(s)") }
        attnNorm = v("attn_norm.weight")
        postAttnNorm = v("post_attn_norm.weight")
        ffnNorm = v("ffn_norm.weight")
        postFfnNorm = v("post_ffn_norm.weight")
        perLayerPostNorm = m("per_layer_post_norm.weight")
            .map(Dense.floats) ?? []
        layerScalar = v("layer_scalar").first ?? 1
        wq = t("attn_q.weight")
        wo = t("attn_output.weight")
        qNorm = v("attn_q_norm.weight")
        let shared = cfg.isShared(il)
        wk = shared ? nil : t("attn_k.weight")
        wv = shared ? nil : m("attn_v.weight")
        kNorm = shared ? nil : v("attn_k_norm.weight")
        nHeadKV = wk.map { w in w.dims[1] / cfg.headDim(il) }
            ?? cfg.headCountKV(il)
        ffnGate = t("ffn_gate.weight")
        ffnUp = t("ffn_up.weight")
        ffnDown = t("ffn_down.weight")
        nFF = ffnGate.dims[1]
        perLayerGate = m("per_layer_gate.weight")
        perLayerProj = m("per_layer_proj.weight")
    }
}

struct Gemma4AssistLayer {
    let attnNorm: GGUFTensor
    let postAttnNorm: GGUFTensor
    let ffnNorm: GGUFTensor
    let postFfnNorm: GGUFTensor
    let qNorm: GGUFTensor
    let layerScalar: Float
    let wq: GGUFTensor
    let wo: GGUFTensor
    let ffnGate: GGUFTensor
    let ffnUp: GGUFTensor
    let ffnDown: GGUFTensor
    let nFF: Int

    init(_ g: GGUF, _ il: Int) {
        func t(_ s: String) -> GGUFTensor { g.tensor("assist.blk.\(il).\(s)") }
        attnNorm = t("attn_norm.weight")
        postAttnNorm = t("post_attn_norm.weight")
        ffnNorm = t("ffn_norm.weight")
        postFfnNorm = t("post_ffn_norm.weight")
        qNorm = t("attn_q_norm.weight")
        layerScalar = Dense.floats(t("layer_scalar")).first ?? 1
        wq = t("attn_q.weight")
        wo = t("attn_output.weight")
        ffnGate = t("ffn_gate.weight")
        ffnUp = t("ffn_up.weight")
        ffnDown = t("ffn_down.weight")
        nFF = ffnGate.dims[1]
    }
}

public struct Gemma4Assist {
    let nLayer: Int
    let nEmbd: Int
    let backbone: Int
    let nHead: Int
    let layerFull: [Bool]
    let layers: [Gemma4AssistLayer]
    let preProj: GGUFTensor
    let postProj: GGUFTensor
    let outputNorm: GGUFTensor
    let output: GGUFTensor
    let centroids: GGUFTensor?
    let tokenOrdering: GGUFTensor?

    var clustered: Bool { centroids != nil && tokenOrdering != nil }

    func isFull(_ il: Int) -> Bool { layerFull[il] }

    static func read(_ g: GGUF, _ trunk: Gemma4Config) -> Gemma4Assist? {
        var out: Gemma4Assist? = nil
        let n = g.int("gemma4.assist.block_count") ?? 0
        if n > 0, g.maybe("assist.pre_proj.weight") != nil {
            out = Gemma4Assist(g, n, trunk)
        }
        return out
    }

    private init(_ g: GGUF, _ n: Int, _ trunk: Gemma4Config) {
        nLayer = n
        nEmbd = g.int("gemma4.assist.embedding_length")!
        backbone = g.int("gemma4.assist.backbone_length")!
        nHead = g.int("gemma4.assist.attention.head_count")!
        layerFull = g.ints("gemma4.assist.layer_types")!.map { t in t == 1 }
        layers = (0..<n).map { il in Gemma4AssistLayer(g, il) }
        preProj = g.tensor("assist.pre_proj.weight")
        postProj = g.tensor("assist.post_proj.weight")
        outputNorm = g.tensor("assist.output_norm.weight")
        output = g.tensor("assist.token_embd.weight")
        centroids = g.maybe("assist.centroids.weight")
        tokenOrdering = g.maybe("assist.token_ordering")
        precondition(backbone == trunk.nEmbd,
                     "assist head fits a \(backbone)-wide trunk, this one "
                     + "is \(trunk.nEmbd)")
        precondition(preProj.dims[0] == 2 * backbone,
                     "assist pre_proj reads \(preProj.dims[0]), expected "
                     + "\(2 * backbone)")
        precondition(g.int("gemma4.assist.attention.head_count_kv")
                     == trunk.nHeadKV
                     && g.int("gemma4.assist.attention.global_head_count_kv")
                     == trunk.nHeadKVFull,
                     "assist kv-head counts must match the trunk's: it "
                     + "reads the trunk's pools")
        precondition(g.int("gemma4.assist.attention.key_length")
                     == trunk.headDimSliding
                     && g.int("gemma4.assist.attention.global_key_length")
                     == trunk.headDimFull,
                     "assist head widths must match the trunk's")
    }
}

public final class Gemma4Model {
    let gguf: GGUF
    public let cfg: Gemma4Config
    let layers: [Gemma4Layer]
    let tokEmbd: GGUFTensor        // [nEmbd, nVocab]
    // The whole per-layer-embedding apparatus, absent on a checkpoint that
    // has none (Gemma4Config.hasPerLayerInputs).
    let perLayerEmbd: GGUFTensor?  // [nLayer * perLayerDim, nVocab]
    // The CONTEXT half of the per-layer input: a projection of the token
    // embedding, normalized per layer slice. The gathered table alone is only
    // the token-identity half -- see project_per_layer_inputs and F29.
    let perLayerModelProj: GGUFTensor?  // [nEmbd, nLayer * perLayerDim]
    let perLayerProjNorm: [Float]       // [perLayerDim]
    let outputNorm: [Float]
    // lm_head, which a tied checkpoint spells as the embedding table itself.
    let output: GGUFTensor
    let assist: Gemma4Assist?

    public init(path: String) throws {
        gguf = try GGUF(path: path)
        cfg = try Gemma4Config(gguf)
        tokEmbd = gguf.tensor("token_embd.weight")
        perLayerEmbd = gguf.maybe("per_layer_token_embd.weight")
        perLayerModelProj = gguf.maybe("per_layer_model_proj.weight")
        perLayerProjNorm = gguf.maybe("per_layer_proj_norm.weight")
            .map(Dense.floats) ?? []
        outputNorm = Dense.floats(gguf.tensor("output_norm.weight"))
        output = gguf.maybe("output.weight") ?? tokEmbd
        let g = gguf, c = cfg
        layers = (0..<c.nLayer).map { il in Gemma4Layer(g, il, c) }
        assist = Gemma4Assist.read(g, c)
    }

    // Whether this file carries a tower for a modality, by the block count it
    // records for it. A UNIFIED checkpoint records zero and means it: its
    // images and audio go through a projection with no encoder behind it, so
    // "absent" here is the architecture rather than a truncated emit.
    public var hasVisionTower: Bool {
        (gguf.int("gemma4.vision.block_count") ?? 0) > 0
    }
    public var hasAudioTower: Bool {
        (gguf.int("gemma4.audio.block_count") ?? 0) > 0
    }

    // Whether a GGUF at `path` is a gemma4 file, by its own architecture key
    // -- the CLI routes on this rather than on a file name.
    public static func isGemma4(path: String) -> Bool {
        var result = false
        if let g = try? GGUF(path: path) {
            result = g.string("general.architecture") == "gemma4"
        }
        return result
    }
}

// A loaded gemma-4: engine + tokenizer + chat template out of the ONE file,
// the public door for consumers outside this module (GGUF and GemmaTokenizer
// are module-internal). The counterpart of BonsaiChat / MetalChat.
public struct GemmaChat {
    public let model: Gemma4Model
    public let engine: Gemma4Engine
    let tokenizer: GemmaTokenizer
    public let chatTemplate: String
    public let samplingPresets: SamplingPresets

    public init(ggufPath: String) throws {
        model = try Gemma4Model(path: ggufPath)
        engine = Gemma4Engine(model)
        tokenizer = try GemmaTokenizer(gguf: model.gguf)
        chatTemplate = model.gguf.string("tokenizer.chat_template") ?? ""
        samplingPresets = SamplingPresets.require(gguf: model.gguf,
                                                  path: ggufPath)
    }

    // Tokenizer surface, re-exported so a caller never names the internal
    // type. eosIds is the SET -- gemma stops on three ids (F23).
    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text, addSpecial: true)
    }
    public func decode(_ ids: [Int32]) -> String { tokenizer.decode(ids) }

    // Encoding a FRAGMENT that lands mid-turn, so no leading special.
    public func encodeRaw(_ text: String) -> [Int32] {
        tokenizer.encode(text, addSpecial: false)
    }
    public var eosIds: Set<Int32> { tokenizer.eosIds }
    public var vocabCount: Int { tokenizer.vocabCount }
    // Both towers and both scalars live in the one file, so the shape is a
    // read of what is already mapped.
    public var shape: ModelShape { ModelShape(gguf: model.gguf) }
    // The template opens every conversation with `{{- bos_token -}}` and
    // renders nothing there unless the spelling is supplied.
    public var bosToken: String { tokenizer.bosToken }

    // The AgentBackend seam, so ChatSession drives gemma exactly as it drives
    // Bonsai and the CoreML engine.
    public func backend() -> Gemma4Backend {
        Gemma4Backend(engine: engine, tokenizer: tokenizer)
    }

    // The GPU door onto the same weights and tokenizer. Built on demand: the
    // Metal engine allocates its own scratch and KV pools, which a text-only
    // SIMD run should not pay for.
    // The audio frontend and turn markup this file describes. Built on
    // demand: a text-only run should not pay to read them.
    public func audioWire() throws -> Gemma4AudioWire {
        try Gemma4AudioWire(model.gguf)
    }

    public func visionWire() throws -> Gemma4VisionWire {
        try Gemma4VisionWire(model.gguf)
    }

    public func videoWire() throws -> Gemma4VideoWire {
        try Gemma4VideoWire(model.gguf)
    }

    public var melConfig: Gemma4MelConfig {
        Gemma4MelConfig(model.gguf) ?? .processorDefault
    }

    // The attachment encoder for this model: a file becomes the SoftSpan a
    // turn carries for it. Handing it the metal backend's context is what puts
    // its towers on the GPU, over the mapping that backend already owns.
    public func media(ctx: MetalContext? = nil) -> Gemma4Media {
        Gemma4Media(self, ctx: ctx)
    }

    public func metalBackend() throws -> Gemma4MetalBackend {
        Gemma4MetalBackend(engine: try Gemma4MetalEngine(model),
                           tokenizer: tokenizer)
    }
}

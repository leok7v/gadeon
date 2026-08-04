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
    func isShared(_ il: Int) -> Bool { il >= firstShared }

    // The layer whose K/V a shared layer reads: the LAST non-shared layer of
    // the same type. HF marks it with store_full_length_kv; for E2B that is
    // L13 (sliding) and L14 (full), and L13 must therefore keep an unwindowed
    // history even though it is a sliding layer (F9).
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
    // repack drops the checkpoint's dead copies (F7).
    let wk: GGUFTensor?
    let wv: GGUFTensor?
    let kNorm: [Float]?

    let ffnGate: GGUFTensor
    let ffnUp: GGUFTensor
    let ffnDown: GGUFTensor
    let nFF: Int

    let perLayerGate: GGUFTensor  // [nEmbd -> perLayerDim]
    let perLayerProj: GGUFTensor  // [perLayerDim -> nEmbd]

    init(_ g: GGUF, _ il: Int, _ cfg: Gemma4Config) {
        func t(_ s: String) -> GGUFTensor { g.tensor("blk.\(il).\(s)") }
        func v(_ s: String) -> [Float] { Dense.floats(t(s)) }
        attnNorm = v("attn_norm.weight")
        postAttnNorm = v("post_attn_norm.weight")
        ffnNorm = v("ffn_norm.weight")
        postFfnNorm = v("post_ffn_norm.weight")
        perLayerPostNorm = v("per_layer_post_norm.weight")
        layerScalar = v("layer_scalar").first ?? 1
        wq = t("attn_q.weight")
        wo = t("attn_output.weight")
        qNorm = v("attn_q_norm.weight")
        let shared = cfg.isShared(il)
        wk = shared ? nil : t("attn_k.weight")
        wv = shared ? nil : t("attn_v.weight")
        kNorm = shared ? nil : v("attn_k_norm.weight")
        ffnGate = t("ffn_gate.weight")
        ffnUp = t("ffn_up.weight")
        ffnDown = t("ffn_down.weight")
        nFF = ffnGate.dims[1]
        perLayerGate = t("per_layer_gate.weight")
        perLayerProj = t("per_layer_proj.weight")
    }
}

public final class Gemma4Model {
    let gguf: GGUF
    public let cfg: Gemma4Config
    let layers: [Gemma4Layer]
    let tokEmbd: GGUFTensor        // [nEmbd, nVocab]
    let perLayerEmbd: GGUFTensor   // [nLayer * perLayerDim, nVocab]
    // The CONTEXT half of the per-layer input: a projection of the token
    // embedding, normalized per layer slice. The gathered table alone is only
    // the token-identity half -- see project_per_layer_inputs and F29.
    let perLayerModelProj: GGUFTensor   // [nEmbd, nLayer * perLayerDim]
    let perLayerProjNorm: [Float]       // [perLayerDim]
    let outputNorm: [Float]
    let output: GGUFTensor         // lm_head; untied on this checkpoint

    public init(path: String) throws {
        gguf = try GGUF(path: path)
        cfg = try Gemma4Config(gguf)
        tokEmbd = gguf.tensor("token_embd.weight")
        perLayerEmbd = gguf.tensor("per_layer_token_embd.weight")
        perLayerModelProj = gguf.tensor("per_layer_model_proj.weight")
        perLayerProjNorm = Dense.floats(
            gguf.tensor("per_layer_proj_norm.weight"))
        outputNorm = Dense.floats(gguf.tensor("output_norm.weight"))
        output = gguf.maybe("output.weight") ?? tokEmbd
        let g = gguf, c = cfg
        layers = (0..<c.nLayer).map { il in Gemma4Layer(g, il, c) }
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
        // The file's single suggested row fills all four cells; a file that
        // also carries the per-mode matrix overlays whichever cells it names.
        let row = SamplerConfig.from(gguf: model.gguf)
        samplingPresets = SamplingPresets.from(
            gguf: model.gguf,
            fallback: SamplingPresets(
                thinkingText: row, thinkingVision: row,
                nonThinkingText: row, nonThinkingVision: row))
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

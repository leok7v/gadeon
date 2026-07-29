// Hyper-parameters + per-layer tensor handles, read straight from the GGUF
// metadata, for the two ternary Bonsai lineages the one engine serves:
//   - qwen35: the 27B GDN hybrid (48 GDN + 16 attention layers, gated attention)
//   - qwen3:  the dense 1.7B (every layer plain ungated GQA, no state-space)
// Names/shapes follow llama.cpp's qwen35 and qwen3 loaders. The illustrative
// comments below are the 27B's values; the 1.7B differs (28 layers, 2048 wide,
// 16/8 heads, full 128-dim rotary, no ssm geometry).
import Foundation

public struct BonsaiConfig {
    public let nEmbd: Int          // 5120
    let nFF: Int            // 17408
    public let nLayer: Int         // 64
    let nHead: Int          // 24
    let nHeadKV: Int        // 4
    let headDim: Int        // 256  (key_length == value_length)
    let nRot: Int           // 64   (rope.dimension_count; partial rotary)
    let ropeBase: Float     // 1e7
    let ropeSections: [Int] // [11,11,10,0]
    let eps: Float          // 1e-6
    public let nVocab: Int         // 248320
    let fullAttnInterval: Int // 4  -> attention on layers where (i+1)%4==0
    // A dense qwen3: no state-space layers, no attention output gate, no fused
    // q|gate projection. Drives the tensor-name + ungated-attention branches.
    let dense: Bool

    // GDN geometry
    let dState: Int         // 128  (head_k_dim == head_v_dim)
    let nKHead: Int         // 16   (ssm_n_group)
    let nVHead: Int         // 48   (ssm_dt_rank)
    let dInner: Int         // 6144 (value_dim)
    let dConv: Int          // 4

    var keyDim: Int { dState * nKHead }     // 2048
    var valueDim: Int { dState * nVHead }   // 6144
    var convDim: Int { keyDim * 2 + valueDim } // 10240
    func isRecurrent(_ il: Int) -> Bool { (il + 1) % fullAttnInterval != 0 }

    init(_ g: GGUF) {
        let arch = g.string("general.architecture") ?? "qwen35"
        dense = arch == "qwen3"
        func i(_ k: String) -> Int { g.int("\(arch)." + k)! }
        nEmbd = i("embedding_length")
        nFF = i("feed_forward_length")
        nLayer = i("block_count")
        nHead = i("attention.head_count")
        nHeadKV = i("attention.head_count_kv")
        headDim = i("attention.key_length")
        // Dense qwen3 ships no rope.dimension_count: it rotates the whole head.
        nRot = g.int("\(arch).rope.dimension_count") ?? headDim
        ropeBase = Float(g.double("\(arch).rope.freq_base")!)
        ropeSections = g.ints("\(arch).rope.dimension_sections") ?? [11, 11, 10, 0]
        eps = Float(g.double("\(arch).attention.layer_norm_rms_epsilon")!)
        nVocab = g.tensor("token_embd.weight").dims[1]
        if dense {
            // fullAttnInterval 1 makes (il+1)%1 == 0 for every layer, so
            // isRecurrent is false everywhere: an all-attention stack.
            fullAttnInterval = 1
            dState = 0; nKHead = 0; nVHead = 0; dInner = 0; dConv = 0
        } else {
            fullAttnInterval = g.int("qwen35.full_attention_interval") ?? 4
            dState = i("ssm.state_size")
            nKHead = i("ssm.group_count")
            nVHead = i("ssm.time_step_rank")
            dInner = i("ssm.inner_size")
            dConv = i("ssm.conv_kernel")
        }
    }
}

// Per-layer weight handles. GDN layers use the ssm_/wqkv_ set; attention layers
// use the attn_ set. FFN + the two block norms are shared by both.
struct BonsaiLayer {
    let attnNorm: GGUFTensor       // input_layernorm
    let attnPostNorm: GGUFTensor   // post_attention_layernorm
    let ffnGate: GGUFTensor
    let ffnUp: GGUFTensor
    let ffnDown: GGUFTensor
    let recurrent: Bool

    // GDN (recurrent) layers
    var wqkv: GGUFTensor?          // in-proj q|k|v  [nEmbd, convDim]
    var wqkvGate: GGUFTensor?      // z gate          [nEmbd, valueDim]
    var ssmConv1d: GGUFTensor?     // [dConv, convDim] f32
    var ssmDt: GGUFTensor?         // [nVHead] f32
    var ssmA: GGUFTensor?          // [nVHead] f32  (= -exp(A_log))
    var ssmBeta: GGUFTensor?       // [nEmbd, nVHead]
    var ssmAlpha: GGUFTensor?      // [nEmbd, nVHead]
    var ssmNorm: GGUFTensor?       // [dState] f32
    var ssmOut: GGUFTensor?        // [valueDim, nEmbd]

    // attention layers
    var wq: GGUFTensor?            // [nEmbd, headDim*2*nHead]  (q | gate fused)
    var wk: GGUFTensor?            // [nEmbd, headDim*nHeadKV]
    var wv: GGUFTensor?
    var wo: GGUFTensor?            // [headDim*nHead, nEmbd]
    var qNorm: GGUFTensor?         // [headDim]
    var kNorm: GGUFTensor?

    init(_ g: GGUF, _ il: Int, _ cfg: BonsaiConfig) {
        func t(_ s: String) -> GGUFTensor { g.tensor("blk.\(il).\(s)") }
        attnNorm = t("attn_norm.weight")
        // Dense qwen3 names the pre-FFN norm ffn_norm; the hybrid uses
        // post_attention_norm. Same slot in the block, different tensor name.
        attnPostNorm = cfg.dense ? t("ffn_norm.weight")
                                 : t("post_attention_norm.weight")
        ffnGate = t("ffn_gate.weight")
        ffnUp = t("ffn_up.weight")
        ffnDown = t("ffn_down.weight")
        recurrent = cfg.isRecurrent(il)
        if recurrent {
            wqkv = t("attn_qkv.weight")
            wqkvGate = t("attn_gate.weight")
            ssmConv1d = t("ssm_conv1d.weight")
            ssmDt = t("ssm_dt.bias")
            ssmA = t("ssm_a")
            ssmBeta = t("ssm_beta.weight")
            ssmAlpha = t("ssm_alpha.weight")
            ssmNorm = t("ssm_norm.weight")
            ssmOut = t("ssm_out.weight")
        } else {
            wq = t("attn_q.weight")
            wk = t("attn_k.weight")
            wv = t("attn_v.weight")
            wo = t("attn_output.weight")
            qNorm = t("attn_q_norm.weight")
            kNorm = t("attn_k_norm.weight")
        }
    }
}

public final class BonsaiModel {
    let gguf: GGUF
    public let cfg: BonsaiConfig
    let layers: [BonsaiLayer]
    let tokEmbd: GGUFTensor
    let outputNorm: GGUFTensor
    let output: GGUFTensor        // lm_head (tied == tokEmbd)

    public init(path: String) throws {
        gguf = try GGUF(path: path)
        cfg = BonsaiConfig(gguf)
        tokEmbd = gguf.tensor("token_embd.weight")
        outputNorm = gguf.tensor("output_norm.weight")
        output = gguf.maybe("output.weight") ?? tokEmbd
        let g = gguf, c = cfg
        layers = (0..<c.nLayer).map { BonsaiLayer(g, $0, c) }
    }
}

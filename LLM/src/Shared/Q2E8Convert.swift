import Foundation

enum ConvertErr: Error, CustomStringConvertible {
    case unmapped(String)
    case shape(String)
    case config(String)

    var description: String {
        let out: String
        switch self {
        case let .unmapped(m): out = "convert: no mapping for \(m)"
        case let .shape(m): out = "convert: \(m)"
        case let .config(m): out = "convert: config.json \(m)"
        }
        return out
    }
}

enum HeadAxis { case none, rows, cols }

struct Xform {
    var plusOne = false
    var negExp = false
    var axis = HeadAxis.none
    var offset = 0
    var width = 0
    var temporal = -1
}

struct TextShape {
    let nEmbd: Int
    let nFF: Int
    let nLayer: Int
    let nHead: Int
    let nHeadKV: Int
    let headDim: Int
    let nRot: Int
    let ropeBase: Double
    let ropeSections: [Int]
    let eps: Double
    let nVocab: Int
    let fullAttnInterval: Int
    let dState: Int
    let nKHead: Int
    let nVHead: Int
    let dConv: Int
    let context: Int
    let mtpLayers: Int
    let tied: Bool

    var keyDim: Int { dState * nKHead }
    var valueDim: Int { dState * nVHead }
    var convDim: Int { keyDim * 2 + valueDim }
    var nRep: Int { nVHead / nKHead }

    init(_ root: [String: Any]) throws {
        let text = root["text_config"] as? [String: Any] ?? root
        func i(_ k: String) throws -> Int {
            let v = (text[k] as? NSNumber)?.intValue
            if v == nil { throw ConvertErr.config("missing \(k)") }
            return v!
        }
        let rope = text["rope_parameters"] as? [String: Any] ?? [:]
        let theta = (rope["rope_theta"] as? NSNumber)?.doubleValue
        let partial = (rope["partial_rotary_factor"] as? NSNumber)?
            .doubleValue ?? 1.0
        let sections = (rope["mrope_section"] as? [NSNumber])?
            .map { n in n.intValue } ?? []
        if theta == nil { throw ConvertErr.config("missing rope_theta") }
        let keyHead = try i("linear_key_head_dim")
        let valueHead = try i("linear_value_head_dim")
        if keyHead != valueHead {
            throw ConvertErr.config(
                "key head \(keyHead) != value head \(valueHead)")
        }
        nEmbd = try i("hidden_size")
        nFF = try i("intermediate_size")
        nLayer = try i("num_hidden_layers")
        nHead = try i("num_attention_heads")
        nHeadKV = try i("num_key_value_heads")
        headDim = try i("head_dim")
        nRot = Int(Double(headDim) * partial)
        ropeBase = theta!
        ropeSections = sections + [Int](repeating: 0, count: 4 - sections.count)
        eps = (text["rms_norm_eps"] as? NSNumber)?.doubleValue ?? 1e-6
        nVocab = try i("vocab_size")
        fullAttnInterval = try i("full_attention_interval")
        dState = keyHead
        nKHead = try i("linear_num_key_heads")
        nVHead = try i("linear_num_value_heads")
        dConv = try i("linear_conv_kernel_dim")
        context = try i("max_position_embeddings")
        mtpLayers = (text["mtp_num_hidden_layers"] as? NSNumber)?
            .intValue ?? 0
        tied = (text["tie_word_embeddings"] as? NSNumber)?.boolValue ?? false
    }

    var headOrder: [Int] {
        (0..<nVHead).map { i in (i % nKHead) * nRep + i / nKHead }
    }
}

struct VisionShape {
    let depth: Int
    let embd: Int
    let ff: Int
    let heads: Int
    let patch: Int
    let merge: Int
    let projDim: Int
    let positions: Int
    let eps: Double
    let mean: [Float]
    let std: [Float]

    var imageSize: Int {
        patch * Int(Double(positions).squareRoot().rounded())
    }

    init?(_ root: [String: Any], _ pre: [String: Any]) {
        let v = root["vision_config"] as? [String: Any]
        if v == nil { return nil }
        func i(_ k: String, _ fallback: Int) -> Int {
            (v![k] as? NSNumber)?.intValue ?? fallback
        }
        depth = i("depth", 0)
        embd = i("hidden_size", 0)
        ff = i("intermediate_size", 0)
        heads = i("num_heads", 0)
        patch = i("patch_size", 16)
        merge = i("spatial_merge_size", 2)
        projDim = i("out_hidden_size", 0)
        positions = i("num_position_embeddings", 0)
        eps = (v!["layer_norm_eps"] as? NSNumber)?.doubleValue ?? 1e-6
        mean = (pre["image_mean"] as? [NSNumber])?
            .map { n in Float(n.doubleValue) } ?? [0.5, 0.5, 0.5]
        std = (pre["image_std"] as? [NSNumber])?
            .map { n in Float(n.doubleValue) } ?? [0.5, 0.5, 0.5]
    }
}

struct Q2E8Policy {
    let trunk: GGUFType
    let embd: GGUFType
    let wide: Set<String>
    let fit: Q2Fit
    let imat: String?
    let hess: String?
    let gptq: Bool
    let ternary: Bool
    let lowRank: Bool
    let rotate: Bool
    let vision: Bool
    let raise: [String: GGUFType]
    let spin: Bool
    let untie: Bool
    let head: GGUFType
    let reconstruct: Bool
    let e8: Bool
    let e8p: Bool
    let rank: Int
    let mu: Float
    let damp: Double
}

struct Q2E8Tensor {
    let gguf: String
    let source: String
    let dims: [Int]
    let type: GGUFType
    let xform: Xform
    let rank: Int
}

enum Q2E8Map {

    static let lang: [String: String] = [
        "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
        "linear_attn.in_proj_z.weight": "attn_gate.weight",
        "linear_attn.out_proj.weight": "ssm_out.weight",
        "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
        "linear_attn.in_proj_b.weight": "ssm_beta.weight",
        "linear_attn.norm.weight": "ssm_norm.weight",
        "linear_attn.dt_bias": "ssm_dt.bias",
        "linear_attn.A_log": "ssm_a",
        "linear_attn.conv1d.weight": "ssm_conv1d.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    ]

    static let nextn: [String: String] = [
        "mtp.fc.weight": "nextn.eh_proj.weight",
        "mtp.pre_fc_norm_embedding.weight": "nextn.enorm.weight",
        "mtp.pre_fc_norm_hidden.weight": "nextn.hnorm.weight",
        "mtp.norm.weight": "nextn.shared_head_norm.weight",
    ]

    static let vision: [String: String] = [
        "attn.qkv.weight": "attn_qkv.weight",
        "attn.qkv.bias": "attn_qkv.bias",
        "attn.proj.weight": "attn_out.weight",
        "attn.proj.bias": "attn_out.bias",
        "mlp.linear_fc1.weight": "ffn_up.weight",
        "mlp.linear_fc1.bias": "ffn_up.bias",
        "mlp.linear_fc2.weight": "ffn_down.weight",
        "mlp.linear_fc2.bias": "ffn_down.bias",
        "norm1.weight": "ln1.weight",
        "norm1.bias": "ln1.bias",
        "norm2.weight": "ln2.weight",
        "norm2.bias": "ln2.bias",
    ]

    static let visionTop: [String: String] = [
        "model.visual.patch_embed.proj.bias": "v.patch_embd.bias",
        "model.visual.pos_embed.weight": "v.position_embd.weight",
        "model.visual.merger.norm.weight": "v.post_ln.weight",
        "model.visual.merger.norm.bias": "v.post_ln.bias",
        "model.visual.merger.linear_fc1.weight": "mm.0.weight",
        "model.visual.merger.linear_fc1.bias": "mm.0.bias",
        "model.visual.merger.linear_fc2.weight": "mm.2.weight",
        "model.visual.merger.linear_fc2.bias": "mm.2.bias",
    ]

    static let plusOne: Set<String> = [
        "attn_norm.weight", "post_attention_norm.weight",
        "attn_q_norm.weight", "attn_k_norm.weight",
        "nextn.enorm.weight", "nextn.hnorm.weight",
        "nextn.shared_head_norm.weight",
    ]

    static let readerNorm: [String: String] = [
        "attn_qkv.weight": "input_layernorm",
        "attn_gate.weight": "input_layernorm",
        "ssm_alpha.weight": "input_layernorm",
        "ssm_beta.weight": "input_layernorm",
        "attn_q.weight": "input_layernorm",
        "attn_k.weight": "input_layernorm",
        "attn_v.weight": "input_layernorm",
        "ffn_gate.weight": "post_attention_layernorm",
        "ffn_up.weight": "post_attention_layernorm",
    ]

    static let writers: Set<String> = [
        "ssm_out.weight", "attn_output.weight", "ffn_down.weight",
    ]

    static let streamNorms: Set<String> = [
        "attn_norm.weight", "post_attention_norm.weight",
    ]

    static let keepF32: Set<String> = [
        "ssm_conv1d.weight", "ssm_alpha.weight", "ssm_beta.weight",
        "ssm_norm.weight",
    ]

    static func headwise(_ leaf: String, _ s: TextShape) -> Xform {
        var out = Xform()
        switch leaf {
        case "attn_qkv.weight":
            out.axis = .rows; out.offset = 2 * s.keyDim; out.width = s.dState
        case "attn_gate.weight":
            out.axis = .rows; out.width = s.dState
        case "ssm_conv1d.weight":
            out.axis = .rows; out.offset = 2 * s.keyDim; out.width = s.dState
        case "ssm_out.weight":
            out.axis = .cols; out.width = s.dState
        case "ssm_alpha.weight", "ssm_beta.weight", "ssm_dt.bias":
            out.axis = .rows; out.width = 1
        case "ssm_a":
            out.axis = .rows; out.width = 1; out.negExp = true
        default:
            out.plusOne = plusOne.contains(leaf)
        }
        if s.nRep == 1 && leaf != "ssm_a" { out.axis = .none }
        return out
    }

    static func leaf(_ name: String, _ prefix: String) -> String {
        var out = ""
        if name.hasPrefix(prefix) {
            let rest = name.dropFirst(prefix.count)
            let cut = rest.firstIndex(of: ".")
            if let cut, Int(rest[rest.startIndex..<cut]) != nil {
                out = String(rest[rest.index(after: cut)...])
            }
        }
        return out
    }

    static func index(_ name: String, _ prefix: String) -> Int {
        var out = -1
        if name.hasPrefix(prefix) {
            let rest = name.dropFirst(prefix.count)
            let cut = rest.firstIndex(of: ".") ?? rest.endIndex
            out = Int(rest[rest.startIndex..<cut]) ?? -1
        }
        return out
    }
}

enum StreamRole {
    case none
    case reader(String)
    case writer
    case embed
    case head
    case blank
}

final class Spin {
    let sign: [Float]
    let nEmbd: Int
    private let store: Safetensors
    private var cache: [String: [Float]] = [:]

    init(_ store: Safetensors, _ nEmbd: Int) {
        self.store = store
        self.nEmbd = nEmbd
        sign = Hadamard.signs(nEmbd)
    }

    func gain(_ hf: String) -> [Float] {
        if let hit = cache[hf] { return hit }
        var v = (try? store.values(hf)) ?? []
        for i in 0..<v.count { v[i] += 1 }
        cache[hf] = v
        return v
    }

    static func role(_ gguf: String, _ s: TextShape) -> StreamRole {
        var out = StreamRole.none
        let part = gguf.split(separator: ".")
        if gguf == "token_embd.weight" {
            out = .embed
        } else if gguf == "output.weight" {
            out = .head
        } else if gguf == "output_norm.weight" {
            out = .blank
        } else if gguf == "mm.2.weight" {
            out = .writer
        } else if part.count > 2, part[0] == "blk", let il = Int(part[1]) {
            let leaf = part[2...].joined(separator: ".")
            if let norm = Q2E8Map.readerNorm[leaf] {
                out = .reader("model.language_model.layers.\(il).\(norm).weight")
            } else if Q2E8Map.writers.contains(leaf) {
                out = .writer
            } else if Q2E8Map.streamNorms.contains(leaf) {
                out = .blank
            }
        }
        return out
    }

    func apply(_ values: inout [Float], _ t: Q2E8Tensor, _ s: TextShape) {
        let k = t.dims[0]
        let rows = values.count / k
        switch Spin.role(t.gguf, s) {
        case .blank:
            for i in 0..<values.count { values[i] = 1 }
        case .embed:
            Hadamard.rows(&values, rows, k, sign)
        case .head:
            scale(&values, rows, k, gain("model.language_model.norm.weight"))
            Hadamard.rows(&values, rows, k, sign)
        case let .reader(norm):
            scale(&values, rows, k, gain(norm))
            Hadamard.rows(&values, rows, k, sign)
        case .writer:
            Hadamard.cols(&values, rows, k, sign)
        case .none:
            break
        }
    }

    private func scale(_ values: inout [Float], _ rows: Int, _ k: Int,
                       _ g: [Float]) {
        if g.count == k {
            for r in 0..<rows {
                for c in 0..<k { values[r * k + c] *= g[c] }
            }
        }
    }
}

struct HessianCache {
    var key = ""
    var h: [Float] = []
    var hinv: [Float] = []
    var values: [Double] = []
    var vectors: [Double] = []
    var spun: [Float] = []
    var spunInv: [Float] = []
}

public struct Q2E8Report: Sendable {
    public var lines: [String] = []
    public var worstCosine = (1.0, "")
    public var worstExact = (0.0, "")
    public var worstError = (Float(0), "")
    public var bytes = 0
}

public struct Q2E8Convert {

    public static func run(from dir: String, to path: String,
                           trunk name: String, embd: String,
                           wide: Set<String>, renorm: Bool, imat: String?,
                           hess: String?, gptq: Bool, lowRank: Bool,
                           rotate: Bool, vision: Bool, raise: String,
                           spin: Bool, untie: Bool, head: String,
                           gate: String?, reconstruct: Bool, e8: Bool, e8p: Bool,
                           rank: Int, mu: Float, damp: Double,
                           log: (String) -> Void) throws -> Q2E8Report {
        let trunk = Q2E8Convert.types[name]
        let table = Q2E8Convert.types[embd.isEmpty ? name : embd]
        if trunk == nil { throw ConvertErr.config("trunk \(name)") }
        if table == nil { throw ConvertErr.config("embd \(embd)") }
        let policy = Q2E8Policy(
            trunk: trunk!, embd: table!, wide: wide,
            fit: Q2Fit(iterations: 8, renorm: renorm && !gptq),
            imat: imat, hess: hess, gptq: gptq,
            ternary: name == "ternary", lowRank: lowRank, rotate: rotate,
            vision: vision, raise: promotions(raise), spin: spin,
            untie: untie || spin,
            head: types[head.isEmpty ? embd : head] ?? table!,
            reconstruct: reconstruct || rank > 0 || (e8 && !e8p)
                || (e8p && trunk! != .q2_e8),
            e8: e8 || e8p, e8p: e8p, rank: rank, mu: mu,
            damp: damp > 0 ? damp : Q2GPTQ.damp)
        return try emit(dir, path, policy, gate, log)
    }

    static func promotions(_ spec: String) -> [String: GGUFType] {
        var out: [String: GGUFType] = [:]
        for item in spec.split(separator: ",") {
            let pair = item.split(separator: "=")
            if pair.count == 2, let t = types[String(pair[1])] {
                out[String(pair[0])] = t
            }
        }
        return out
    }

    static let types: [String: GGUFType] = [
        "q2_e8": .q2_e8, "q4_0": .q4_0, "q8_0": .q8_0, "ternary": .q2_0,
        "bf16": .bf16, "f16": .f16, "f32": .f32,
    ]

    static func emit(_ dir: String, _ path: String, _ p: Q2E8Policy,
                     _ gate: String?,
                     _ log: (String) -> Void) throws -> Q2E8Report {
        let root = try json(dir + "/config.json")
        let pre = (try? json(dir + "/preprocessor_config.json")) ?? [:]
        let text = try TextShape(root)
        let eye = VisionShape(root, pre)
        let store = try Safetensors(dir: dir)
        let spin = p.spin ? Spin(store, text.nEmbd) : nil
        let plan = try plans(store, text, eye, p)
        let writer = try GGUFWriter(path: path)
        try meta(writer, dir, root, text, eye, storage(p.trunk, p), p.vision)
        for t in plan {
            writer.declare(t.gguf, dims: t.dims, type: storage(t.type, p))
        }
        writer.finishHeader()
        log("\(plan.count) tensors, \(writer.declaredBytes / 1_000_000) MB "
            + "-> \(path)")
        let reference = try gate.map { at in try GGUF(path: at) }
        var cache = HessianCache()
        var report = Q2E8Report()
        var lattice = E8Stats()
        var lowBytes = 0
        var lowWeights = 0
        var lowGap = 0.0
        report.bytes = writer.declaredBytes
        for (i, t) in plan.enumerated() {
            let raw = try store.values(t.source)
            var values = apply(raw, t, text)
            spin?.apply(&values, t, text)
            var fit = p.fit
            fit.importance = importance(t.gguf, t.dims[0], p.imat)
            let site = squares(t, p, &cache)
            let spinning = p.rotate && !site.spun.isEmpty
            if spinning {
                Hadamard.rows(&values, values.count / t.dims[0], t.dims[0],
                              Hadamard.signs(t.dims[0]))
            }
            var feed = values
            var lift: [Float] = []
            if p.rank > 0 && quantized(t.type) {
                let k = t.dims[0]
                let rows = values.count / k
                let got = LowRank.split(values, rows, k, p.rank)
                lift = got.0
                var gap = 0.0
                for i in 0..<feed.count {
                    feed[i] -= lift[i]
                    gap += Double(feed[i]) * Double(feed[i])
                }
                lowBytes += p.rank * (rows + k) * 2
                lowWeights += values.count
                lowGap = max(lowGap, abs(gap - got.1) / max(got.1, 1e-30))
            }
            let (bytes, back0) = encode(feed, t, fit, p.ternary,
                                        p.gptq ? (spinning ? site.spunInv
                                                           : site.hinv) : [],
                                        p.e8 && quantized(t.type),
                                        p.e8p, p.mu, &lattice)
            var back = back0
            for i in 0..<lift.count { back[i] += lift[i] }
            let payload = storage(t.type, p) == t.type ? bytes
                                                       : BF16.encode(back)
            try payload.withUnsafeBufferPointer { b in
                try writer.append(UnsafeRawBufferPointer(b), expecting: t.gguf)
            }
            let mark = curve(t, values, back, site)
                + relative(t, values, back,
                           spinning ? site.spun : site.h, &report)
                + judge(t, back, reference, !site.h.isEmpty, &report)
            log("  [\(i + 1)/\(plan.count)] " + pad(t.gguf, 34)
                + pad(label(t.type), 6) + pad("\(t.dims)", 18) + mark)
        }
        try writer.close()
        if lowWeights > 0 {
            log(String(format: "[rank%d] +%.3f bits/weight over %d weights, "
                       + "%.1f MB, worst residual gap %.2e  %@", p.rank,
                       8 * Double(lowBytes) / Double(lowWeights), lowWeights,
                       Double(lowBytes) / 1e6, lowGap,
                       lowGap < 1e-4 ? "PASS" : "FAIL -- not the top-r space"))
        }
        if lattice.vectors > 0 {
            let sum = E8P.checksum()
            log(String(format: "[e8p] table %d entries, %d even, sum %.6f  %@",
                       sum.0, sum.1, sum.2,
                       sum.0 == 256 ? "" : "FAIL -- table is not 256 entries"))
            log(String(format: "[e8] %d vectors, %.4f%% walked back into the "
                       + "shell, peak norm2 %.1f of %.1f  %@", lattice.vectors,
                       100 * Double(lattice.clamped) / Double(lattice.vectors),
                       lattice.peak, E8.bound(p.e8p),
                       lattice.peak <= E8.bound(p.e8p) ? "PASS"
                           : "FAIL -- codebook is not 2 bits, result is void"))
        }
        return report
    }

    static func pad(_ text: String, _ width: Int) -> String {
        text.count >= width ? text + " "
            : text + String(repeating: " ", count: width - text.count)
    }

    static func meta(_ w: GGUFWriter, _ dir: String,
                     _ root: [String: Any], _ s: TextShape,
                     _ eye: VisionShape?, _ trunk: GGUFType,
                     _ wantsEye: Bool) throws {
        w.meta("general.architecture", .string("qwen35"))
        w.meta("general.type", .string("model"))
        w.meta("general.name",
               .string((dir as NSString).lastPathComponent))
        w.meta("general.quantization_version", .u32(2))
        w.meta("general.alignment", .u32(16384))
        w.meta("general.file_type",
               .u32(UInt32(bitPattern: trunk.rawValue)))
        w.meta("qwen35.block_count", .u32(UInt32(s.nLayer + s.mtpLayers)))
        w.meta("qwen35.context_length", .u32(UInt32(s.context)))
        w.meta("qwen35.embedding_length", .u32(UInt32(s.nEmbd)))
        w.meta("qwen35.feed_forward_length", .u32(UInt32(s.nFF)))
        w.meta("qwen35.attention.head_count", .u32(UInt32(s.nHead)))
        w.meta("qwen35.attention.head_count_kv", .u32(UInt32(s.nHeadKV)))
        w.meta("qwen35.attention.key_length", .u32(UInt32(s.headDim)))
        w.meta("qwen35.attention.value_length", .u32(UInt32(s.headDim)))
        w.meta("qwen35.attention.layer_norm_rms_epsilon",
               .f32(Float(s.eps)))
        w.meta("qwen35.rope.freq_base", .f32(Float(s.ropeBase)))
        w.meta("qwen35.rope.dimension_count", .u32(UInt32(s.nRot)))
        w.meta("qwen35.rope.dimension_sections",
               .i32s(s.ropeSections.map { v in Int32(v) }))
        w.meta("qwen35.full_attention_interval",
               .u32(UInt32(s.fullAttnInterval)))
        w.meta("qwen35.ssm.conv_kernel", .u32(UInt32(s.dConv)))
        w.meta("qwen35.ssm.state_size", .u32(UInt32(s.dState)))
        w.meta("qwen35.ssm.group_count", .u32(UInt32(s.nKHead)))
        w.meta("qwen35.ssm.time_step_rank", .u32(UInt32(s.nVHead)))
        w.meta("qwen35.ssm.inner_size", .u32(UInt32(s.valueDim)))
        if s.mtpLayers > 0 {
            w.meta("qwen35.nextn_predict_layers",
                   .u32(UInt32(s.mtpLayers)))
        }
        if let eye, wantsEye { vision(w, eye) }
        try tokenizer(w, dir, s)
        sidecars(w, dir)
    }

    static func vision(_ w: GGUFWriter, _ e: VisionShape) {
        w.meta("clip.has_vision_encoder", .bool(true))
        w.meta("clip.projector_type", .string("qwen3vl_merger"))
        w.meta("clip.use_gelu", .bool(true))
        w.meta("clip.vision.projection_dim", .u32(UInt32(e.projDim)))
        w.meta("clip.vision.image_size", .u32(UInt32(e.imageSize)))
        w.meta("clip.vision.patch_size", .u32(UInt32(e.patch)))
        w.meta("clip.vision.embedding_length", .u32(UInt32(e.embd)))
        w.meta("clip.vision.feed_forward_length", .u32(UInt32(e.ff)))
        w.meta("clip.vision.block_count", .u32(UInt32(e.depth)))
        w.meta("clip.vision.attention.head_count", .u32(UInt32(e.heads)))
        w.meta("clip.vision.spatial_merge_size", .u32(UInt32(e.merge)))
        w.meta("clip.vision.attention.layer_norm_epsilon",
               .f32(Float(e.eps)))
        w.meta("clip.vision.image_mean", .f32s(e.mean))
        w.meta("clip.vision.image_std", .f32s(e.std))
    }

    static func tokenizer(_ w: GGUFWriter, _ dir: String,
                          _ s: TextShape) throws {
        let spec = try json(dir + "/tokenizer.json")
        let cfg = (try? json(dir + "/tokenizer_config.json")) ?? [:]
        let model = spec["model"] as? [String: Any] ?? [:]
        var tokens = [String](repeating: "", count: s.nVocab)
        var types = [Int32](repeating: 5, count: s.nVocab)
        for (piece, any) in model["vocab"] as? [String: Any] ?? [:] {
            let id = (any as? NSNumber)?.intValue ?? -1
            if id >= 0 && id < s.nVocab {
                tokens[id] = piece
                types[id] = 1
            }
        }
        for (key, any) in cfg["added_tokens_decoder"]
            as? [String: Any] ?? [:] {
            let id = Int(key) ?? -1
            let item = any as? [String: Any] ?? [:]
            let content = item["content"] as? String ?? ""
            let flag = (item["special"] as? NSNumber)?.boolValue ?? false
            if id >= 0 && id < s.nVocab && !content.isEmpty {
                tokens[id] = content
                types[id] = flag ? 3 : 4
            }
        }
        for i in 0..<s.nVocab where tokens[i].isEmpty {
            tokens[i] = "[PAD\(i)]"
        }
        w.meta("tokenizer.ggml.model", .string("gpt2"))
        w.meta("tokenizer.ggml.pre", .string("qwen35"))
        w.meta("tokenizer.ggml.tokens", .strings(tokens))
        w.meta("tokenizer.ggml.token_type", .i32s(types))
        w.meta("tokenizer.ggml.merges", .strings(merges(model)))
        w.meta("tokenizer.ggml.add_bos_token", .bool(false))
        var byPiece: [String: Int32] = [:]
        for (i, piece) in tokens.enumerated() { byPiece[piece] = Int32(i) }
        special(w, "eos", cfg["eos_token"], byPiece, "<|im_end|>")
        special(w, "padding", cfg["pad_token"], byPiece, "<|endoftext|>")
        let jinja = try? String(contentsOfFile: dir + "/chat_template.jinja",
                                encoding: .utf8)
        if let jinja { w.meta("tokenizer.chat_template", .string(jinja)) }
    }

    static func merges(_ model: [String: Any]) -> [String] {
        var out: [String] = []
        for item in model["merges"] as? [Any] ?? [] {
            if let one = item as? String {
                out.append(one)
            } else if let pair = item as? [String], pair.count == 2 {
                out.append(pair[0] + " " + pair[1])
            }
        }
        return out
    }

    static func special(_ w: GGUFWriter, _ role: String, _ any: Any?,
                        _ byPiece: [String: Int32], _ fallback: String) {
        let named = (any as? String)
            ?? ((any as? [String: Any])?["content"] as? String)
            ?? fallback
        if let id = byPiece[named] ?? byPiece[fallback] {
            w.meta("tokenizer.ggml.\(role)_token_id", .u32(UInt32(id)))
        }
    }

    static func sidecars(_ w: GGUFWriter, _ dir: String) {
        let listing = ((try? FileManager.default
            .contentsOfDirectory(atPath: dir)) ?? []).sorted()
        for name in listing where carried(name) {
            let text = try? String(contentsOfFile: dir + "/" + name,
                                   encoding: .utf8)
            if let text {
                w.meta("qwen35.source."
                       + name.replacingOccurrences(of: ".", with: "_"),
                       .string(text))
            }
        }
    }

    static func carried(_ name: String) -> Bool {
        name == "config.json" || name.hasSuffix("_config.json")
            || name.hasSuffix(".jinja")
    }

    static func json(_ path: String) throws -> [String: Any] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let root = try JSONSerialization.jsonObject(with: data)
        return root as? [String: Any] ?? [:]
    }

    static func label(_ t: GGUFType) -> String {
        let out: String
        switch t {
        case .f32: out = "F32"
        case .f16: out = "F16"
        case .bf16: out = "BF16"
        case .q4_0: out = "Q4_0"
        case .q8_0: out = "Q8_0"
        case .q2_0: out = "TQ"
        case .q2_e8: out = "Q2_E8"
        default: out = "?"
        }
        return out
    }

    static func plans(_ store: Safetensors, _ s: TextShape,
                      _ eye: VisionShape?,
                      _ p: Q2E8Policy) throws -> [Q2E8Tensor] {
        var out: [Q2E8Tensor] = []
        for name in store.names where !name.hasSuffix(".weight_scale_inv") {
            let seen = try map(name, store, s, eye, p)
            out += p.vision ? seen : seen.filter { t in t.rank != 4 }
        }
        out.sort { a, b in
            a.rank == b.rank ? a.gguf < b.gguf : a.rank < b.rank
        }
        return out
    }

    static func map(_ name: String, _ store: Safetensors, _ s: TextShape,
                    _ eye: VisionShape?,
                    _ p: Q2E8Policy) throws -> [Q2E8Tensor] {
        let shape = try store.shape(name)
        let layers = "model.language_model.layers."
        let blocks = "model.visual.blocks."
        var out: [Q2E8Tensor] = []
        if name == "model.language_model.embed_tokens.weight" {
            out = [make("token_embd.weight", name, shape, Xform(), 0,
                        s, p)]
            if p.untie && s.tied {
                out.append(make("output.weight", name, shape, Xform(), 3,
                                s, p))
            }
        } else if name == "lm_head.weight" {
            out = [make("output.weight", name, shape, Xform(), 3,
                        s, p)]
        } else if name == "model.language_model.norm.weight" {
            var x = Xform(); x.plusOne = true
            out = [make("output_norm.weight", name, shape, x, 3,
                        s, p)]
        } else if name.hasPrefix(layers) {
            out = [try block(name, Q2E8Map.leaf(name, layers),
                             Q2E8Map.index(name, layers), shape, s, p)]
        } else if let tail = Q2E8Map.nextn[name] {
            var x = Xform(); x.plusOne = Q2E8Map.plusOne.contains(tail)
            out = [make("blk.\(s.nLayer).\(tail)", name, shape, x,
                        2, s, p)]
        } else if name.hasPrefix("mtp.layers.") {
            out = [try block(name, Q2E8Map.leaf(name, "mtp.layers."),
                             s.nLayer, shape, s, p)]
        } else if name == "model.visual.patch_embed.proj.weight" {
            out = try patch(name, shape, s, eye)
        } else if let tail = Q2E8Map.visionTop[name] {
            out = [make(tail, name, shape, Xform(), 4, s, p)]
        } else if name.hasPrefix(blocks) {
            let tail = Q2E8Map.vision[Q2E8Map.leaf(name, blocks)]
            let il = Q2E8Map.index(name, blocks)
            if tail == nil || il < 0 { throw ConvertErr.unmapped(name) }
            out = [make("v.blk.\(il).\(tail!)", name, shape, Xform(), 4,
                        s, p)]
        } else {
            throw ConvertErr.unmapped(name)
        }
        return out
    }

    static func block(_ name: String, _ tail: String, _ il: Int,
                      _ shape: [Int], _ s: TextShape,
                      _ p: Q2E8Policy) throws -> Q2E8Tensor {
        let leaf = Q2E8Map.lang[tail]
        if leaf == nil || il < 0 { throw ConvertErr.unmapped(name) }
        return make("blk.\(il).\(leaf!)", name, shape,
                    Q2E8Map.headwise(leaf!, s), il == s.nLayer ? 2 : 1, s, p)
    }

    static func patch(_ name: String, _ shape: [Int], _ s: TextShape,
                      _ eye: VisionShape?) throws -> [Q2E8Tensor] {
        if shape.count != 5 || eye == nil {
            throw ConvertErr.shape("\(name): shape \(shape)")
        }
        let slice = [shape[0], shape[1], shape[3], shape[4]]
        var out: [Q2E8Tensor] = []
        for t in 0..<shape[2] {
            var x = Xform()
            x.temporal = t
            out.append(Q2E8Tensor(
                gguf: t == 0 ? "v.patch_embd.weight"
                             : "v.patch_embd.weight.\(t)",
                source: name, dims: dims(slice), type: .f16, xform: x,
                rank: 4))
        }
        return out
    }

    static func dims(_ shape: [Int]) -> [Int] {
        let out = shape.reversed().filter { d in d != 1 }
        return out.isEmpty ? [1] : Array(out)
    }

    static func make(_ gguf: String, _ source: String, _ shape: [Int],
                     _ xform: Xform, _ rank: Int, _ s: TextShape,
                     _ p: Q2E8Policy) -> Q2E8Tensor {
        let d = dims(shape)
        return Q2E8Tensor(gguf: gguf, source: source, dims: d,
                         type: kind(gguf, d, s, p), xform: xform, rank: rank)
    }

    static func kind(_ gguf: String, _ dims: [Int], _ s: TextShape,
                     _ p: Q2E8Policy) -> GGUFType {
        let leaf = gguf.hasPrefix("blk.")
            ? gguf.split(separator: ".").dropFirst(2).joined(separator: ".")
            : gguf
        let out: GGUFType
        if dims.count < 2 {
            out = .f32
        } else if gguf == "output.weight" {
            out = p.head
        } else if gguf == "token_embd.weight" {
            out = p.embd
        } else if gguf == "v.position_embd.weight" {
            out = .f32
        } else if gguf.hasPrefix("v.") || gguf.hasPrefix("mm.") {
            out = .f16
        } else if gguf.hasPrefix("blk.\(s.nLayer).") {
            out = .bf16
        } else if Q2E8Map.keepF32.contains(leaf) {
            out = .f32
        } else if let bumped = p.raise[leaf], dims[0] % Q80.qk == 0 {
            out = bumped
        } else if p.wide.contains(leaf) && dims[0] % Q40.qk == 0 {
            out = .q4_0
        } else if dims[0] % Q2E8.qk == 0 {
            out = p.trunk
        } else {
            out = .f32
        }
        return out
    }

    static func apply(_ raw: [Float], _ t: Q2E8Tensor,
                      _ s: TextShape) -> [Float] {
        var out = raw
        if t.xform.negExp {
            for i in 0..<out.count { out[i] = -Foundation.exp(out[i]) }
        }
        if t.xform.plusOne {
            for i in 0..<out.count { out[i] += 1 }
        }
        if t.xform.temporal >= 0 {
            out = slice(out, t)
        }
        if t.xform.axis != .none {
            out = permute(out, t, s)
        }
        return out
    }

    static func slice(_ raw: [Float], _ t: Q2E8Tensor) -> [Float] {
        let span = t.dims[0] * t.dims[1]
        let groups = t.dims[2] * t.dims[3]
        let stride = raw.count / groups
        var out = [Float](repeating: 0, count: groups * span)
        for g in 0..<groups {
            let from = g * stride + t.xform.temporal * span
            for i in 0..<span { out[g * span + i] = raw[from + i] }
        }
        return out
    }

    static func permute(_ raw: [Float], _ t: Q2E8Tensor,
                        _ s: TextShape) -> [Float] {
        let cols = t.xform.axis == .rows
            ? raw.count / t.dims[t.dims.count - 1] : t.dims[0]
        let rows = raw.count / cols
        let order = s.headOrder
        var out = raw
        for (dst, src) in order.enumerated() {
            let lo = t.xform.offset + dst * t.xform.width
            let from = t.xform.offset + src * t.xform.width
            for w in 0..<t.xform.width {
                if t.xform.axis == .rows {
                    let a = (lo + w) * cols
                    let b = (from + w) * cols
                    for c in 0..<cols { out[a + c] = raw[b + c] }
                } else {
                    for r in 0..<rows {
                        out[r * cols + lo + w] = raw[r * cols + from + w]
                    }
                }
            }
        }
        return out
    }

    static func encode(_ values: [Float], _ t: Q2E8Tensor, _ fit: Q2Fit,
                       _ ternary: Bool, _ hinv: [Float], _ e8: Bool,
                       _ padded: Bool, _ mu: Float,
                       _ stats: inout E8Stats) -> ([UInt8], [Float]) {
        let out: ([UInt8], [Float])
        switch t.type {
        case .f32:
            var bytes = [UInt8](repeating: 0, count: values.count * 4)
            bytes.withUnsafeMutableBufferPointer { b in
                values.withUnsafeBytes { src in
                    _ = memcpy(b.baseAddress!, src.baseAddress!, src.count)
                }
            }
            out = (bytes, values)
        case .f16:
            out = (F16.encode(values), values)
        case .bf16:
            out = (BF16.encode(values), values)
        case .q4_0:
            out = Q40.quantize(values)
        case .q8_0:
            out = Q80.quantize(values)
        default:
            let k = t.dims[0]
            let rows = values.count / k
            if e8 {
                let packer = t.type == .q2_e8 && padded
                let got = hinv.isEmpty
                    ? E8.quantize(values, rows: rows, k: k, opts: fit,
                                  padded: padded, mu: mu)
                    : E8.gptq(values, rows: rows, k: k, opts: fit,
                              hinv: hinv, padded: padded, mu: mu)
                stats.vectors += got.1.vectors
                stats.clamped += got.1.clamped
                stats.peak = max(stats.peak, got.1.peak)
                let bytes = packer && !got.1.codes.isEmpty
                    ? Q2E8Pack.e8p(got.1.codes, got.1.scales, rows: rows, k: k)
                    : []
                out = (bytes, got.0)
            } else if !hinv.isEmpty {
                out = Q2GPTQ.quantize(values, rows: rows, k: k,
                                      soa: t.type == .q2_e8, opts: fit,
                                      ternary: ternary, hinv: hinv)
            } else if ternary {
                out = Ternary.quantize(values, rows: rows, k: k,
                                       importance: fit.importance)
            } else {
                out = Q2E8.quantize(values, rows: rows, k: k,
                                   soa: t.type == .q2_e8, opts: fit)
            }
        }
        return out
    }

    static let site: [String: String] = [
        "attn_qkv.weight": "in", "attn_gate.weight": "in",
        "attn_q.weight": "in", "attn_k.weight": "in", "attn_v.weight": "in",
        "ffn_gate.weight": "post", "ffn_up.weight": "post",
        "ffn_down.weight": "ffn_down", "ssm_out.weight": "ssm_out",
        "attn_output.weight": "attn_out",
    ]

    static func siteOf(_ gguf: String) -> String {
        var out = ""
        let part = gguf.split(separator: ".")
        if part.count > 2 && part[0] == "blk", let il = Int(part[1]),
           let s = site[part[2...].joined(separator: ".")] {
            out = "l\(il).\(s)"
        }
        return out
    }

    static func importance(_ gguf: String, _ k: Int,
                           _ dir: String?) -> [Float]? {
        var out: [Float]? = nil
        let key = siteOf(gguf)
        if let dir, !key.isEmpty,
           let raw = FileManager.default.contents(
               atPath: dir + "/" + key + ".bin"), raw.count == k * 4 {
            var v = [Float](repeating: 0, count: k)
            v.withUnsafeMutableBytes { dst in
                _ = raw.copyBytes(to: dst)
            }
            let mean = v.reduce(0.0) { a, x in a + Double(x) } / Double(k)
            if mean > 0 {
                out = v.map { x in x / Float(mean) }
            }
        }
        return out
    }

    static func squares(_ t: Q2E8Tensor, _ p: Q2E8Policy,
                        _ cache: inout HessianCache) -> HessianCache {
        var out = HessianCache()
        let key = siteOf(t.gguf)
        let k = t.dims[0]
        let usable = quantized(t.type) && !key.isEmpty
        if let dir = p.hess, usable {
            if cache.key == key {
                out = cache
            } else if let raw = FileManager.default.contents(
                          atPath: dir + "/" + key + ".h32"),
                      raw.count == k * k * 4 {
                var h = [Float](repeating: 0, count: k * k)
                h.withUnsafeMutableBytes { dst in _ = raw.copyBytes(to: dst) }
                let basis = p.lowRank ? LowRank.basis(h, k)
                                      : ([Double](), [Double]())
                var spun: [Float] = []
                var spunInv: [Float] = []
                if p.rotate {
                    spun = h
                    Hadamard.congruence(&spun, k, Hadamard.signs(k))
                    spunInv = Q2GPTQ.inverseCholesky(spun, k, damp: p.damp)
                }
                out = HessianCache(key: key, h: h,
                                   hinv: Q2GPTQ.inverseCholesky(h, k, damp: p.damp),
                                   values: basis.0, vectors: basis.1,
                                   spun: spun, spunInv: spunInv)
                cache = out
            }
        }
        return out
    }

    static func quantized(_ type: GGUFType) -> Bool {
        type == .q2_e8 || type == .q2_0
    }

    // --reconstruct keeps the quantizer and swaps only the CONTAINER, so a
    // codebook nobody has written a kernel for still yields a file that
    // llama.cpp can score. Quality is exact; size and speed are not.

    static func lossy(_ type: GGUFType) -> Bool {
        quantized(type) || type == .q4_0 || type == .q8_0
    }

    static func storage(_ type: GGUFType, _ p: Q2E8Policy) -> GGUFType {
        p.reconstruct && lossy(type) ? .bf16 : type
    }

    static func curve(_ t: Q2E8Tensor, _ values: [Float], _ back: [Float],
                      _ site: HessianCache) -> String {
        var out = ""
        if !site.vectors.isEmpty {
            let k = t.dims[0]
            var delta = values
            for i in 0..<delta.count { delta[i] -= back[i] }
            let rows = delta.count / k
            let best = LowRank.remaining(delta, rows, k, site.values,
                                         site.vectors)
            let fixed = LowRank.shared(delta, rows, k, site.values,
                                       site.vectors)
            out = "best " + zip(LowRank.ranks, best).map { r, v in
                "\(r):" + String(format: "%.2f", v)
            }.joined(separator: " ")
                + "  shared " + zip(LowRank.ranks, fixed).map { r, v in
                    "\(r):" + String(format: "%.2f", v)
                }.joined(separator: " ") + "  "
        }
        return out
    }

    static func relative(_ t: Q2E8Tensor, _ values: [Float], _ back: [Float],
                         _ h: [Float], _ report: inout Q2E8Report) -> String {
        var out = ""
        if !h.isEmpty {
            let k = t.dims[0]
            var delta = values
            for i in 0..<delta.count { delta[i] -= back[i] }
            let e = Q2GPTQ.outputError(delta, values, rows: values.count / k,
                                       k: k, h)
            out = String(format: "err %.4f  ", e)
            if e > report.worstError.0 { report.worstError = (e, t.gguf) }
        }
        return out
    }

    static func judge(_ t: Q2E8Tensor, _ back: [Float], _ reference: GGUF?,
                      _ scored: Bool, _ report: inout Q2E8Report) -> String {
        var out = ""
        if let ref = reference?.maybe(t.gguf) {
            if !Q2E8Convert.readable.contains(ref.type) {
                out = "ref is \(label(ref.type)), not comparable"
            } else if ref.count != back.count {
                out = "count \(back.count) vs ref \(ref.count)  <-- LOW"
            } else {
                out = compare(t, back, Dense.floats(ref), scored, &report)
            }
        }
        return out
    }

    static let readable: Set<GGUFType> = [.f32, .f16, .bf16, .q8_0]

    static func compare(_ t: Q2E8Tensor, _ back: [Float], _ want: [Float],
                        _ scored: Bool,
                        _ report: inout Q2E8Report) -> String {
        var dot = 0.0, na = 0.0, nb = 0.0, gap = 0.0, peak = 0.0
        for i in 0..<back.count {
            dot += Double(back[i]) * Double(want[i])
            na += Double(back[i]) * Double(back[i])
            nb += Double(want[i]) * Double(want[i])
            gap = max(gap, Double(abs(back[i] - want[i])))
            peak = max(peak, Double(abs(want[i])))
        }
        let cos = dot / ((na * nb).squareRoot() + 1e-30)
        let rel = gap / max(peak, 1e-30)
        let exact = t.type == .f32 || t.type == .bf16
        var out: String
        if exact {
            out = String(format: "maxdiff %.2e rel %.1e", gap, rel)
            if rel > report.worstExact.0 {
                report.worstExact = (rel, t.gguf)
            }
            if rel > 1e-6 { out += "  <-- LOW" }
        } else {
            out = String(format: "cos %.4f", cos)
            if cos < report.worstCosine.0 {
                report.worstCosine = (cos, t.gguf)
            }
            if cos < 0.85 && !scored { out += "  <-- LOW" }
        }
        return out
    }
}

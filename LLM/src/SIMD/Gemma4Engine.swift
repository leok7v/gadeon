// The gemma-4-E2B text forward, one token at a time, mirroring
// Gemma4TextDecoderLayer.forward in transformers 5.14.1 exactly:
//
//   r = h;  h = attn(rms(h, attn_norm));  h = rms(h, post_attn_norm);  h = r+h
//   r = h;  h = mlp(rms(h, ffn_norm));    h = rms(h, post_ffn_norm);   h = r+h
//   r = h;  h = per_layer_proj(gelu(per_layer_gate(h)) * ple[il])
//           h = rms(h, per_layer_post_norm);                           h = r+h
//   h *= layer_scalar
//
// The post-* norms sit BEFORE their residual add (Gemma's sandwich), which is
// where this differs from the Bonsai/Qwen block that reuses the second norm as
// the pre-FFN one. Three seams the plan calls out are load-bearing here:
// attention scaling is 1.0 (F4), V is RMS-normalized without a weight (F6),
// and the full-attention layers rotate only their first 64 pairs (F5).
import Foundation

// One attention layer's K/V history. Sliding layers may evict, EXCEPT the one
// that a shared layer reads from -- L13 is a sliding layer that must keep an
// unwindowed history because layers 15+ read past its own window (F9).
final class GemmaKV {
    private(set) var k: [[Float]] = []
    private(set) var v: [[Float]] = []
    // Absolute position of k[0]; grows as a windowed store evicts.
    private(set) var first = 0
    private let keep: Int?

    init(keep: Int?) { self.keep = keep }

    func append(_ kk: [Float], _ vv: [Float]) {
        k.append(kk)
        v.append(vv)
        if let keep, k.count > keep {
            k.removeFirst()
            v.removeFirst()
            first += 1
        }
    }

    var end: Int { first + k.count }

    func reset() {
        k.removeAll()
        v.removeAll()
        first = 0
    }

    struct Snapshot: Sendable {
        let k: [[Float]]
        let v: [[Float]]
        let first: Int
    }
    func snapshot() -> Snapshot { Snapshot(k: k, v: v, first: first) }
    func restore(_ s: Snapshot) {
        k = s.k
        v = s.v
        first = s.first
    }
}

public final class Gemma4Engine {
    let model: Gemma4Model
    public let cfg: Gemma4Config
    // One store per NON-shared layer (0..14); shared layers read their
    // source's.
    var kv: [Int: GemmaKV] = [:]
    public private(set) var pos = 0
    var sampler: Sampler?
    // The same lock-backed stop the GPU engine carries: this forward is
    // synchronous too, so Task cancellation never reaches it and a Stop during
    // a long prefill would otherwise be ignored entirely.
    private let stopSignal = MetalStopSignal()
    public func requestStop() { stopSignal.raise() }
    public func shouldStop() -> Bool { stopSignal.raisedNow }
    // Reused scratch for the per-token PLE gather: one row of the
    // [nLayer*perLayerDim, nVocab] table covers every layer's input.
    private var pleRow: [Float]
    // The trained activation clamps, keyed by weight name. Empty on a file
    // that predates them, where every clamp is the identity.
    private let srq: [String: SRQ]

    public init(_ model: Gemma4Model) {
        self.model = model
        cfg = model.cfg
        pleRow = [Float](repeating: 0,
                         count: cfg.nLayer * cfg.perLayerDim)
        var scales: [String: SRQ] = [:]
        for L in model.layers {
            for w in [L.wq, L.wo, L.wk, L.wv, L.ffnGate, L.ffnUp, L.ffnDown,
                      L.perLayerGate, L.perLayerProj] {
                if let w { scales[w.name] = SRQ(model.gguf, w.name) }
            }
        }
        scales[model.output.name] = SRQ(model.gguf, model.output.name)
        srq = scales
        for il in 0..<cfg.nLayer where !cfg.isShared(il) {
            // A shared layer's source keeps everything; every other sliding
            // layer only ever reads its own window.
            let source = cfg.layerFull.indices.contains(il)
                && (0..<cfg.nLayer).contains { j in
                    cfg.isShared(j) && cfg.sharedSource(j) == il
                }
            let keep = (cfg.isFull(il) || source) ? nil : cfg.slidingWindow
            kv[il] = GemmaKV(keep: keep)
        }
    }

    public func reset() {
        pos = 0
        stopSignal.clear()
        for (_, store) in kv { store.reset() }
    }

    // Prefill onto the CURRENT state, returning the prediction after the last
    // id. Token by token: the sliding/full split and the shared histories make
    // a batched form a separate piece of work, and correctness comes first.
    public func extend(_ ids: [Int32]) -> Int32 {
        stopSignal.clear()
        var hidden = [Float]()
        var i = 0
        // Stop as the loop predicate, so it lands on a token boundary and
        // leaves `pos` and the KV consistent.
        while i < ids.count && !stopSignal.raisedNow {
            hidden = forward(token: Int(ids[i]), pos: pos)
            pos += 1
            i += 1
        }
        // A stop before the first token leaves no hidden state, and the
        // lm_head cannot be run on an empty vector. The caller throws
        // EngineError.stopped and discards this either way.
        return hidden.isEmpty ? 0 : pick(logits(hidden))
    }

    public func decode(_ token: Int32) -> Int32 {
        let hidden = forward(token: Int(token), pos: pos)
        pos += 1
        return pick(logits(hidden))
    }

    // Prefill a turn that carries soft tokens: `softAt` returns a tower
    // feature for an id that is a placeholder and nil for a real token, so
    // one turn can carry SEVERAL modalities -- image and audio placeholders
    // are different ids. The loop lives here because the position is this
    // engine's to advance.
    public func extend(_ ids: [Int32],
                       softAt: (Int32) -> [Float]?) -> Int32 {
        stopSignal.clear()
        let e = cfg.nEmbd
        let blocks = cfg.blockwiseVision
            ? cfg.visionBlocks(ids, from: pos)
            : [(Int, Int)](repeating: (0, 0), count: ids.count)
        // The per-layer-embedding checkpoints stay on the token path: their
        // PLE is gathered and averaged per token, which the batched layers do
        // not carry. They are also the ones the GPU already prefills fast.
        let width = cfg.hasPerLayerInputs ? 1 : Gemma4Engine.batch
        var hidden = [Float]()
        var i = 0
        while i < ids.count && !stopSignal.raisedNow {
            let n = Gemma4Config.chunkLength(
                blocks, at: i, want: min(width, ids.count - i))
            if n == 1 {
                if let feature = softAt(ids[i]) {
                    hidden = forward(embedding: feature, pos: pos)
                } else {
                    hidden = forward(token: Int(ids[i]), pos: pos)
                }
            } else {
                var x = [Float](repeating: 0, count: n * e)
                for j in 0..<n {
                    let id = ids[i + j]
                    // A soft row enters UNSCALED; embed_scale is for text.
                    let row = softAt(id) ?? embed(Int(id))
                    for c in 0..<e { x[j * e + c] = row[c] }
                }
                layersBatch(&x, n: n, basePos: pos,
                            Array(blocks[i..<(i + n)]))
                hidden = GK.rmsnorm(
                    Array(x[((n - 1) * e)..<(n * e)]),
                    model.outputNorm, cfg.eps)
            }
            pos += n
            i += n
        }
        return hidden.isEmpty ? 0 : pick(logits(hidden))
    }

    // Tokens per prefill chunk on the CPU. The same lever the GPU engine
    // reads, so one setting describes a run whichever engine drives it.
    static let batch: Int = {
        let raw = ProcessInfo.processInfo.environment["LLM_GEMMA_BATCH"]
        return max(1, raw.flatMap { v in Int(v) } ?? 128)
    }()

    func pick(_ logits: [Float]) -> Int32 {
        var out: Int32
        if sampler != nil {
            var work = logits
            let picked = sampler!.sample(&work)
            sampler!.accept(picked)
            out = picked
        } else {
            out = Int32(argmax(logits))
        }
        return out
    }

    public func argmax(_ v: [Float]) -> Int {
        var bi = 0
        var bv = -Float.greatestFiniteMagnitude
        for i in 0..<v.count where v[i] > bv { bv = v[i]; bi = i }
        return bi
    }

    // A parked conversation's bytes. The bookmark already holds the K/V it
    // needs, so this is a pure format change -- no engine state is read here
    // and rollback stays what it was.

    public func serialize(_ b: Bookmark) -> Data {
        var out = Data()
        StateBytes.putHeader(&out)
        StateBytes.putInt(&out, b.pos)
        StateBytes.putInt(&out, b.kv.count)
        for il in b.kv.keys.sorted() {
            let s = b.kv[il]!
            StateBytes.putInt(&out, il)
            StateBytes.putInt(&out, s.first)
            StateBytes.putInt(&out, s.k.count)
            for row in s.k { StateBytes.putFloats(&out, row) }
            for row in s.v { StateBytes.putFloats(&out, row) }
        }
        return out
    }

    public func deserialize(_ data: Data) -> Bookmark? {
        let b = [UInt8](data)
        var p = 0
        var mark: Bookmark? = nil
        if StateBytes.readHeader(b, &p) {
            let pos = StateBytes.getInt(b, &p)
            var kv: [Int: GemmaKV.Snapshot] = [:]
            for _ in 0..<StateBytes.getInt(b, &p) {
                let il = StateBytes.getInt(b, &p)
                let first = StateBytes.getInt(b, &p)
                let rows = StateBytes.getInt(b, &p)
                var k: [[Float]] = [], v: [[Float]] = []
                for _ in 0..<rows { k.append(StateBytes.getFloats(b, &p)) }
                for _ in 0..<rows { v.append(StateBytes.getFloats(b, &p)) }
                kv[il] = GemmaKV.Snapshot(k: k, v: v, first: first)
            }
            mark = Bookmark(pos: pos, kv: kv)
        }
        return mark
    }

    public struct Bookmark: @unchecked Sendable {
        let pos: Int
        let kv: [Int: GemmaKV.Snapshot]
    }

    public func bookmark() -> Bookmark {
        var s: [Int: GemmaKV.Snapshot] = [:]
        for (il, store) in kv { s[il] = store.snapshot() }
        return Bookmark(pos: pos, kv: s)
    }

    public func restore(_ b: Bookmark) {
        pos = b.pos
        for (il, s) in b.kv { kv[il]!.restore(s) }
    }

    // ---- pieces ---------------------------------------------------------

    // One QUANTIZED linear with the trained activation clamp around it. Every
    // quantized projection in this engine routes through here, so a site
    // cannot silently skip the clamp: the shapes are identical either way and
    // only the answer changes.
    private func linear(_ w: GGUFTensor, _ x: [Float],
                        _ out: inout [Float]) {
        let s = srq[w.name] ?? SRQ.none
        var h = x
        SRQ.apply(&h, s.input)
        GQ.matvec(w, x: h, out: &out)
        SRQ.apply(&out, s.output)
    }

    // token_embd row * sqrt(hidden_size): Gemma4TextScaledWordEmbedding
    // multiplies the embedding by embed_scale, an architectural constant the
    // repack recorded rather than baking into the weights.
    func embed(_ token: Int) -> [Float] {
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        GQ.dequantSpan(model.tokEmbd, row: token, from: 0,
                       count: cfg.nEmbd, into: &out)
        let s = cfg.embedScale
        for i in 0..<cfg.nEmbd { out[i] *= s }
        return out
    }

    // The per-layer inputs for one token, both halves. The gathered table is
    // only the TOKEN-IDENTITY half; HF also projects the (already scaled)
    // token embedding through per_layer_model_projection, scales it by
    // 1/sqrt(hidden_size), RMS-normalizes each 256-wide layer slice, and
    // averages the two halves by 1/sqrt(2). Using the gather alone is close
    // enough to produce fluent-looking numbers and wrong everywhere (F29).
    private func buildPLE(_ token: Int, _ embed: [Float]) {
        GQ.dequantSpan(model.perLayerEmbd!, row: token, from: 0,
                       count: pleRow.count, into: &pleRow)
        let s = cfg.perLayerEmbedScale
        for i in 0..<pleRow.count { pleRow[i] *= s }

        var proj = [Float](repeating: 0, count: pleRow.count)
        GQ.matvec(model.perLayerModelProj!, x: embed, out: &proj)
        let ps = 1 / Float(cfg.nEmbd).squareRoot()
        for i in 0..<proj.count { proj[i] *= ps }
        GK.rmsnormRows(&proj, d: cfg.perLayerDim, rows: cfg.nLayer,
                       w: model.perLayerProjNorm, eps: cfg.eps)

        let half = 1 / Float(2).squareRoot()
        for i in 0..<pleRow.count { pleRow[i] = (proj[i] + pleRow[i]) * half }
    }

    private func mlp(_ x: [Float], _ L: Gemma4Layer) -> [Float] {
        var gate = [Float](repeating: 0, count: L.nFF)
        var up = [Float](repeating: 0, count: L.nFF)
        linear(L.ffnGate, x, &gate)
        linear(L.ffnUp, x, &up)
        let act = cfg.activation
        for i in 0..<L.nFF { gate[i] = act.apply(gate[i]) * up[i] }
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        linear(L.ffnDown, gate, &out)
        return out
    }

    // per_layer_input_gate -> gelu -> * ple slice -> per_layer_projection.
    private func perLayer(_ x: [Float], _ L: Gemma4Layer,
                          _ il: Int) -> [Float] {
        let d = cfg.perLayerDim
        var g = [Float](repeating: 0, count: d)
        linear(L.perLayerGate!, x, &g)
        let off = il * d
        let act = cfg.activation
        for i in 0..<d { g[i] = act.apply(g[i]) * pleRow[off + i] }
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        linear(L.perLayerProj!, g, &out)
        return out
    }

    // ---- batched prefill ---------------------------------------------------
    // WHY, and it is not only speed: a vision block's tokens read FORWARD
    // across the block, so every one of its keys must be in the store before
    // any of its queries runs. A per-token forward has appended only 0..q, so
    // it cannot express that mask at all -- batching is what makes the 12B
    // read an image correctly on the CPU, not just faster.
    //
    // The speed is the other half. A mat-VEC unpacks a 32-weight block for one
    // dot product and discards it, so a 294-token prefill unpacked every
    // weight in the model 294 times; GQ.matmul unpacks once for all N rows.

    private func layersBatch(_ x: inout [Float], n: Int, basePos: Int,
                             _ blocks: [(Int, Int)]) {
        let e = cfg.nEmbd
        for il in 0..<cfg.nLayer {
            let L = model.layers[il]
            var h = x
            GK.rmsnormRows(&h, d: e, rows: n, w: L.attnNorm, eps: cfg.eps)
            var attn = attentionBatch(h, L, il, basePos: basePos, n: n, blocks)
            GK.rmsnormRows(&attn, d: e, rows: n, w: L.postAttnNorm,
                           eps: cfg.eps)
            for i in 0..<(n * e) { x[i] += attn[i] }

            h = x
            GK.rmsnormRows(&h, d: e, rows: n, w: L.ffnNorm, eps: cfg.eps)
            var ff = mlpBatch(h, L, n)
            GK.rmsnormRows(&ff, d: e, rows: n, w: L.postFfnNorm, eps: cfg.eps)
            for i in 0..<(n * e) { x[i] += ff[i] }

            let s = L.layerScalar
            for i in 0..<(n * e) { x[i] *= s }
        }
    }

    // One quantized linear over N rows, with the trained clamp on both sides.
    // SRQ is elementwise, so a batch clamps exactly as N single rows would.
    private func linearBatch(_ w: GGUFTensor, _ X: [Float], _ n: Int,
                             _ outDim: Int) -> [Float] {
        let s = srq[w.name] ?? SRQ.none
        var h = X
        SRQ.apply(&h, s.input)
        var out = [Float](repeating: 0, count: n * outDim)
        GQ.matmul(w, X: h, N: n, out: &out)
        SRQ.apply(&out, s.output)
        return out
    }

    private func mlpBatch(_ h: [Float], _ L: Gemma4Layer,
                          _ n: Int) -> [Float] {
        var gate = linearBatch(L.ffnGate, h, n, L.nFF)
        let up = linearBatch(L.ffnUp, h, n, L.nFF)
        let act = cfg.activation
        for i in 0..<(n * L.nFF) { gate[i] = act.apply(gate[i]) * up[i] }
        return linearBatch(L.ffnDown, gate, n, cfg.nEmbd)
    }

    // Every row's K/V is appended BEFORE any row attends, which is the whole
    // point: a query inside a vision block reads keys that come after it.
    private func attentionBatch(_ h: [Float], _ L: Gemma4Layer, _ il: Int,
                                basePos: Int, n: Int,
                                _ blocks: [(Int, Int)]) -> [Float] {
        let hd = cfg.headDim(il)
        let nH = cfg.nHead
        let nKV = L.nHeadKV
        let full = cfg.isFull(il)
        let base = full ? cfg.ropeBaseFull : cfg.ropeBaseSliding
        let rot = full ? cfg.rotatedPairsFull : cfg.rotatedPairsSliding
        var q = linearBatch(L.wq, h, n, hd * nH)
        GK.rmsnormRows(&q, d: hd, rows: n * nH, w: L.qNorm, eps: cfg.eps)
        ropeRows(&q, hd, nH, rot, base, basePos, n)
        let store = kv[cfg.isShared(il) ? cfg.sharedSource(il) : il]!
        if !cfg.isShared(il) {
            var k = linearBatch(L.wk!, h, n, hd * nKV)
            var v = linearBatch(L.wv ?? L.wk!, h, n, hd * nKV)
            GK.rmsnormRows(&k, d: hd, rows: n * nKV, w: L.kNorm!, eps: cfg.eps)
            ropeRows(&k, hd, nKV, rot, base, basePos, n)
            GK.rmsnormRowsNoWeight(&v, d: hd, rows: n * nKV, eps: cfg.eps)
            let width = hd * nKV
            for j in 0..<n {
                store.append(Array(k[(j * width)..<((j + 1) * width)]),
                             Array(v[(j * width)..<((j + 1) * width)]))
            }
        }
        var ctx = [Float](repeating: 0, count: n * hd * nH)
        for j in 0..<n {
            let row = Array(q[(j * hd * nH)..<((j + 1) * hd * nH)])
            let out = attend(row, store, il, pos: basePos + j,
                             block: blocks[j], hd: hd,
                             nH: nH, nKV: nKV)
            for i in 0..<(hd * nH) { ctx[j * hd * nH + i] = out[i] }
        }
        return linearBatch(L.wo, ctx, n, cfg.nEmbd)
    }

    // gemma's rope over N rows, each at its own absolute position.
    private func ropeRows(_ t: inout [Float], _ hd: Int, _ heads: Int,
                          _ rot: Int, _ base: Float, _ basePos: Int,
                          _ n: Int) {
        let width = hd * heads
        for j in 0..<n {
            var row = Array(t[(j * width)..<((j + 1) * width)])
            GK.rope(&row, headDim: hd, nHead: heads, rotated: rot,
                    base: base, pos: basePos + j)
            for i in 0..<width { t[j * width + i] = row[i] }
        }
    }

    private func attention(_ n: [Float], _ L: Gemma4Layer, _ il: Int,
                           pos: Int, block: (Int, Int) = (0, 0)) -> [Float] {
        let hd = cfg.headDim(il)
        let nH = cfg.nHead
        let nKV = L.nHeadKV
        let full = cfg.isFull(il)
        let base = full ? cfg.ropeBaseFull : cfg.ropeBaseSliding
        let rot = full ? cfg.rotatedPairsFull : cfg.rotatedPairsSliding

        var q = [Float](repeating: 0, count: hd * nH)
        linear(L.wq, n, &q)
        GK.rmsnormRows(&q, d: hd, rows: nH, w: L.qNorm, eps: cfg.eps)
        GK.rope(&q, headDim: hd, nHead: nH, rotated: rot, base: base,
                pos: pos)

        // A shared layer owns no projections: it reads the last non-shared
        // layer of its own type, whose history is stored unwindowed.
        let store = kv[cfg.isShared(il) ? cfg.sharedSource(il) : il]!
        if !cfg.isShared(il) {
            var k = [Float](repeating: 0, count: hd * nKV)
            linear(L.wk!, n, &k)
            // A layer with no value projection takes the KEY projection as its
            // value -- projected again from the same input rather than copied
            // out of `k`, which is what keeps k_norm and rope below from
            // reaching it.
            var v = [Float](repeating: 0, count: hd * nKV)
            linear(L.wv ?? L.wk!, n, &v)
            GK.rmsnormRows(&k, d: hd, rows: nKV, w: L.kNorm!, eps: cfg.eps)
            GK.rope(&k, headDim: hd, nHead: nKV, rotated: rot, base: base,
                    pos: pos)
            // v_norm is Gemma4RMSNorm(with_scale=False): a real op that
            // leaves no tensor in the checkpoint, so a tensor scan cannot
            // see it (F6).
            GK.rmsnormRowsNoWeight(&v, d: hd, rows: nKV, eps: cfg.eps)
            store.append(k, v)
        }

        let attnOut = attend(q, store, il, pos: pos, block: block,
                             hd: hd, nH: nH, nKV: nKV)
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        linear(L.wo, attnOut, &out)
        return out
    }

    // One query against a store. The RANGE is the mask and no per-key test is
    // needed: a vision block always CONTAINS its query, so [lo, pos] and
    // [block.0, block.1) overlap and their union is the contiguous
    // [lo, max(pos + 1, block.1)). The window still bounds it from below, so
    // an image longer than the window is cut exactly as HF's AND cuts it.
    // An empty block reduces this to the plain causal range.
    private func attend(_ q: [Float], _ store: GemmaKV, _ il: Int,
                        pos: Int, block: (Int, Int), hd: Int, nH: Int,
                        nKV: Int) -> [Float] {
        let full = cfg.isFull(il)
        let lo = full ? store.first
                      : max(store.first, pos - cfg.slidingWindow + 1)
        let hi = min(store.end, max(pos + 1, block.1))
        let group = nH / nKV
        var out = [Float](repeating: 0, count: hd * nH)
        var scores = [Float](repeating: 0, count: max(hi - lo, 1))
        for h in 0..<nH {
            let kvh = h / group
            let qh = h * hd
            var t = lo
            while t < hi {
                let row = store.k[t - store.first]
                var s: Float = 0
                for i in 0..<hd { s += q[qh + i] * row[kvh * hd + i] }
                // self.scaling = 1.0 -- q_norm is the only normalization of
                // the query, there is no 1/sqrt(head_dim) anywhere (F4).
                scores[t - lo] = s
                t += 1
            }
            GK.softmaxInPlace(&scores, hi - lo)
            t = lo
            while t < hi {
                let w = scores[t - lo]
                let row = store.v[t - store.first]
                for i in 0..<hd { out[qh + i] += w * row[kvh * hd + i] }
                t += 1
            }
        }
        return out
    }

    // One token at absolute position `pos`; returns the post-final-norm
    // hidden. `tap` observes named intermediates for the offline oracle gate.
    @discardableResult
    public func forward(token: Int, pos: Int,
                        tap: ((String, Int, [Float]) -> Void)? = nil)
        -> [Float] {
        let x = embed(token)
        if cfg.hasPerLayerInputs { buildPLE(token, x) }
        return layers(x, pos: pos, tap: tap)
    }

    // One SOFT token: a tower's projected feature stands in for an embedding.
    // Two things differ from a text token and neither is guessable. The
    // per-layer input gathers the PAD row, not the placeholder's -- HF
    // rewrites every multimodal position to pad_token_id before the gather --
    // while the CONTEXT half still projects the feature itself. And the
    // feature enters UNSCALED: embed_scale multiplies text embeddings only.
    @discardableResult
    public func forward(embedding: [Float], pos: Int,
                        tap: ((String, Int, [Float]) -> Void)? = nil)
        -> [Float] {
        if cfg.hasPerLayerInputs { buildPLE(cfg.padTokenId, embedding) }
        return layers(embedding, pos: pos, tap: tap)
    }

    private func layers(_ start: [Float], pos: Int,
                        tap: ((String, Int, [Float]) -> Void)?) -> [Float] {
        var x = start
        tap?("embed", -1, x)
        for il in 0..<cfg.nLayer {
            let L = model.layers[il]
            var r = x
            var h = GK.rmsnorm(x, L.attnNorm, cfg.eps)
            h = attention(h, L, il, pos: pos)
            tap?("attn", il, h)
            h = GK.rmsnorm(h, L.postAttnNorm, cfg.eps)
            for i in 0..<cfg.nEmbd { h[i] += r[i] }

            r = h
            var f = GK.rmsnorm(h, L.ffnNorm, cfg.eps)
            f = mlp(f, L)
            tap?("mlp", il, f)
            f = GK.rmsnorm(f, L.postFfnNorm, cfg.eps)
            for i in 0..<cfg.nEmbd { f[i] += r[i] }

            if cfg.hasPerLayerInputs {
                r = f
                var p = perLayer(f, L, il)
                p = GK.rmsnorm(p, L.perLayerPostNorm, cfg.eps)
                for i in 0..<cfg.nEmbd { p[i] += r[i] }
                f = p
            }

            // The layer scalar multiplies the WHOLE layer output, last.
            let s = L.layerScalar
            for i in 0..<cfg.nEmbd { f[i] *= s }
            x = f
            tap?("l_out", il, x)
        }
        let out = GK.rmsnorm(x, model.outputNorm, cfg.eps)
        tap?("final", -1, out)
        return out
    }

    // logits = lm_head @ hidden, then the final softcap: HF applies
    // tanh(logits / cap) * cap, which is why nothing in the reference dump
    // ever exceeds +/-30 (F18).
    public func logits(_ hidden: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: cfg.nVocab)
        linear(model.output, hidden, &out)
        let cap = cfg.logitSoftcap
        if cap > 0 {
            for i in 0..<out.count { out[i] = tanhf(out[i] / cap) * cap }
        }
        return out
    }
}

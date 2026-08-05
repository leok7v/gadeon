// The qwen35 forward loop: 64-layer GGGA stack (GDN + attention), token by token,
// threading per-layer conv/rec/KV state. Mirrors qwen35.cpp graph::graph().
import Foundation

public final class BonsaiEngine {
    let model: BonsaiModel
    let cfg: BonsaiConfig
    var gdn: [Int: GDNState] = [:]
    var kvc: [Int: KVCache] = [:]
    // Live decode position (absolute token index) + the optional sampler the
    // ChatSession installs; nil -> greedy argmax.
    public private(set) var pos = 0
    var sampler: Sampler?

    public init(_ model: BonsaiModel) {
        self.model = model
        cfg = model.cfg
        for il in 0..<cfg.nLayer {
            if cfg.isRecurrent(il) { gdn[il] = GDNState(cfg) } else { kvc[il] = KVCache(cfg) }
        }
    }

    // Fresh conversation: clear GDN conv/rec + attention KV + position.
    public func reset() {
        pos = 0
        for il in 0..<cfg.nLayer {
            if cfg.isRecurrent(il) { gdn[il] = GDNState(cfg) } else { kvc[il] = KVCache(cfg) }
        }
    }

    // Prefill `ids` onto the CURRENT state (no reset), returning the next-token
    // prediction after the last id. Token-by-token (the recurrence is exact).
    public func extend(_ ids: [Int32]) -> Int32 {
        var hidden = [Float]()
        for id in ids { hidden = forward(token: Int(id), pos: pos); pos += 1 }
        return pick(logits(hidden))
    }

    // Advance one token and return the next-token prediction.
    public func decode(_ token: Int32) -> Int32 {
        let hidden = forward(token: Int(token), pos: pos)
        pos += 1
        return pick(logits(hidden))
    }

    // Sample through the installed Sampler, else greedy argmax.
    func pick(_ logits: [Float]) -> Int32 {
        if sampler != nil {
            var work = logits
            let picked = sampler!.sample(&work)
            sampler!.accept(picked)
            return picked
        }
        return Int32(argmax(logits))
    }

    // A restorable snapshot of the whole generation state (GDN conv/rec + KV +
    // position), for the ChatSession mark/rewind + park/resume seam.
    // A restorable snapshot: the GDN recurrence copied whole (it cannot be
    // paged/truncated), the attention KV shared as append-only page snapshots
    // (cheap -- completed pages are shared, only a partial tail copies on the
    // next append). This is the mark/rewind + park/resume seam.
    // A parked conversation's bytes: the GDN recurrence whole (it cannot be
    // paged) and the KV pages. Format shared with the other engines through
    // StateBytes. [[never-reprefill]]

    public func serialize(_ b: Bookmark) -> Data {
        var out = Data()
        StateBytes.putHeader(&out)
        StateBytes.putInt(&out, b.pos)
        StateBytes.putInt(&out, b.gdn.count)
        for il in b.gdn.keys.sorted() {
            StateBytes.putInt(&out, il)
            StateBytes.putFloats(&out, b.gdn[il]!.conv)
            StateBytes.putFloats(&out, b.gdn[il]!.rec)
        }
        StateBytes.putInt(&out, b.kv.count)
        for il in b.kv.keys.sorted() {
            let s = b.kv[il]!
            StateBytes.putInt(&out, il)
            StateBytes.putInt(&out, s.len)
            StateBytes.putInt(&out, s.kPages.count)
            for page in s.kPages { StateBytes.putFloats(&out, page) }
            for page in s.vPages { StateBytes.putFloats(&out, page) }
        }
        return out
    }

    public func deserialize(_ data: Data) -> Bookmark? {
        let b = [UInt8](data)
        var p = 0
        guard StateBytes.readHeader(b, &p) else { return nil }
        let pos = StateBytes.getInt(b, &p)
        var gdn: [Int: (conv: [Float], rec: [Float])] = [:]
        for _ in 0..<StateBytes.getInt(b, &p) {
            let il = StateBytes.getInt(b, &p)
            let conv = StateBytes.getFloats(b, &p)
            gdn[il] = (conv: conv, rec: StateBytes.getFloats(b, &p))
        }
        var kv: [Int: KVCache.Snapshot] = [:]
        for _ in 0..<StateBytes.getInt(b, &p) {
            let il = StateBytes.getInt(b, &p)
            let len = StateBytes.getInt(b, &p)
            let n = StateBytes.getInt(b, &p)
            var kp: [[Float]] = [], vp: [[Float]] = []
            for _ in 0..<n { kp.append(StateBytes.getFloats(b, &p)) }
            for _ in 0..<n { vp.append(StateBytes.getFloats(b, &p)) }
            kv[il] = KVCache.Snapshot(kPages: kp, vPages: vp, len: len)
        }
        return Bookmark(pos: pos, gdn: gdn, kv: kv)
    }

    public struct Bookmark: @unchecked Sendable {
        let pos: Int
        let gdn: [Int: (conv: [Float], rec: [Float])]
        let kv: [Int: KVCache.Snapshot]
    }

    public func bookmark() -> Bookmark {
        var g: [Int: (conv: [Float], rec: [Float])] = [:]
        for (il, s) in gdn { g[il] = (s.conv, s.rec) }
        var k: [Int: KVCache.Snapshot] = [:]
        for (il, c) in kvc { k[il] = c.snapshot() }
        return Bookmark(pos: pos, gdn: g, kv: k)
    }

    public func restore(_ b: Bookmark) {
        pos = b.pos
        for (il, s) in b.gdn { gdn[il]!.conv = s.conv; gdn[il]!.rec = s.rec }
        for (il, s) in b.kv { kvc[il]!.restore(s) }
    }

    // embedding row for a token id (tok_embd is Q2_0, ne0 = nEmbd)
    func embed(_ token: Int) -> [Float] {
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        let rowBytes = cfg.nEmbd / Q2_0.qk * Q2_0.blockBytes
        Q2_0.dequant(model.tokEmbd.base + token * rowBytes, count: cfg.nEmbd, into: &out)
        return out
    }

    func ffn(_ x: [Float], _ L: BonsaiLayer) -> [Float] {
        var gate = [Float](repeating: 0, count: cfg.nFF)
        var up = [Float](repeating: 0, count: cfg.nFF)
        Q2_0.matvec(L.ffnGate, x: x, out: &gate)
        Q2_0.matvec(L.ffnUp, x: x, out: &up)
        for i in 0..<cfg.nFF { gate[i] = silu(gate[i]) * up[i] }
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        Q2_0.matvec(L.ffnDown, x: gate, out: &out)
        return out
    }

    // Run one token at absolute position `pos`. Returns the post-final-norm hidden.
    // Logits are computed separately (only for the tokens we sample).
    @discardableResult
    public func forward(token: Int, pos: Int, tap: ((String, Int, [Float]) -> Void)? = nil) -> [Float] {
        var x = embed(token)
        tap?("embed", -1, x)
        for il in 0..<cfg.nLayer {
            let L = model.layers[il]
            let normed = Kern.rmsnorm(x, F32T.ptr(L.attnNorm), cfg.eps)
            tap?("attn_norm", il, normed)
            let attn: [Float]
            if L.recurrent {
                attn = GDN.step(normed, L, gdn[il]!, cfg)
            } else {
                attn = Attn.step(normed, L, kvc[il]!, pos: pos, cfg)
            }
            tap?("attn_out", il, attn)
            for i in 0..<cfg.nEmbd { x[i] += attn[i] }
            let ffnRes = x
            let postNormed = Kern.rmsnorm(x, F32T.ptr(L.attnPostNorm), cfg.eps)
            let f = ffn(postNormed, L)
            x = ffnRes
            for i in 0..<cfg.nEmbd { x[i] += f[i] }
            tap?("l_out", il, x)
        }
        return Kern.rmsnorm(x, F32T.ptr(model.outputNorm), cfg.eps)
    }

    // logits = output @ hidden  (tied Q2_0 lm_head), returns [nVocab]
    public func logits(_ hidden: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: cfg.nVocab)
        Q2_0.matvec(model.output, x: hidden, out: &out)
        return out
    }

    public func argmax(_ v: [Float]) -> Int {
        var bi = 0; var bv = -Float.greatestFiniteMagnitude
        for i in 0..<v.count where v[i] > bv { bv = v[i]; bi = i }
        return bi
    }
}

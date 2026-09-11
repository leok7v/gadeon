// A 6-layer BERT sentence encoder (all-MiniLM-L6-v2), the embedder behind the
// Slugs semantic index. A pure-Swift port of the reference llm.c: parse the
// GGUF (reusing GGUF.swift), dequantize Q4_0/Q8_0/F16 weights to f32, tokenize
// with BERT WordPiece, run the 6 encoder blocks (full bidirectional attention,
// post-LN, GELU), mean-pool and L2-normalize to a 384-d unit vector. It matches
// llama.cpp / the C reference to cosine ~1.0 on clean English (the parity gate).
//
// This is an EMBEDDING model, not generative -- no KV cache, no causal mask, no
// sampling. It is a separate engine from the Qwen chat stack; it only shares
// GGUF.swift and the fixed-width-type idioms, not the decode engine.
import Foundation

// The BERT hyper-parameters, the small buffers (biases, norms, positional/type
// embeddings) dequantized to f32 at load, and the big weight matrices left
// QUANTIZED in the GGUF mmap and dequantized ROW-BY-ROW on the fly during the
// forward (into one reused scratch row). So a live encoder pins only the ~36 MB
// file-backed mmap (clean, evictable) instead of ~90 MB of dirty f32 -- there is
// nothing to free, mirroring SignIndex's read-straight-from-the-mmap design.
final class MiniLM {
    struct Layer {
        let wq: GGUFTensor, bq: [Float]
        let wk: GGUFTensor, bk: [Float]
        let wv: GGUFTensor, bv: [Float]
        let wo: GGUFTensor, bo: [Float]
        let attnNormW: [Float], attnNormB: [Float]
        let wUp: GGUFTensor, bUp: [Float]
        let wDown: GGUFTensor, bDown: [Float]
        let outNormW: [Float], outNormB: [Float]
    }

    let nLayer: Int
    let nEmbd: Int
    let nFF: Int
    let nHead: Int
    let nCtx: Int
    let lnEps: Float
    private let clsId: Int32
    private let sepId: Int32
    private let unkId: Int32

    // The GGUF is pinned so its mmap stays mapped while we read weight rows out
    // of it; the big matrices (token embedding + per-layer projections) live
    // there quantized and are expanded per row into `rowScratch`.
    private let gguf: GGUF
    private let tokEmb: GGUFTensor
    private let posEmb: [Float]
    private let typeEmb: [Float]
    private let embNormW: [Float]
    private let embNormB: [Float]
    private let layers: [Layer]
    private let vocab: [String: Int32]
    // One reused row buffer for on-the-fly dequant; sized to the widest matrix
    // row (nFF). The encoder runs single-threaded per embed (one WikiSlugs per
    // query), so a shared scratch is safe.
    private var rowScratch: [Float]

    var dim: Int { nEmbd }

    init(gguf g: GGUF) {
        nLayer = g.int("bert.block_count") ?? 6
        nEmbd = g.int("bert.embedding_length") ?? 384
        nFF = g.int("bert.feed_forward_length") ?? 1536
        nHead = g.int("bert.attention.head_count") ?? 12
        nCtx = g.int("bert.context_length") ?? 512
        lnEps = Float(g.double("bert.attention.layer_norm_epsilon") ?? 1e-12)
        clsId = Int32(g.int("tokenizer.ggml.cls_token_id") ?? 101)
        sepId = Int32(g.int("tokenizer.ggml.seperator_token_id") ?? 102)
        unkId = Int32(g.int("tokenizer.ggml.unknown_token_id") ?? 100)

        gguf = g
        tokEmb = g.tensor("token_embd.weight")
        posEmb = MiniLM.dequant(g.tensor("position_embd.weight"))
        typeEmb = MiniLM.dequant(g.tensor("token_types.weight"))
        embNormW = MiniLM.dequant(g.tensor("token_embd_norm.weight"))
        embNormB = MiniLM.dequant(g.tensor("token_embd_norm.bias"))
        rowScratch = [Float](repeating: 0, count: nFF)
        let n = nLayer
        // Big matrices kept as mmap handles (w); small vectors dequantized (f).
        layers = (0..<n).map { il in
            func w(_ s: String) -> GGUFTensor { g.tensor("blk.\(il).\(s)") }
            func f(_ s: String) -> [Float] {
                MiniLM.dequant(g.tensor("blk.\(il).\(s)"))
            }
            return Layer(
                wq: w("attn_q.weight"), bq: f("attn_q.bias"),
                wk: w("attn_k.weight"), bk: f("attn_k.bias"),
                wv: w("attn_v.weight"), bv: f("attn_v.bias"),
                wo: w("attn_output.weight"), bo: f("attn_output.bias"),
                attnNormW: f("attn_output_norm.weight"),
                attnNormB: f("attn_output_norm.bias"),
                wUp: w("ffn_up.weight"), bUp: f("ffn_up.bias"),
                wDown: w("ffn_down.weight"), bDown: f("ffn_down.bias"),
                outNormW: f("layer_output_norm.weight"),
                outNormB: f("layer_output_norm.bias"))
        }
        var v = [String: Int32](minimumCapacity: 30720)
        if let tokens = g.strings("tokenizer.ggml.tokens") {
            for (i, piece) in tokens.enumerated() { v[piece] = Int32(i) }
        }
        vocab = v
    }

    // ---- dequant (Q4_0 / Q8_0 / F16 / F32 -> f32) -----------------------
    // Expand `count` weights starting at element `start` of `t` into buf[0..<count]
    // (buf must be at least that long). The mmap'd tensor bytes are addressed raw
    // (unaligned loads), laid out exactly as llm.c's dequant (row-major, ne0
    // fastest). For block-quantized types `start` and `count` are multiples of 32
    // -- weight rows are a multiple of 32 wide, so a row maps to whole blocks.
    static func dequant(_ t: GGUFTensor, _ start: Int, _ count: Int,
                        into buf: inout [Float]) {
        let base = t.base
        switch t.type {
        case .f32:
            buf.withUnsafeMutableBytes {
                _ = memcpy($0.baseAddress!, base + start * 4, count * 4)
            }
        case .f16, .bf16:
            for i in 0..<count {
                let h = base.loadUnaligned(
                    fromByteOffset: (start + i) * 2, as: UInt16.self)
                buf[i] = Float(Float16(bitPattern: h))
            }
        case .q4_0:
            let b0 = start / 32
            for b in 0..<(count / 32) {
                let p = base + (b0 + b) * 18
                let d = Float(Float16(bitPattern:
                    p.loadUnaligned(as: UInt16.self)))
                for j in 0..<16 {
                    let byte = p.load(fromByteOffset: 2 + j, as: UInt8.self)
                    let lo = Int(byte & 0x0f) - 8
                    let hi = Int(byte >> 4) - 8
                    buf[b * 32 + j] = Float(lo) * d
                    buf[b * 32 + j + 16] = Float(hi) * d
                }
            }
        case .q8_0:
            let b0 = start / 32
            for b in 0..<(count / 32) {
                let p = base + (b0 + b) * 34
                let d = Float(Float16(bitPattern:
                    p.loadUnaligned(as: UInt16.self)))
                for j in 0..<32 {
                    let q = p.load(fromByteOffset: 2 + j, as: Int8.self)
                    buf[b * 32 + j] = Float(q) * d
                }
            }
        default:
            preconditionFailure("MiniLM: unsupported ggml type \(t.type)")
        }
    }

    // Whole-tensor dequant into a fresh buffer, for the small always-resident
    // vectors (biases, norms, positional/type embeddings).
    static func dequant(_ t: GGUFTensor) -> [Float] {
        var out = [Float](repeating: 0, count: t.count)
        dequant(t, 0, t.count, into: &out)
        return out
    }

    // ---- WordPiece tokenizer (BERT WPM) ---------------------------------
    private func isSpace(_ c: UInt8) -> Bool {
        c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D || c == 0x0C
            || c == 0x0B
    }
    private func isPunct(_ c: UInt8) -> Bool {
        (c >= 33 && c <= 47) || (c >= 58 && c <= 64)
            || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)
    }
    private func lowerAscii(_ c: UInt8) -> UInt8 {
        (c >= 0x41 && c <= 0x5A) ? c + 32 : c
    }

    // BERT's basic tokenizer NFD-decomposes then drops combining marks, so
    // accented Latin folds to its ASCII base ("Zurich" from "Zürich", "maric"
    // from "Marić"). These tables fold the precomposed Latin-1 Supplement
    // (0x00C0..0x00FF) and Latin Extended-A (0x0100..0x017F) letters to a
    // lowercase ASCII base; '.' marks the non-decomposable ones (ss, o-slash,
    // l-stroke, ligatures), left as-is (pass-through, so they become [UNK]).
    private static func foldTable(_ s: String) -> [UInt8] {
        s.utf8.map { $0 == UInt8(ascii: ".") ? 0 : $0 }
    }
    private static let latin1Fold = MiniLM.foldTable(
        "aaaaaa.ceeeeiiii" + ".nooooo..uuuuy.." +
        "aaaaaa.ceeeeiiii" + ".nooooo..uuuuy.y")
    private static let latinAFold = MiniLM.foldTable(
        "aaaaaaccccccccdd" + "..eeeeeeeeeegggg" +
        "gggghh..iiiiiiii" + "ii..jjkk.lllllll" +
        "l..nnnnnn...oooo" + "oo..rrrrrrssssss" +
        "sstttt..uuuuuuuu" + "uuuuwwyyyzzzzzzs")

    // Decode one UTF-8 codepoint at bytes[i] (len bytes available); the returned
    // adv is its byte length. Well-formed UTF-8 (Wikipedia text) is assumed; a
    // truncated lead is returned as its raw byte.
    private func utf8Decode(_ bytes: [UInt8], _ i: Int, _ len: Int)
        -> (cp: UInt32, adv: Int) {
        let c = bytes[i]
        var cp: UInt32
        var n: Int
        if c < 0x80 { cp = UInt32(c); n = 1 }
        else if c & 0xE0 == 0xC0 { cp = UInt32(c & 0x1F); n = 2 }
        else if c & 0xF0 == 0xE0 { cp = UInt32(c & 0x0F); n = 3 }
        else if c & 0xF8 == 0xF0 { cp = UInt32(c & 0x07); n = 4 }
        else { cp = UInt32(c); n = 1 }
        if i + n > len { cp = UInt32(c); n = 1 }
        var k = 1
        while k < n {
            cp = (cp << 6) | UInt32(bytes[i + k] & 0x3F)
            k += 1
        }
        return (cp, n)
    }

    // How a codepoint tokenizes, matching BERT's basic tokenizer: chars fold
    // into the current word, punct/space end it (punct is also its own token),
    // combining marks drop, and non-Latin (CJK, Greek) passes through as raw
    // bytes -> [UNK], the previous behaviour.
    private enum CpKind { case chars, pass, punct, space, drop }

    // Classify `cp` and, for chars/punct, the single folded ASCII byte. ASCII
    // keeps its space/punct/lower handling; Latin accents fold; common Unicode
    // dashes/quotes count as punctuation (word boundaries).
    private func foldCp(_ cp: UInt32) -> (kind: CpKind, folded: UInt8) {
        var kind: CpKind = .pass
        var folded: UInt8 = 0
        if cp < 0x80 {
            let c = UInt8(cp)
            if isSpace(c) { kind = .space }
            else if isPunct(c) { kind = .punct; folded = c }
            else { kind = .chars; folded = lowerAscii(c) }
        } else if cp >= 0x0300 && cp <= 0x036F {
            kind = .drop                                  // combining marks
        } else if cp == 0x00A0 {
            kind = .space                                 // no-break space
        } else if cp >= 0x00C0 && cp <= 0x00FF
            && MiniLM.latin1Fold[Int(cp - 0x00C0)] != 0 {
            kind = .chars; folded = MiniLM.latin1Fold[Int(cp - 0x00C0)]
        } else if cp >= 0x0100 && cp <= 0x017F
            && MiniLM.latinAFold[Int(cp - 0x0100)] != 0 {
            kind = .chars; folded = MiniLM.latinAFold[Int(cp - 0x0100)]
        } else if cp >= 0x2010 && cp <= 0x2015 {
            kind = .punct; folded = 0x2D                  // hyphens / dashes
        } else if cp == 0x2018 || cp == 0x2019 || cp == 0x02BB {
            kind = .punct; folded = 0x27                  // curly / okina '
        } else if cp == 0x201C || cp == 0x201D {
            kind = .punct; folded = 0x22                  // curly double "
        } else {
            kind = .pass                                  // CJK, Greek, ...
        }
        return (kind, folded)
    }

    // Greedy longest-match WordPiece over the word prefixed by U+2581 (the
    // llama.cpp phantom space): word-initial pieces carry the prefix, so the
    // whole word is matched front-to-back. nil when any span is out-of-vocab
    // (the caller then emits one [UNK]).
    private func wordpiece(_ w: [UInt8]) -> [Int32]? {
        var word1: [UInt8] = [0xE2, 0x96, 0x81]
        word1.append(contentsOf: w)
        let n = word1.count
        var pieces: [Int32] = []
        var i = 0
        var alive = true
        while i < n && alive {
            var j = n
            var found: Int32? = nil
            while j > i && found == nil {
                let key = String(decoding: word1[i..<j], as: UTF8.self)
                found = vocab[key]
                if found == nil { j -= 1 }
            }
            if let found {
                pieces.append(found)
                i = j
            } else {
                alive = false
            }
        }
        return alive ? pieces : nil
    }

    private func addWord(_ w: [UInt8], _ ids: inout [Int32]) {
        if let pieces = wordpiece(w) {
            ids.append(contentsOf: pieces)
        } else {
            ids.append(unkId)
        }
    }

    // [CLS] word-pieces... [SEP], truncated to nCtx. Reads UTF-8 codepoints,
    // folding accents to ASCII and lowercasing (foldCp); a word accumulates
    // folded chars and flushes at each space/punct boundary, where a punct
    // codepoint is also emitted as its own single-char token.
    func tokenize(_ text: String) -> [Int32] {
        var ids: [Int32] = [clsId]
        let bytes = Array(text.utf8)
        let n = bytes.count
        var word: [UInt8] = []
        var i = 0
        while i < n && ids.count < nCtx - 1 {
            let (cp, adv) = utf8Decode(bytes, i, n)
            let (kind, folded) = foldCp(cp)
            switch kind {
            case .chars:
                word.append(folded)
            case .pass:
                word.append(contentsOf: bytes[i ..< i + adv])
            case .space, .punct:
                if !word.isEmpty {
                    addWord(word, &ids)
                    word.removeAll(keepingCapacity: true)
                }
                if kind == .punct { addWord([folded], &ids) }
            case .drop:
                break
            }
            i += adv
        }
        if !word.isEmpty && ids.count < nCtx - 1 { addWord(word, &ids) }
        ids.append(sepId)
        if ids.count > nCtx { ids.removeLast(ids.count - nCtx) }
        return ids
    }

    // ---- math primitives ------------------------------------------------
    // y[o] = W[o][*] . x + b[o]  (W row-major [out][in], ne0=in fastest). W's
    // row o is dequantized from the mmap into rowScratch just before its dot
    // product; the dot loop stays a plain contiguous UnsafeBufferPointer run so
    // the optimizer can vectorize it.
    private func linear(_ w: GGUFTensor, _ b: [Float], _ x: [Float],
                        xoff: Int, inn: Int, out: Int,
                        _ y: inout [Float], yoff: Int) {
        x.withUnsafeBufferPointer { xp in
            let xb = xp.baseAddress! + xoff
            for o in 0..<out {
                MiniLM.dequant(w, o * inn, inn, into: &rowScratch)
                var acc: Float = 0
                rowScratch.withUnsafeBufferPointer { rp in
                    let row = rp.baseAddress!
                    for i in 0..<inn { acc += row[i] * xb[i] }
                }
                y[yoff + o] = acc + b[o]
            }
        }
    }

    // LayerNorm over one n-vector at x[off..]: (x-mean)/sqrt(var+eps)*w + b.
    private func layerNorm(_ x: inout [Float], off: Int, n: Int,
                           _ w: [Float], _ b: [Float]) {
        var mean: Float = 0
        for i in 0..<n { mean += x[off + i] }
        mean /= Float(n)
        var varr: Float = 0
        for i in 0..<n {
            let d = x[off + i] - mean
            varr += d * d
        }
        varr /= Float(n)
        let inv = 1 / (varr + lnEps).squareRoot()
        for i in 0..<n {
            x[off + i] = (x[off + i] - mean) * inv * w[i] + b[i]
        }
    }

    private func gelu(_ x: inout [Float], n: Int) {
        let k: Float = 0.7978845608028654   // sqrt(2/pi)
        for i in 0..<n {
            let v = x[i]
            x[i] = 0.5 * v * (1 + tanhf(k * (v + 0.044715 * v * v * v)))
        }
    }

    private func softmax(_ s: inout [Float], n: Int) {
        var mx = s[0]
        for i in 1..<n where s[i] > mx { mx = s[i] }
        var sum: Float = 0
        for i in 0..<n {
            s[i] = expf(s[i] - mx)
            sum += s[i]
        }
        let inv = 1 / sum
        for i in 0..<n { s[i] *= inv }
    }

    // ---- forward pass ---------------------------------------------------
    // Full bidirectional self-attention: q/k/v projections for all T tokens,
    // then per head per query a softmax over ALL keys and a value-weighted
    // context. Post-LN, so the residual add precedes the LayerNorm.
    private func encoderLayer(_ L: Layer, _ x: inout [Float], T: Int) {
        let ne = nEmbd, hd = nEmbd / nHead
        let scale = 1 / Float(hd).squareRoot()
        var q = [Float](repeating: 0, count: T * ne)
        var k = [Float](repeating: 0, count: T * ne)
        var vv = [Float](repeating: 0, count: T * ne)
        var ctx = [Float](repeating: 0, count: T * ne)
        for t in 0..<T {
            linear(L.wq, L.bq, x, xoff: t * ne, inn: ne, out: ne, &q, yoff: t * ne)
            linear(L.wk, L.bk, x, xoff: t * ne, inn: ne, out: ne, &k, yoff: t * ne)
            linear(L.wv, L.bv, x, xoff: t * ne, inn: ne, out: ne, &vv, yoff: t * ne)
        }
        var scores = [Float](repeating: 0, count: T)
        for h in 0..<nHead {
            let off = h * hd
            for i in 0..<T {
                q.withUnsafeBufferPointer { qp in
                    k.withUnsafeBufferPointer { kp in
                        let qi = qp.baseAddress! + i * ne + off
                        for j in 0..<T {
                            let kj = kp.baseAddress! + j * ne + off
                            var acc: Float = 0
                            for d in 0..<hd { acc += qi[d] * kj[d] }
                            scores[j] = acc * scale
                        }
                    }
                }
                softmax(&scores, n: T)
                for d in 0..<hd { ctx[i * ne + off + d] = 0 }
                for j in 0..<T {
                    let wj = scores[j]
                    for d in 0..<hd {
                        ctx[i * ne + off + d] += wj * vv[j * ne + off + d]
                    }
                }
            }
        }
        var tmp = [Float](repeating: 0, count: ne)
        var ff = [Float](repeating: 0, count: nFF)
        for t in 0..<T {
            linear(L.wo, L.bo, ctx, xoff: t * ne, inn: ne, out: ne, &tmp, yoff: 0)
            for d in 0..<ne { x[t * ne + d] += tmp[d] }
            layerNorm(&x, off: t * ne, n: ne, L.attnNormW, L.attnNormB)
            linear(L.wUp, L.bUp, x, xoff: t * ne, inn: ne, out: nFF, &ff, yoff: 0)
            gelu(&ff, n: nFF)
            linear(L.wDown, L.bDown, ff, xoff: 0, inn: nFF, out: ne, &tmp, yoff: 0)
            for d in 0..<ne { x[t * ne + d] += tmp[d] }
            layerNorm(&x, off: t * ne, n: ne, L.outNormW, L.outNormB)
        }
    }

    // token+position+type embedding + LayerNorm; type is always 0 (one segment).
    // Each token's embedding row is dequantized from the mmap on demand (row
    // width ne is a multiple of 32, so id*ne lands on a block boundary).
    private func embedTokens(_ ids: [Int32], _ x: inout [Float]) {
        let ne = nEmbd
        for t in 0..<ids.count {
            MiniLM.dequant(tokEmb, Int(ids[t]) * ne, ne, into: &rowScratch)
            let pe = t * ne
            for d in 0..<ne {
                x[t * ne + d] = rowScratch[d] + posEmb[pe + d] + typeEmb[d]
            }
            layerNorm(&x, off: t * ne, n: ne, embNormW, embNormB)
        }
    }

    // Encode text into a unit-normalized nEmbd embedding; returns the token
    // count (0 -> the out vector is untouched).
    @discardableResult
    func embed(_ text: String, into out: inout [Float]) -> Int {
        let ids = tokenize(text)
        let T = ids.count
        if T > 0 {
            let ne = nEmbd
            var x = [Float](repeating: 0, count: T * ne)
            embedTokens(ids, &x)
            for l in 0..<nLayer { encoderLayer(layers[l], &x, T: T) }
            for d in 0..<ne { out[d] = 0 }
            for t in 0..<T {
                for d in 0..<ne { out[d] += x[t * ne + d] }
            }
            let invT = 1 / Float(T)
            for d in 0..<ne { out[d] *= invT }
            var ss: Float = 0
            for d in 0..<ne { ss += out[d] * out[d] }
            let inv = ss > 0 ? 1 / ss.squareRoot() : 0
            for d in 0..<ne { out[d] *= inv }
        }
        return T
    }

    func embed(_ text: String) -> [Float] {
        var out = [Float](repeating: 0, count: nEmbd)
        embed(text, into: &out)
        return out
    }
}

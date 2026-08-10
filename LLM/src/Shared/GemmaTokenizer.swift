import Foundation

// The SentencePiece-flavoured tokenizer gemma ships, driven ENTIRELY by the
// GGUF's own metadata rather than by a model name. `Tokenizer.swift` is
// GPT-2 byte-level in both of its inits -- a different code path, not a
// parameter -- so this is a second implementation, not a branch (F24).
//
// What the file carries, and what each piece is for:
//   tokenizer.ggml.metaspace       " " was rewritten to this (U+2581)
//   tokenizer.ggml.byte_fallback   unknown characters decompose to <0xNN>
//   tokenizer.ggml.ignore_merges   a piece already in vocab skips the merges
//   tokenizer.ggml.token_type      1 NORMAL, 3 CONTROL, 6 BYTE
//   tokenizer.ggml.eos_token_ids   gemma stops on THREE ids, not one (F23)
public struct GemmaTokenizer: Sendable {
    private let vocab: [String: Int32]
    private let pieces: [String]
    private let types: [Int]
    private let ranks: [String: Int]
    private let byteId: [Int32]          // 0..255 -> the <0xNN> token id
    private let metaspace: String
    private let byteFallback: Bool
    private let ignoreMerges: Bool
    private let specials: [(text: String, id: Int32)]

    public let eosId: Int32
    // The authoritative stop set. A runtime that compares against the scalar
    // alone runs straight past <end_of_turn> into the next turn.
    public let eosIds: Set<Int32>
    public let bosId: Int32
    public let padId: Int32
    public let unkId: Int32

    // Internal like Tokenizer's own gguf init: GGUF is a module-internal
    // type, so the public door is GemmaChat.
    init(gguf g: GGUF) throws {
        let listed = g.strings("tokenizer.ggml.tokens")
        if listed == nil {
            throw GGUFErr.parse("tokenizer.ggml.tokens missing")
        }
        let toks = listed!
        pieces = toks
        types = g.ints("tokenizer.ggml.token_type") ?? []
        metaspace = g.string("tokenizer.ggml.metaspace") ?? "\u{2581}"
        byteFallback = (g.int("tokenizer.ggml.byte_fallback") ?? 0) != 0
        ignoreMerges = (g.int("tokenizer.ggml.ignore_merges") ?? 0) != 0

        var v = [String: Int32](minimumCapacity: toks.count)
        for (i, p) in toks.enumerated() { v[p] = Int32(i) }
        vocab = v

        var r = [String: Int](minimumCapacity: 520_000)
        for (i, m) in (g.strings("tokenizer.ggml.merges") ?? []).enumerated() {
            r[m] = i                       // already space-joined "a b"
        }
        ranks = r

        // The 256 <0xNN> tokens, located by their declared BYTE type rather
        // than by scanning for a spelling.
        var bytes = [Int32](repeating: -1, count: 256)
        for (i, p) in toks.enumerated()
        where i < types.count && types[i] == 6 {
            let hex = p.dropFirst(3).dropLast()
            if let b = UInt8(hex, radix: 16) { bytes[Int(b)] = Int32(i) }
        }
        byteId = bytes

        var sp: [(String, Int32)] = []
        for (i, p) in toks.enumerated()
        where i < types.count && types[i] == 3 {
            sp.append((p, Int32(i)))
        }
        sp.sort { a, b in a.0.count > b.0.count }
        specials = sp

        let ids = g.ints("tokenizer.ggml.eos_token_ids")?
            .map { id in Int32(id) }
        let scalar = Int32(g.int("tokenizer.ggml.eos_token_id") ?? 1)
        eosId = scalar
        eosIds = Set(ids ?? [scalar])
        bosId = Int32(g.int("tokenizer.ggml.bos_token_id") ?? 2)
        padId = Int32(g.int("tokenizer.ggml.padding_token_id") ?? 0)
        unkId = Int32(g.int("tokenizer.ggml.unknown_token_id") ?? 3)
    }

    public var vocabCount: Int { pieces.count }

    // The spelling, not the id: the chat template opens the conversation by
    // naming `bos_token` and needs bytes to render.
    public var bosToken: String {
        let i = Int(bosId)
        return i >= 0 && i < pieces.count ? pieces[i] : ""
    }

    // ---- encode ---------------------------------------------------------

    // Control tokens are matched first and atomically (the chat wire is made
    // of them); everything between them goes through the SentencePiece path.
    public func encode(_ text: String, addSpecial: Bool = true) -> [Int32] {
        var out: [Int32] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            let hit = addSpecial ? firstSpecial(in: rest) : nil
            if let hit {
                let head = rest[rest.startIndex ..< hit.range.lowerBound]
                out.append(contentsOf: encodePlain(String(head)))
                out.append(hit.id)
                rest = rest[hit.range.upperBound...]
            } else {
                out.append(contentsOf: encodePlain(String(rest)))
                rest = rest[rest.endIndex...]
            }
        }
        return out
    }

    private func firstSpecial(
        in s: Substring
    ) -> (range: Range<Substring.Index>, id: Int32)? {
        var best: (Range<Substring.Index>, Int32)? = nil
        for (tok, id) in specials {
            if let rng = s.range(of: tok),
               best == nil || rng.lowerBound < best!.0.lowerBound {
                best = (rng, id)
            }
        }
        return best.map { hit in (range: hit.0, id: hit.1) }
    }

    private func encodePlain(_ text: String) -> [Int32] {
        var out: [Int32] = []
        if !text.isEmpty {
            // The normalizer's whole job: " " becomes the metaspace mark, so
            // word boundaries live inside the pieces.
            let normalized = text.replacingOccurrences(of: " ",
                                                       with: metaspace)
            if ignoreMerges, let id = vocab[normalized] {
                out = [id]
            } else {
                for sym in bpe(normalized) {
                    if let id = vocab[sym] {
                        out.append(id)
                    } else {
                        out.append(contentsOf: fallback(sym))
                    }
                }
            }
        }
        return out
    }

    // A piece with no vocab entry decomposes to its raw UTF-8 bytes; without
    // byte fallback there is nothing left but <unk>.
    private func fallback(_ sym: String) -> [Int32] {
        var out: [Int32] = []
        if byteFallback {
            for b in sym.utf8 {
                let id = byteId[Int(b)]
                out.append(id >= 0 ? id : unkId)
            }
        } else {
            out.append(unkId)
        }
        return out
    }

    // Merge-rank BPE over characters, lowest rank first -- the same loop
    // Tokenizer.bpe runs, over metaspace-normalized text instead of the
    // gpt2 byte alphabet.
    private func bpe(_ token: String) -> [String] {
        var word = token.map { ch in String(ch) }
        var merging = word.count >= 2
        while merging {
            var minRank = Int.max
            var minPair = ""
            for i in 0 ..< (word.count - 1) {
                let pair = word[i] + " " + word[i + 1]
                if let rank = ranks[pair], rank < minRank {
                    minRank = rank
                    minPair = pair
                }
            }
            if minRank == Int.max {
                merging = false
            } else {
                let parts = minPair.split(separator: " ", maxSplits: 1)
                let a = String(parts[0]), b = String(parts[1])
                var merged: [String] = []
                var i = 0
                while i < word.count {
                    if i < word.count - 1 && word[i] == a && word[i + 1] == b {
                        merged.append(a + b)
                        i += 2
                    } else {
                        merged.append(word[i])
                        i += 1
                    }
                }
                word = merged
                merging = word.count >= 2
            }
        }
        return word
    }

    // ---- decode ---------------------------------------------------------

    // Byte tokens contribute their raw byte, control tokens their literal
    // spelling, everything else its piece with the metaspace mark turned back
    // into a space. The converter's gate proved all three are needed: the
    // emoji probe fails without byte fallback, the leading-space probe
    // without the metaspace rewrite.
    public func decodeBytes(_ ids: [Int32]) -> [UInt8] {
        var out: [UInt8] = []
        for id in ids {
            let i = Int(id)
            if i >= 0 && i < pieces.count {
                let t = i < types.count ? types[i] : 1
                if t == 6 {
                    let hex = pieces[i].dropFirst(3).dropLast()
                    if let b = UInt8(hex, radix: 16) { out.append(b) }
                } else if t == 3 {
                    out.append(contentsOf: pieces[i].utf8)
                } else {
                    out.append(contentsOf: pieces[i]
                        .replacingOccurrences(of: metaspace, with: " ").utf8)
                }
            }
        }
        return out
    }

    public func decode(_ ids: [Int32]) -> String {
        String(decoding: decodeBytes(ids), as: UTF8.self)
    }

    public func tokenBytes(_ id: Int32) -> [UInt8] { decodeBytes([id]) }
}

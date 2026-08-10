import Foundation

// GPT-2 byte-level BPE reconstructed from tokenizer.json. Swift Regex covers
// the pretokenizer pattern, so there is no hand-written Unicode scanner.
public struct Tokenizer: Sendable {
    // Regex matching is thread-safe; Regex is just not marked Sendable yet.
    private struct RegexBox: @unchecked Sendable {
        let regex: Regex<AnyRegexOutput>
    }

    private let vocab: [String: Int32]
    private let idToBytes: [Int32: [UInt8]]
    private let ranks: [String: Int]
    private let byteToUni: [UInt8: Character]
    private let specials: [(text: String, id: Int32)]
    private let splitRegex: RegexBox
    public let eosId: Int32

    public init(modelsDir: URL) throws {
        let url = modelsDir.appendingPathComponent("tokenizer.json")
        let raw = try Data(contentsOf: url)
        let root = try JSONSerialization.jsonObject(with: raw)
            as! [String: Any]
        let model = root["model"] as! [String: Any]

        var v = [String: Int32](minimumCapacity: 250_000)
        for (piece, id) in model["vocab"] as! [String: Int] {
            v[piece] = Int32(id)
        }
        var r = [String: Int](minimumCapacity: 250_000)
        // merges is either one space-joined string per pair (older export) or
        // a 2-element [a, b] array (transformers 5.x); normalize to the
        // space-joined key bpe() reads.
        for (i, m) in (model["merges"] as! [Any]).enumerated() {
            var key = m as? String
            if key == nil, let pair = m as? [String], pair.count == 2 {
                key = pair[0] + " " + pair[1]
            }
            if let key { r[key] = i }
        }

        let b2u = Tokenizer.bytesToUnicode()
        var u2b = [Character: UInt8](minimumCapacity: 256)
        for (b, c) in b2u { u2b[c] = b }

        var special: [(String, Int32)] = []
        for t in root["added_tokens"] as! [[String: Any]] {
            let content = t["content"] as! String
            let id = Int32(t["id"] as! Int)
            v[content] = id
            special.append((content, id))
        }
        special.sort { $0.0.count > $1.0.count }

        let specialSet = Set(special.map { $0.0 })
        var i2b = [Int32: [UInt8]](minimumCapacity: 250_000)
        for (piece, id) in v {
            if specialSet.contains(piece) {
                i2b[id] = Array(piece.utf8)
            } else {
                var bytes: [UInt8] = []
                bytes.reserveCapacity(piece.count)
                for ch in piece where u2b[ch] != nil {
                    bytes.append(u2b[ch]!)
                }
                i2b[id] = bytes
            }
        }

        let pre = root["pre_tokenizer"] as! [String: Any]
        let seq = pre["pretokenizers"] as! [[String: Any]]
        let patt = (seq[0]["pattern"] as! [String: Any])["Regex"]
            as! String

        self.vocab = v
        self.idToBytes = i2b
        self.ranks = r
        self.byteToUni = b2u
        self.specials = special
        // Native Regex covers our pretokenizers (\p{L}/\p{N}, lookahead, \b)
        // with none of C++ std::regex's Unicode-class gaps. Open risk:
        // semantic parity with Python `regex` (which trained the vocab). If a
        // future model's split diverges (a token-id diff), port unicode.c's
        // byte-level engine rather than bending Regex to match.
        self.splitRegex = RegexBox(regex: try Regex(patt))
        self.eosId = v["<|im_end|>"] ?? 0
    }

    // The Qwen (gpt2 byte-level) pretokenizer split pattern. The GGUF stores only
    // the pre-tokenizer NAME (tokenizer.ggml.pre = "qwen35"/"qwen2"), not the
    // regex, so it is a constant here -- verified byte-identical to the pattern
    // the Qwen3.5 tokenizer.json carries (the CoreML sets and Bonsai share one
    // tokenizer). Swift Regex covers it (\p{L}/\p{N}/\p{M}, lookahead).
    static let qwenPretokenizer =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    // Build the tokenizer from a GGUF's embedded metadata -- the single
    // self-contained source (tokenizer.ggml.tokens / .merges / .token_type /
    // .eos_token_id), no tokenizer.json. token_type CONTROL(3)/USER_DEFINED(4)
    // are the special tokens (their piece string IS their bytes); NORMAL(1) are
    // byte-level-BPE pieces (mapped through the gpt2 byte<->unicode table).
    init(gguf g: GGUF) throws {
        let listed = g.strings("tokenizer.ggml.tokens")
        if listed == nil {
            throw GGUFErr.parse("tokenizer.ggml.tokens missing")
        }
        let tokens = listed!
        let merges = g.strings("tokenizer.ggml.merges") ?? []
        let types = g.ints("tokenizer.ggml.token_type") ?? []

        var v = [String: Int32](minimumCapacity: tokens.count)
        for (i, piece) in tokens.enumerated() { v[piece] = Int32(i) }
        var r = [String: Int](minimumCapacity: merges.count)
        for (i, m) in merges.enumerated() { r[m] = i }   // already space-joined

        let b2u = Tokenizer.bytesToUnicode()
        var u2b = [Character: UInt8](minimumCapacity: 256)
        for (b, c) in b2u { u2b[c] = b }

        // specials = CONTROL / USER_DEFINED tokens; their bytes are literal.
        var special: [(String, Int32)] = []
        var specialSet = Set<String>()
        for (i, piece) in tokens.enumerated() {
            let t = i < types.count ? types[i] : 1
            if t == 3 || t == 4 { special.append((piece, Int32(i))); specialSet.insert(piece) }
        }
        special.sort { $0.0.count > $1.0.count }

        var i2b = [Int32: [UInt8]](minimumCapacity: tokens.count)
        for (piece, id) in v {
            if specialSet.contains(piece) {
                i2b[id] = Array(piece.utf8)
            } else {
                var bytes: [UInt8] = []
                bytes.reserveCapacity(piece.count)
                for ch in piece where u2b[ch] != nil { bytes.append(u2b[ch]!) }
                i2b[id] = bytes
            }
        }

        self.vocab = v
        self.idToBytes = i2b
        self.ranks = r
        self.byteToUni = b2u
        self.specials = special
        self.splitRegex = RegexBox(regex: try Regex(Tokenizer.qwenPretokenizer))
        self.eosId = Int32(g.int("tokenizer.ggml.eos_token_id") ?? 0)
    }

    static func bytesToUnicode() -> [UInt8: Character] {
        var bs = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var map = [UInt8: Character](minimumCapacity: 256)
        for (b, c) in zip(bs, cs) {
            map[UInt8(b)] = Character(UnicodeScalar(c)!)
        }
        return map
    }

    public func encode(_ text: String,
                       addSpecial: Bool = true) -> [Int32] {
        var out: [Int32] = []
        if addSpecial {
            var rest = Substring(text)
            while !rest.isEmpty {
                if let hit = firstSpecial(in: rest) {
                    let head = rest[rest.startIndex ..< hit.range.lowerBound]
                    out.append(contentsOf: encodePlain(String(head)))
                    out.append(hit.id)
                    rest = rest[hit.range.upperBound...]
                } else {
                    out.append(contentsOf: encodePlain(String(rest)))
                    rest = rest[rest.endIndex...]
                }
            }
        } else {
            out = encodePlain(text)
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
        return best.map { (range: $0.0, id: $0.1) }
    }

    private func encodePlain(_ text: String) -> [Int32] {
        var out: [Int32] = []
        if !text.isEmpty {
            for match in text.matches(of: splitRegex.regex) {
                var mapped = ""
                for byte in String(text[match.range]).utf8 {
                    mapped.append(byteToUni[byte]!)
                }
                for sym in bpe(mapped) where vocab[sym] != nil {
                    out.append(vocab[sym]!)
                }
            }
        }
        return out
    }

    private func bpe(_ token: String) -> [String] {
        var word = token.map { String($0) }
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
                    if i < word.count - 1
                        && word[i] == a && word[i + 1] == b {
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

    public func decode(_ ids: [Int32]) -> String {
        String(decoding: decodeBytes(ids), as: UTF8.self)
    }

    // Raw byte stream for the ids (byte-level BPE): lets a streaming caller
    // hold an incomplete trailing UTF-8 sequence (a multi-token emoji) instead
    // of emitting an uncorrectable replacement char.
    public func decodeBytes(_ ids: [Int32]) -> [UInt8] {
        var bytes: [UInt8] = []
        for id in ids where idToBytes[id] != nil {
            bytes.append(contentsOf: idToBytes[id]!)
        }
        return bytes
    }

    // Token count (for sizing a Sampler's vocab bound), no byte table built.
    public var vocabCount: Int { idToBytes.count }

    // Every token's byte string, indexed by id, for the grammar mask (it
    // forbids tokens by id, so needs the ordered vocab as bytes; tokens can be
    // invalid UTF-8 in isolation, hence bytes).
    public func vocabBytes() -> [[UInt8]] {
        var out = [[UInt8]](repeating: [], count: idToBytes.count)
        for (id, bytes) in idToBytes where Int(id) < out.count && id >= 0 {
            out[Int(id)] = bytes
        }
        return out
    }

    // Longest prefix of `b` ending on a complete UTF-8 boundary (a trailing
    // lead byte missing its continuation bytes is excluded). This is a SCALAR
    // boundary, not a grapheme one: a multi-scalar emoji streams scalar-by-
    // scalar and the UI reassembles the cluster.
    public static func completeUTF8Count(_ b: [UInt8]) -> Int {
        var j = b.count - 1
        var cont = 0
        while j >= 0, (b[j] & 0xC0) == 0x80 { cont += 1; j -= 1 }
        var n = b.count
        if j >= 0 {
            let lead = b[j]
            let need: Int
            if lead & 0x80 == 0 {
                need = 0
            } else if lead & 0xE0 == 0xC0 {
                need = 1
            } else if lead & 0xF0 == 0xE0 {
                need = 2
            } else if lead & 0xF8 == 0xF0 {
                need = 3
            } else {
                need = 0
            }
            n = cont == need ? b.count : j
        }
        return n
    }
}

// Incremental detokenizer for streaming. Feed the growing id list each step;
// it returns text completed since the last call, cut on UTF-8 scalar
// boundaries so a multi-token character is never split. Diffs by BYTE length,
// not grapheme count: a trailing scalar that merges into the current grapheme
// (U+FE0F, a skin-tone modifier, a ZWJ join) does not grow the grapheme count,
// so a count-based diff would drop it. Byte diffing emits every scalar.
public struct StreamDecoder {
    private var shownBytes = 0

    public init() {}

    public mutating func step(_ ids: [Int32], _ tokenizer: Tokenizer) -> String {
        let bytes = tokenizer.decodeBytes(ids)
        let n = Tokenizer.completeUTF8Count(bytes)
        var piece = ""
        if n > shownBytes {
            piece = String(decoding: bytes[shownBytes ..< n], as: UTF8.self)
            shownBytes = n
        }
        return piece
    }
}

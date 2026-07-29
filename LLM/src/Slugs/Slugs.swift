// The Slugs semantic-search engine: a MiniLM sentence embedder plus a vector
// index, tying a natural-language query to matching Simple English Wikipedia
// articles WITHOUT the query leaving the device -- only the resolved article id
// is fetched online. The privacy-preserving alternative to a web search.
//
// The two seams are deliberately general so wikipedia is just "the first
// corpus": an Embedder (MiniLM here) turns text into a unit vector, and a
// VectorIndex maps a vector to ranked records. Today the only index is the
// prebuilt SignIndex read from the model's GGUF trailer.
//
// FUTURE (documented, no code): local-document RAG. Add a FloatIndex (store raw
// 384-d embeddings, exact cosine -- simpler + more accurate for the small
// on-device corpora where sign-bit compaction is not needed) and a passage
// chunker (MiniLM caps at 512 tokens), building a per-collection index under
// applicationSupport/<id>/index/ from documents in .../<id>/Documents/. The
// Embedder is shared unchanged; only the index backend and a chunker are new.
import Foundation

// Text -> unit-normalized embedding. MiniLM is the only conformer today.
protocol Embedder {
    var dim: Int { get }
    func embed(_ text: String) -> [Float]
}

// Embedding -> ranked article records. SignIndex (prebuilt, PCA+sign+Hamming)
// is today's only conformer; FloatIndex (exact cosine) is the future local-RAG
// backend.
protocol VectorIndex {
    var count: Int { get }
    func search(_ emb: [Float], topK: Int) -> [(index: Int, distance: Int)]
    func record(_ i: Int) -> (id: String, title: String)
}

// A loaded Slugs engine over one self-contained GGUF (MiniLM weights + the SLGX
// index trailer). Both the embedder and the index read straight from the GGUF's
// mmap (weights dequantized per row on the fly, keys Hamming-scanned in place),
// so a live engine pins only the ~36 MB file-backed mapping -- clean, evictable,
// nothing to free -- and can be created per query cheaply or held indefinitely.
public final class WikiSlugs {
    private let gguf: GGUF
    private let embedder: MiniLM
    private let index: SignIndex

    public var articleCount: Int { index.count }

    // The minilm.gguf shipped inside this package (Slugs/minilm.gguf), so
    // every consumer resolves the same copy. Bundle.module exists only under
    // SwiftPM; the app links LLM as a native framework, whose own bundle
    // (found via this class) carries the resource instead.
    public static var bundledModel: URL? {
        #if SWIFT_PACKAGE
        Bundle.module.url(forResource: "minilm", withExtension: "gguf")
        #else
        Bundle(for: WikiSlugs.self).url(forResource: "minilm",
                                        withExtension: "gguf")
        #endif
    }

    // Fails when the file is not a GGUF or carries no SLGX index trailer.
    public init?(ggufPath: String) {
        guard let g = try? GGUF(path: ggufPath),
              let idx = SignIndex(gguf: g) else {
            return nil
        }
        gguf = g
        embedder = MiniLM(gguf: g)
        index = idx
    }

    // The raw MiniLM unit embedding of `text` -- exposed for the C-reference
    // parity gate and reusable by a future local-document index.
    public func embed(_ text: String) -> [Float] {
        embedder.embed(text)
    }

    // The topK best-matching articles for a query, nearest first. The query
    // text is embedded and matched entirely on-device; nothing is sent out.
    public func query(_ text: String, topK: Int = 5) -> [SlugHit] {
        let emb = embedder.embed(text)
        var hits: [SlugHit] = []
        for hit in index.search(emb, topK: topK) {
            let r = index.record(hit.index)
            hits.append(SlugHit(id: r.id, title: r.title,
                                distance: hit.distance))
        }
        return hits
    }

    // Exact-title rescue for an embedding miss: the LONGEST article title
    // (>= 3 chars) appearing as whole words inside `text`, distance 0 --
    // "what does wikipedia say about dark matter" names "Dark matter"
    // verbatim even when its embedding lands past the confidence cutoff.
    // A linear scan over the title table, so callers run it only on a
    // low-confidence result, where a network fetch would follow anyway.
    public func titleMatch(_ text: String) -> SlugHit? {
        let hay = " " + WikiSlugs.foldWords(text) + " "
        var best: (id: String, title: String)? = nil
        var bestLen = 2
        for i in 0 ..< index.count {
            let r = index.record(i)
            let folded = WikiSlugs.foldWords(r.title)
            if folded.count > bestLen,
               !WikiSlugs.functionWords.contains(folded),
               hay.contains(" " + folded + " ") {
                best = r
                bestLen = folded.count
            }
        }
        return best.map { r in
            SlugHit(id: r.id, title: r.title, distance: 0)
        }
    }

    // Single function words are article titles too (the corpus really has
    // "This"); a question's SUBJECT is never one, so they must not rescue.
    // Multi-word titles pass -- "The Who" is a real subject.
    private static let functionWords: Set<String> = [
        "a", "an", "the", "this", "that", "these", "those", "it", "its",
        "is", "are", "was", "were", "be", "been", "being", "do", "does",
        "did", "have", "has", "had", "will", "would", "can", "could",
        "should", "shall", "may", "might", "must", "and", "or", "but",
        "not", "no", "yes", "if", "then", "than", "so", "as", "of", "in",
        "on", "at", "to", "for", "with", "by", "from", "about", "into",
        "over", "under", "again", "there", "here", "when", "where", "why",
        "how", "what", "who", "whom", "which", "you", "your", "yours",
        "me", "my", "mine", "we", "us", "our", "ours", "he", "him", "his",
        "she", "her", "hers", "they", "them", "their", "theirs", "all",
        "any", "some", "such", "only", "very", "just", "also", "tell",
        "say", "says", "know", "like", "please", "thing", "things",
    ]

    // Case-fold and collapse every non-alphanumeric run to one space, so
    // "St. Louis" and "st louis" compare equal and the space-padded
    // containment check above is a whole-word match.
    static func foldWords(_ s: String) -> String {
        var out = ""
        var pendingSpace = false
        for c in s.lowercased() {
            if c.isLetter || c.isNumber {
                if pendingSpace && !out.isEmpty { out.append(" ") }
                pendingSpace = false
                out.append(c)
            } else {
                pendingSpace = true
            }
        }
        return out
    }
}

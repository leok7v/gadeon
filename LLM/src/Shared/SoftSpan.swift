import Foundation

// One attachment resolved into everything a turn needs from it: the
// placeholder id the chat template emitted for it, the id block that replaces
// that placeholder, and the tower rows spliced over the block's placeholder
// positions.
//
// The template decides WHERE a modality sits in the turn -- it writes one
// `<|image|>` / `<|audio|>` / `<|video|>` per attachment -- and only the
// expansion lives here. That is the same split HF draws between jinja and the
// processor, so a template change moves the attachment without touching code.
//
// `ids` is a whole BLOCK rather than a count because the shapes differ: an
// image or a clip is `begin + placeholder * N + end`, while a video repeats
// that per frame with an mm:ss stamp before each. And N is not a constant --
// a native-resolution tower's output follows the aspect ratio, so it is known
// only after the tower has run.
public struct SoftSpan: Sendable {
    public let placeholder: Int32
    public let ids: [Int32]
    public let features: [Float]

    public init(placeholder: Int32, ids: [Int32], features: [Float]) {
        self.placeholder = placeholder
        self.ids = ids
        self.features = features
    }

    // Placeholder positions in the block, which is how many feature rows it
    // has to carry.
    public var rows: Int {
        ids.filter { id in id == placeholder }.count
    }

    // The id shape every modality shares: begin, N placeholders, end.
    public static func bracket(begin: Int32, placeholder: Int32,
                               end: Int32, count: Int) -> [Int32] {
        var out = [begin]
        out.append(contentsOf: [Int32](repeating: placeholder, count: count))
        out.append(end)
        return out
    }

    public static func bracketed(begin: Int32, placeholder: Int32,
                                 end: Int32, count: Int,
                                 features: [Float]) -> SoftSpan {
        SoftSpan(placeholder: placeholder,
                 ids: bracket(begin: begin, placeholder: placeholder,
                              end: end, count: count),
                 features: features)
    }
}

// The per-position feature lookup a soft prefill drives. Each placeholder id
// hands out its own rows in order, so ONE turn can carry several modalities:
// an image and a clip are different placeholder ids walking independent
// cursors, and two images share a cursor in the order the template emitted
// them.
//
// A class because the prefill loop asks it once per token and the cursors have
// to survive between those calls.
public final class SoftFeed {
    private var rows: [Int32: [Float]] = [:]
    private var width: [Int32: Int] = [:]
    private var taken: [Int32: Int] = [:]

    public init(_ spans: [SoftSpan]) {
        for span in spans {
            let n = span.rows
            precondition(n > 0 && span.features.count % n == 0,
                         "span carries \(span.features.count) floats for "
                         + "\(n) placeholder positions")
            rows[span.placeholder, default: []]
                .append(contentsOf: span.features)
            width[span.placeholder] = span.features.count / n
        }
    }

    // The feature for a placeholder position, nil for an ordinary token.
    public func row(_ id: Int32) -> [Float]? {
        var out: [Float]? = nil
        if let e = width[id], let all = rows[id] {
            let at = (taken[id] ?? 0) * e
            if at + e <= all.count {
                out = Array(all[at ..< (at + e)])
                taken[id] = (taken[id] ?? 0) + 1
            }
        }
        return out
    }
}

import Foundation

// Shared pure helpers for the ChatSession turn loop: vision pad expansion
// and the runaway loop breaker. Offline-testable, no backend touched.
public enum Continuation {
    // Replicate each `pad` token into its image's own number of copies -- the
    // tower's merged vision rows the chat template emits a single placeholder
    // for. `counts` is per image in order of appearance; a tower with a fixed
    // rate passes one value per image all the same.
    //
    // Per image rather than one rate because a NATIVE-RESOLUTION tower strips
    // padding after pooling, so its output length follows the aspect ratio:
    // gemma-4 measures 252 to 273 across the fixture set against a configured
    // "280" that is a ceiling no real image reaches. The starts are a running
    // offset for the same reason -- with unequal runs there is no stride.
    // Missing entries fall back to the last count, so a short list behaves
    // like the fixed-rate case rather than dropping images.
    static func expandPads(_ ids: [Int32], pad: Int32, counts: [Int])
        -> (ids: [Int32], starts: [Int]) {
        var out: [Int32] = []
        var starts: [Int] = []
        var image = 0
        for id in ids {
            if id == pad {
                let n = counts.isEmpty ? 0
                    : counts[min(image, counts.count - 1)]
                starts.append(out.count)
                out.append(contentsOf: Array(repeating: pad, count: n))
                image += 1
            } else {
                out.append(id)
            }
        }
        return (out, starts)
    }

    // Every image at the same rate.
    static func expandPads(_ ids: [Int32], pad: Int32, count: Int)
        -> (ids: [Int32], starts: [Int]) {
        expandPads(ids, pad: pad, counts: [count])
    }

    // Replace each span's placeholder with that span's own id block. The
    // template writes ONE placeholder per attachment, so several spans sharing
    // a placeholder id are consumed in the order the template emitted them.
    //
    // A block's OWN inner placeholders pass through untouched -- they are the
    // soft positions the features land on, not further attachments -- which is
    // why this walks the input once rather than expanding what it appends.
    public static func expandSpans(_ ids: [Int32], _ spans: [SoftSpan]) -> [Int32] {
        var blocks: [Int32: [[Int32]]] = [:]
        for span in spans {
            blocks[span.placeholder, default: []].append(span.ids)
        }
        var used: [Int32: Int] = [:]
        var out: [Int32] = []
        for id in ids {
            let at = used[id] ?? 0
            if let queue = blocks[id], at < queue.count {
                out.append(contentsOf: queue[at])
                used[id] = at + 1
            } else {
                out.append(id)
            }
        }
        return out
    }

    // Runaway loop breaker: an uncapped no-EOS decode often degenerates into
    // a cycle. True when the tail is a k-gram (k=1..4) repeated `reps` times
    // back to back (5 identical short blocks is a loop, not repetition), or a
    // LONGER k-gram (5..longK) repeated `longReps` times -- a whole sentence
    // or paragraph re-emitted verbatim cycles with a period the short scan
    // cannot see, and 3 identical >=5-token blocks in a row is degeneration,
    // not style. A k-gram of ONLY structural punctuation is legitimate
    // repetition -- a wide markdown table's separator row emits "|---" per
    // column, a rule or ASCII-art border repeats "-"/"=" far past 5 -- so
    // when `tokenBytes` is provided such a tail needs `structuralReps`
    // repeats to count as a loop (a real degeneration is infinite, so it
    // still trips, just later).
    static func isLooping(_ ids: [Int32], reps: Int = 5,
                          structuralReps: Int = 24,
                          longK: Int = 64, longReps: Int = 3,
                          tokenBytes: ((Int32) -> [UInt8])? = nil) -> Bool {
        var result = false
        var k = 1
        while k <= longK && !result {
            let structural = tokenBytes.map { bytes in
                structuralGram(ids, k, bytes)
            } ?? false
            let need = structural ? structuralReps
                                  : (k <= 4 ? reps : longReps)
            result = tailRepeats(ids, k, need)
            k += 1
        }
        return result
    }

    // Table / rule / border bytes plus whitespace, and numerals with the
    // group comma: a large number is a LEGITIMATE ",000" cycle ("6,000,000"
    // and up), repeating the same way a table's separator row repeats.
    private static let structuralBytes: Set<UInt8> =
        Set("|-=+:_*#~. \t\n0123456789,".utf8)

    private static func structuralGram(_ ids: [Int32], _ k: Int,
                                       _ bytes: (Int32) -> [UInt8]) -> Bool {
        var structural = ids.count >= k
        var i = max(0, ids.count - k)
        while structural && i < ids.count {
            let b = bytes(ids[i])
            structural = !b.isEmpty && b.allSatisfy { byte in
                structuralBytes.contains(byte)
            }
            i += 1
        }
        return structural
    }

    // Whether the last `k` tokens repeat as `reps` identical back-to-back
    // blocks.
    private static func tailRepeats(_ ids: [Int32], _ k: Int,
                                    _ reps: Int) -> Bool {
        var result = ids.count >= k * reps
        var block = 1
        while block < reps && result {
            var i = 0
            while i < k && result {
                let a = ids[ids.count - 1 - i]
                let b = ids[ids.count - 1 - block * k - i]
                if a != b { result = false }
                i += 1
            }
            block += 1
        }
        return result
    }
}

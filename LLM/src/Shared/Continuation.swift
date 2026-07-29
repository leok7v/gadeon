import Foundation

// Shared pure helpers for the ChatSession turn loop: vision pad expansion
// and the runaway loop breaker. Offline-testable, no backend touched.
enum Continuation {
    // Replicate each `pad` token into `count` copies -- the tower's merged
    // vision rows the chat template emits a single <|image_pad|> for. Returns
    // the expanded ids and the start index of each expanded run, so the caller
    // can splice the image feats and (re)build a KV whose length matches the
    // committed vision state across turns.
    static func expandPads(_ ids: [Int32], pad: Int32, count: Int)
        -> (ids: [Int32], starts: [Int]) {
        var out: [Int32] = []
        var starts: [Int] = []
        for id in ids {
            if id == pad {
                starts.append(out.count)
                out.append(contentsOf: Array(repeating: pad, count: count))
            } else {
                out.append(id)
            }
        }
        return (out, starts)
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

import Foundation

public enum Continuation {
    // Replace each span's placeholder with that span's own id block. The
    // template writes ONE placeholder per attachment, so several spans sharing
    // a placeholder id are consumed in the order the template emitted them.
    //
    // A block's OWN inner placeholders pass through untouched -- they are the
    // soft positions the features land on, not further attachments -- which is
    // why this walks the input once rather than expanding what it appends.
    public static func expandSpans(_ ids: [Int32], _ spans: [SoftSpan]) -> [Int32] {
        var queues: [Int32: [SoftSpan]] = [:]
        for span in spans {
            queues[span.placeholder, default: []].append(span)
        }
        var used: [Int32: Int] = [:]
        var out: [Int32] = []
        var i = 0
        while i < ids.count {
            let id = ids[i]
            let at = used[id] ?? 0
            if let queue = queues[id], at < queue.count {
                let span = queue[at]
                if let wrap = span.wrap, out.last == wrap.begin {
                    out.removeLast()
                }
                out.append(contentsOf: span.ids)
                used[id] = at + 1
                if let wrap = span.wrap, i + 1 < ids.count,
                   ids[i + 1] == wrap.end {
                    i += 1
                }
            } else {
                out.append(id)
            }
            i += 1
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

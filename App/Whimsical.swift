import Foundation

// Playful per-stage status phrases, one bundled App/Whimsical/*.txt list per
// Stage (reasoning / prefill / documents / vision / consulting / optimizing),
// each line "emoji Word emoji". Shown in the status bar, as the transcript's
// live "working" line, and on the one-time Optimizing screen. trio() returns
// three distinct phrases, one per live surface.

enum Whimsical {

    enum Stage: String {
        case reasoning, prefill, documents, vision, consulting, optimizing,
             downloading
    }

    static let reasoning = load("reasoning").shuffled()
    static let prefill = load("prefill").shuffled()
    static let documents = load("documents").shuffled()
    static let vision = load("vision").shuffled()
    static let consulting = load("consulting").shuffled()
    static let optimizing = load("optimizing").shuffled()
    static let downloading = load("downloading").shuffled()

    static func list(_ stage: Stage) -> [String] {
        let result: [String]
        switch stage {
        case .reasoning: result = reasoning
        case .prefill: result = prefill
        case .documents: result = documents
        case .vision: result = vision
        case .consulting: result = consulting
        case .optimizing: result = optimizing
        case .downloading: result = downloading
        }
        return result
    }

    private static func load(_ name: String) -> [String] {
        let url = Bundle.main.url(forResource: name, withExtension: "txt")
        let text = url.flatMap { u in
            try? String(contentsOf: u, encoding: .utf8)
        }
        var seen: Set<String> = []
        var out: [String] = []
        for line in (text ?? "").split(whereSeparator: \.isNewline) {
            let phrase = line.trimmingCharacters(in: .whitespaces)
            if !phrase.isEmpty, seen.insert(phrase).inserted {
                out.append(phrase)
            }
        }
        return out.isEmpty ? ["Thinking"] : out
    }

    // Per-stage ring cursors: one walks up, one walks down, both wrapping.
    // Guarded by a lock (callers span MainActor tasks and view inits).
    private static let lock = NSLock()
    nonisolated(unsafe) private static var cursors: [Stage: (Int, Int)] = [:]
    nonisolated(unsafe) private static var held:
        [Stage: (idx: Int, at: Date)] = [:]

    // The stage's CURRENT phrase, advancing through the shuffled ring at most
    // once per `hold` seconds NO MATTER how many views or tickers ask --
    // concurrent tasks and view re-inits are idempotent reads and cannot
    // fast-forward or ping-pong the rotation.
    static func current(_ stage: Stage,
                        hold: TimeInterval = 7) -> String {
        let all = list(stage)
        var result = "Thinking"
        if !all.isEmpty {
            lock.lock()
            var (idx, at) = held[stage] ?? (0, Date())
            if Date().timeIntervalSince(at) >= hold {
                idx = (idx + 1) % all.count
                at = Date()
            }
            held[stage] = (idx, at)
            result = all[idx]
            lock.unlock()
        }
        return result
    }

    // Three phrases from the stage's shuffled ring, one per surface that can
    // show a phrase at once (transcript quip, reasoning label, status bar).
    // The up and down cursors wrap around; the third rides one past the up
    // cursor; collisions skip a slot, so with three or more phrases in the
    // list no two surfaces ever show the same word.
    static func trio(_ stage: Stage) -> (first: String, second: String,
                                         third: String) {
        let all = list(stage)
        let n = all.count
        var result = (first: "Thinking", second: "Thinking",
                      third: "Thinking")
        if n > 0 {
            lock.lock()
            var (up, down) = cursors[stage] ?? (0, n - 1)
            if up == down { down = (down + n - 1) % n }
            var mid = (up + 1) % n
            if mid == down { mid = (mid + 1) % n }
            result = (first: all[up], second: all[down], third: all[mid])
            cursors[stage] = ((up + 1) % n, (down + n - 1) % n)
            lock.unlock()
        }
        return result
    }

}

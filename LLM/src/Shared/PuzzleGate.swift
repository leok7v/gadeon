import Foundation

public struct Puzzle: Sendable {
    public let category: String
    public let prompt: String
    public let accept: [String]

    public init(category: String, prompt: String, accept: [String]) {
        self.category = category
        self.prompt = prompt
        self.accept = accept
    }
}

public enum PuzzleVerdict: String, Sendable, Codable {
    case pass = "PASS"
    case fail = "FAIL"
    case incomplete = "INCM"
}

public struct PuzzleRun: Sendable, Codable {
    public var category = ""
    public var prompt = ""
    public var verdict = PuzzleVerdict.incomplete
    public var seconds = 0
    public var reasoning = ""
    public var content = ""

    public init() {}
}

public enum PuzzleGate {

    // The answer is what follows the reasoning, and only its tail: a long
    // deliberation contains the trap answer, every discarded candidate, and
    // the phrase "no, wait".
    static let window = 300

    // Six of the M4 host's sixteen: the ones with ONE short answer a live
    // model can reach. The knave / zebra / lateral / rebus / anagram items
    // accept several defensible answers or need a paragraph, which scores the
    // scorer rather than the model. The full set is parked in tmp/.
    public static let puzzles: [Puzzle] = [
        Puzzle(category: "trap",
               prompt: "A bat and a ball cost 1.10 dollars in total. The bat "
                   + "costs 1.00 dollar more than the ball. How much does "
                   + "the ball cost?",
               accept: [#"0\.05"#, #"\b5 cents"#, "five cents"]),
        Puzzle(category: "trap",
               prompt: "If 5 machines take 5 minutes to make 5 widgets, how "
                   + "long would 100 machines take to make 100 widgets?",
               accept: [#"\b5 minutes"#, #"\bfive minutes"#]),
        Puzzle(category: "trap",
               prompt: "A lily patch doubles in size every day and covers "
                   + "the whole lake on day 48. On which day does it cover "
                   + "half the lake?",
               accept: [#"\b47\b"#, "forty[- ]seven"]),
        Puzzle(category: "trap",
               prompt: "Sally has 3 brothers. Each brother has 2 sisters. "
                   + "How many sisters does Sally have?",
               accept: [#"\bone\b"#, #"\b1\b"#]),
        Puzzle(category: "trap",
               prompt: "In a race you overtake the person in second place. "
                   + "What place are you in now?",
               accept: ["second", #"\b2nd"#]),
        Puzzle(category: "zebra",
               prompt: "Anna, Ben and Carl each drink one of tea, milk, "
                   + "juice. Anna does not drink tea. Ben drinks neither tea "
                   + "nor juice. What does Carl drink?",
               accept: [#"\btea\b"#]),
    ]

    // Markdown emphasis around the answer word is why a literal "1 sister"
    // pattern misses "**1** sister".
    public static func normalize(_ text: String) -> String {
        let stripped = text.filter { c in !"*_`$".contains(c) }
        return stripped.split(whereSeparator: { c in c.isWhitespace })
            .joined(separator: " ").lowercased()
    }

    public static func verdict(_ puzzle: Puzzle,
                               content: String) -> PuzzleVerdict {
        let answer = normalize(content)
        let tail = String(answer.suffix(window))
        var out = PuzzleVerdict.incomplete
        if !tail.isEmpty {
            out = puzzle.accept.contains { pattern in
                matches(pattern, tail)
            } ? .pass : .fail
        }
        return out
    }

    public static func matches(_ pattern: String, _ text: String) -> Bool {
        var found = false
        if let re = try? NSRegularExpression(pattern: pattern) {
            found = re.firstMatch(in: text,
                                  range: NSRange(text.startIndex...,
                                                 in: text)) != nil
        }
        return found
    }

    public static func score(_ runs: [PuzzleRun]) -> (Int, Int) {
        let passed = runs.filter { r in r.verdict == .pass }.count
        let stalled = runs.filter { r in r.verdict == .incomplete }.count
        return (passed, stalled)
    }
}

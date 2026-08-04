import Foundation
import Testing

// The gemma cases run against the REAL repacked GGUF -- several GB, so it is
// not in the tree -- and are gated on GADEON_GEMMA_GGUF pointing at it.
//
// A TRAIT rather than an `if` inside each body, because the two report
// differently and the difference matters: a guarded body that quietly does
// nothing is reported as PASSED, so a plain `swift test` reads as though it
// covered the gemma engine, tower and speech paths when it ran none of them.
// Skipped says so, and names the variable that would turn them on.
let gemmaGgufPath = ProcessInfo.processInfo.environment["GADEON_GEMMA_GGUF"]

let needsGemmaWeights = ConditionTrait.enabled(
    if: gemmaGgufPath != nil,
    "set GADEON_GEMMA_GGUF to the repacked gemma-4 GGUF")

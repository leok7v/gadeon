import CoreML
import Foundation

// Per-op compute-device audit via MLComputePlan: WHERE did each op land? The
// one honest answer to "does this program really run on the Neural Engine".
// A single silent CPU/GPU fallback is invisible at runtime (the tokens are
// still correct, they were merely computed on the wrong processor) and only a
// placement plan reveals it. Swift counterpart of the Python placement
// gate, with the SAME rules so the two agree:
//   * `const` is data, not compute -- never counted as an offender.
//   * an op whose usage is nil was NOT planned. MLComputePlan plans only the
//     model's default function, so in a multifunction package every
//     non-default function reports all-nil; such ops are skipped, and a
//     function with zero planned ops is "not planned" (no signal), not a
//     failure.
//   * PASS = >= 99% of the planned ops landed on the Neural Engine.
//
// Everything CoreML (the plan, the model structure, the operations) lives and
// dies inside `audit` -- nothing non-Sendable escapes it -- so the API is
// clean under Swift 6 strict concurrency. `Result` is Ints/Strings only.

public enum Placement {
    // Not 100: a graph is allowed a handful of genuinely-unknown ops without
    // being called a fallback. Matches placement.py's MIN_NE_PERCENT.
    public static let minNEPercent = 99.0
    // A `const` reports off-engine because it is data; it is not an offender.
    static let notCompute: Set<String> = ["const"]

    public struct Result: Sendable {
        public let label: String
        public var ne = 0
        public var gpu = 0
        public var cpu = 0
        public var unknown = 0
        public var offenders: [String: Int] = [:]

        public var placed: Int { ne + gpu + cpu }

        // False for a function MLComputePlan did not plan (every op nil) --
        // such a result carries no signal and must be skipped, not failed.
        public var planned: Bool { placed > 0 }

        public var pctNE: Double {
            placed > 0 ? 100.0 * Double(ne) / Double(placed) : 0
        }

        // Two per-function audits sharing a fingerprint mean MLComputePlan
        // ignored functionName and planned the one default function twice; the
        // caller collapses those rather than print per-function numbers it
        // would be inventing.
        public var counts: [Int] { [ne, gpu, cpu, unknown] }
    }

    // Audit ONE compiled .mlmodelc, optionally selecting a function. Loads the
    // compute plan, walks the program's ops, and tallies where each planned op
    // landed. `computeUnits` should match how the engine loads the set
    // (.cpuAndNeuralEngine in production).

    public static func audit(
        _ url: URL, function: String? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        label: String? = nil
    ) async throws -> Result {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        if let function { cfg.functionName = function }
        let plan = try await MLComputePlan.load(contentsOf: url,
                                                configuration: cfg)
        let named = url.lastPathComponent
            + (function.map { fn in ":\(fn)" } ?? "")
        var result = Result(label: label ?? named)
        if case let .program(program) = plan.modelStructure {
            for (_, fn) in program.functions {
                for op in fn.block.operations {
                    if let usage = plan.deviceUsage(for: op) {
                        tally(&result, usage.preferred, op.operatorName)
                    }
                }
            }
        }
        return result
    }

    // Fold one planned op into the tally, naming the offender when it landed
    // anywhere but the Neural Engine.

    private static func tally(_ result: inout Result,
                              _ device: MLComputeDevice, _ op: String) {
        switch device {
        case .neuralEngine:
            result.ne += 1
        case .gpu:
            result.gpu += 1
            result.offenders["GPU:\(op)", default: 0] += 1
        case .cpu:
            result.cpu += 1
            if !notCompute.contains(op) {
                result.offenders["CPU:\(op)", default: 0] += 1
            }
        @unknown default:
            result.unknown += 1
        }
    }

    // True when EVERY planned function meets the threshold. A set with no
    // planned function at all cannot pass (there is nothing to judge). Mirrors
    // placement.py gate().

    public static func gate(_ results: [Result],
                            minNE: Double = minNEPercent) -> Bool {
        let planned = results.filter { r in r.planned }
        return !planned.isEmpty
            && planned.allSatisfy { r in r.pctNE >= minNE }
    }
}

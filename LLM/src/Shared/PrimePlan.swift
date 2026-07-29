import Foundation

// The compile contract a cache-priming helper walks: which functions of a
// compiled program to load, in the order the engine will need them. The
// program's own metadata.json says which functions it carries, so the plan
// never asks for one it does not (CoreML logs such a miss as an error) --
// one contract serves MTP and non-MTP sets, merged and legacy layouts alike.
public enum PrimePlan {

    // Candidate function names for one .mlmodelc, in engine load order
    // (essential set first: mf decode before prefill, head embed before the
    // rest); nil = a plain (single-function) load.
    public static func functions(for name: String) -> [String?] {
        let result: [String?]
        if name.hasPrefix("mf") {
            result = ["decode", "prefill", "verify"]
        } else if name == "head.mlmodelc" {
            result = ["embed", "lm_head", "embed_batched", "lm_head_batched"]
        } else {
            result = [nil]
        }
        return result
    }

    // The functions a compiled program actually carries, read from its
    // metadata.json ([0].functions[].name); empty = single-function.
    public static func available(at url: URL) -> [String] {
        var out: [String] = []
        if let data = try? Data(contentsOf:
               url.appendingPathComponent("metadata.json")),
           let arr = try? JSONSerialization.jsonObject(with: data) as? [Any],
           let first = arr.first as? [String: Any],
           let fns = first["functions"] as? [[String: Any]] {
            out = fns.compactMap { fn in fn["name"] as? String }
        }
        return out
    }

    // What a primer compiles for this program: the preference order
    // intersected with what the program carries.
    public static func plan(at url: URL) -> [String?] {
        let present = available(at: url)
        let result: [String?]
        if present.isEmpty {
            result = [nil]
        } else {
            result = functions(for: url.lastPathComponent).filter { fn in
                fn.map { present.contains($0) } ?? false
            }
        }
        return result
    }

    // The name[:function] comma spec gadeon-cli's --compile mode consumes.
    public static func spec(at url: URL) -> String {
        plan(at: url)
            .map { fn in
                fn.map { "\(url.lastPathComponent):\($0)" }
                    ?? url.lastPathComponent
            }
            .joined(separator: ",")
    }
}

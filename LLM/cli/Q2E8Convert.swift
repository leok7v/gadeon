import Foundation
import LLM

func runQ2E8Convert(_ args: [String]) {
    let at = args.firstIndex(of: "--q2e8-convert")!
    let dir = at + 1 < args.count ? args[at + 1] : ""
    let out = at + 2 < args.count ? args[at + 2] : ""
    var trunk = "q2_x"
    if let i = args.firstIndex(of: "--trunk"), i + 1 < args.count {
        trunk = args[i + 1]
    }
    var embd = ""
    if let i = args.firstIndex(of: "--embd"), i + 1 < args.count {
        embd = args[i + 1]
    }
    var wide: Set<String> = ["ffn_down.weight"]
    if let i = args.firstIndex(of: "--wide"), i + 1 < args.count {
        wide = Set(args[i + 1].split(separator: ",").map(String.init)
            .filter { s in !s.isEmpty })
    }
    var gate: String? = nil
    if let i = args.firstIndex(of: "--gate"), i + 1 < args.count {
        gate = args[i + 1]
    }
    let renorm = !args.contains("--no-renorm")
    var imat: String? = nil
    if let i = args.firstIndex(of: "--imat"), i + 1 < args.count {
        imat = args[i + 1]
    }
    var hess: String? = nil
    if let i = args.firstIndex(of: "--hess"), i + 1 < args.count {
        hess = args[i + 1]
    }
    let gptq = !args.contains("--no-gptq")
    let lowRank = args.contains("--lowrank")
    let rotate = args.contains("--rotate")
    let vision = !args.contains("--no-vision")
    let spin = args.contains("--spin")
    let untie = args.contains("--untie")
    let e8 = args.contains("--e8")
    let e8p = args.contains("--e8p")
    var damp = 0.0
    if let i = args.firstIndex(of: "--damp"), i + 1 < args.count {
        damp = Double(args[i + 1]) ?? 0
    }
    var mu: Float = 0
    if let i = args.firstIndex(of: "--mu"), i + 1 < args.count {
        mu = Float(args[i + 1]) ?? 0
    }
    var rank = 0
    if let i = args.firstIndex(of: "--rank"), i + 1 < args.count {
        rank = Int(args[i + 1]) ?? 0
    }
    let reconstruct = args.contains("--reconstruct")
    var head = ""
    if let i = args.firstIndex(of: "--head"), i + 1 < args.count {
        head = args[i + 1]
    }
    var raise = ""
    if let i = args.firstIndex(of: "--raise"), i + 1 < args.count {
        raise = args[i + 1]
    }
    var status: Int32 = 0
    if dir.isEmpty || out.isEmpty {
        err("usage: gadeon-cli x --q2e8-convert <hf-dir> <out.gguf> "
            + "[--trunk q2_x|q2_e8|q4_0|ternary|bf16] [--embd type] [--wide a,b] "
            + "[--no-renorm] [--imat dir] [--hess dir] [--no-gptq] [--gate ref]\n")
        status = 2
    } else {
        let t0 = Date()
        do {
            let report = try Q2E8Convert.run(
                from: dir, to: out, trunk: trunk, embd: embd, wide: wide,
                renorm: renorm, imat: imat, hess: hess, gptq: gptq,
                lowRank: lowRank, rotate: rotate, vision: vision, raise: raise, spin: spin, untie: untie, head: head,
                gate: gate, reconstruct: reconstruct, e8: e8, e8p: e8p,
                rank: rank, mu: mu, damp: damp,
                log: { line in err(line + "\n") })
            err(String(format: "[q2e8] %.1f GB in %.0fs\n",
                       Double(report.bytes) / 1e9,
                       Date().timeIntervalSince(t0)))
            if gate != nil {
                err(String(format: "[q2e8] worst cosine %.4f on %@\n",
                           report.worstCosine.0, report.worstCosine.1))
                err(String(format: "[q2e8] worst exact rel %.2e on %@\n",
                           report.worstExact.0, report.worstExact.1))
            }
            if report.worstError.0 > 0 {
                err(String(format: "[q2e8] worst output error %.4f on %@\n",
                           report.worstError.0, report.worstError.1))
            }
        } catch {
            err("[q2e8] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

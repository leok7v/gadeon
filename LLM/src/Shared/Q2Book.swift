import Foundation

public enum Q2Book {

    public static let candidates: [Float] = [4, 5, 6, 7, 9, 12]

    public static func sweep(from dir: String, hess: String, imat: String?,
                             layers: [Int],
                             log: (String) -> Void) throws {
        let root = try Q2E8Convert.json(dir + "/config.json")
        let pre = (try? Q2E8Convert.json(dir
                                        + "/preprocessor_config.json")) ?? [:]
        let text = try TextShape(root)
        let eye = VisionShape(root, pre)
        let store = try Safetensors(dir: dir)
        let policy = Q2E8Policy(
            trunk: .q2_e8, embd: .q2_e8, wide: [],
            fit: Q2Fit(iterations: 8, renorm: false), imat: imat,
            hess: hess, gptq: true, ternary: false, lowRank: false,
            rotate: false, vision: false, raise: [:], spin: false,
            untie: false, head: .q2_e8, reconstruct: false, e8: false,
            e8p: false, rank: 0, mu: 0,
            damp: Q2GPTQ.damp)
        let plan = try Q2E8Convert.plans(store, text, eye, policy)
        var cache = HessianCache()
        for t in plan where keep(t, layers) {
            let site = Q2E8Convert.squares(t, policy, &cache)
            if site.h.isEmpty {
                log("  \(t.gguf): no hessian at "
                    + Q2E8Convert.siteOf(t.gguf))
            } else {
                try report(t, site, store, text, imat, log)
            }
        }
    }

    static func keep(_ t: Q2E8Tensor, _ layers: [Int]) -> Bool {
        let part = t.gguf.split(separator: ".")
        var out = false
        if part.count > 2, part[0] == "blk", let il = Int(part[1]) {
            out = layers.contains(il) && !Q2E8Convert.siteOf(t.gguf).isEmpty
        }
        return out
    }

    static func report(_ t: Q2E8Tensor, _ site: HessianCache,
                       _ store: Safetensors, _ text: TextShape,
                       _ imat: String?, _ log: (String) -> Void) throws {
        let raw = try store.values(t.source)
        let values = Q2E8Convert.apply(raw, t, text)
        let k = t.dims[0]
        let rows = values.count / k
        var fit = Q2Fit(iterations: 8, renorm: false)
        fit.importance = Q2E8Convert.importance(t.gguf, k, imat)
        let flat = Q2GPTQ.quantize(values, rows: rows, k: k, soa: false,
                                   opts: fit, ternary: false, hinv: site.hinv)
        let base = measure(values, flat.1, k, site.h)
        for a in candidates {
            var alt = fit
            alt.wide = a
            let got = Q2GPTQ.quantize(values, rows: rows, k: k, soa: false,
                                      opts: alt, ternary: false,
                                      hinv: site.hinv)
            let e = measure(values, got.1, k, site.h)
            log(String(format: "%-28s a=%-3.0f err %.4f -> %.4f  %+6.1f%%"
                       + "   wide %5.1f%% of blocks",
                       (t.gguf as NSString).utf8String!, Double(a),
                       base, e, 100 * (e - base) / max(base, 1e-30),
                       100 * picked(got.0, rows, k)))
        }
    }

    static func measure(_ values: [Float], _ back: [Float], _ k: Int,
                        _ h: [Float]) -> Float {
        var delta = values
        for i in 0..<delta.count { delta[i] -= back[i] }
        return Q2GPTQ.outputError(delta, values, rows: values.count / k,
                                  k: k, h)
    }

    static func picked(_ packed: [UInt8], _ rows: Int, _ k: Int) -> Double {
        let nblk = k / Q2E8.qk
        let rowBytes = nblk * Q2E8.blockBytes
        var wide = 0
        for r in 0..<rows {
            for b in 0..<nblk
            where packed[r * rowBytes + b * Q2E8.blockBytes + 1] & 0x80 != 0 {
                wide += 1
            }
        }
        return Double(wide) / Double(rows * nblk)
    }
}

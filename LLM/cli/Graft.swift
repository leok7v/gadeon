import Foundation
import LLM

func runMeta(_ args: [String]) {
    let at = args.firstIndex(of: "--meta")!
    let src = at + 1 < args.count ? args[at + 1] : ""
    let out = at + 2 < args.count ? args[at + 2] : ""
    let cfg = at + 3 < args.count ? args[at + 3] : ""
    var status: Int32 = 0
    if src.isEmpty || out.isEmpty || cfg.isEmpty {
        err("usage: gadeon-cli x --meta <src.gguf> <out.gguf> "
            + "<generation_config.json>\n")
        status = 2
    } else {
        do {
            let report = try GGUFGraft.meta(src, to: out, config: cfg)
            err(String(format: "[meta] %d tensors, %d keys, %d eos, "
                       + "%.1f GB\n", report.text, report.keys,
                       report.tower, Double(report.bytes) / 1e9))
        } catch {
            err("[meta] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

func runDrafter(_ args: [String]) {
    let at = args.firstIndex(of: "--drafter")!
    let text = at + 1 < args.count ? args[at + 1] : ""
    let donor = at + 2 < args.count ? args[at + 2] : ""
    let out = at + 3 < args.count ? args[at + 3] : ""
    var status: Int32 = 0
    if text.isEmpty || donor.isEmpty || out.isEmpty {
        err("usage: gadeon-cli x --drafter <text.gguf> <donor.gguf> "
            + "<out.gguf>\n")
        status = 2
    } else {
        let t0 = Date()
        do {
            let report = try GGUFGraft.graft(
                text: text, donor: donor, to: out, tensors: ["blk."],
                keys: ["qwen35.block_count",
                       "qwen35.nextn_predict_layers"])
            err(String(format: "[drafter] %d text + %d drafter tensors, "
                       + "%d keys, %.1f GB in %.1fs\n", report.text,
                       report.tower, report.keys,
                       Double(report.bytes) / 1e9,
                       Date().timeIntervalSince(t0)))
            if report.tower == 0 {
                err("[drafter] FAIL -- the donor adds no block\n")
                status = 1
            }
        } catch {
            err("[drafter] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

func runAssist(_ args: [String]) {
    let at = args.firstIndex(of: "--assist")!
    let text = at + 1 < args.count ? args[at + 1] : ""
    let donor = at + 2 < args.count ? args[at + 2] : ""
    let out = at + 3 < args.count ? args[at + 3] : ""
    var status: Int32 = 0
    if text.isEmpty || donor.isEmpty || out.isEmpty {
        err("usage: gadeon-cli x --assist <text.gguf> <donor.gguf> "
            + "<out.gguf>\n")
        status = 2
    } else {
        let t0 = Date()
        do {
            let report = try GGUFGraft.graft(
                text: text, donor: donor, to: out,
                tensors: ["assist."], keys: ["gemma4.assist."])
            err(String(format: "[assist] %d text + %d head tensors, "
                       + "%d keys, %.1f GB in %.1fs\n", report.text,
                       report.tower, report.keys,
                       Double(report.bytes) / 1e9,
                       Date().timeIntervalSince(t0)))
            if report.tower == 0 {
                err("[assist] FAIL -- the donor adds no head\n")
                status = 1
            }
        } catch {
            err("[assist] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

func runGraft(_ args: [String]) {
    let at = args.firstIndex(of: "--graft")!
    let text = at + 1 < args.count ? args[at + 1] : ""
    let donor = at + 2 < args.count ? args[at + 2] : ""
    let out = at + 3 < args.count ? args[at + 3] : ""
    var status: Int32 = 0
    if text.isEmpty || donor.isEmpty || out.isEmpty {
        err("usage: gadeon-cli x --graft <text.gguf> <donor.gguf> "
            + "<out.gguf>\n")
        status = 2
    } else {
        let t0 = Date()
        do {
            let report = try GGUFGraft.graft(text: text, donor: donor,
                                             to: out)
            err(String(format: "[graft] %d text + %d tower tensors, %d keys, "
                       + "%.1f GB in %.1fs\n", report.text, report.tower,
                       report.keys, Double(report.bytes) / 1e9,
                       Date().timeIntervalSince(t0)))
        } catch {
            err("[graft] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

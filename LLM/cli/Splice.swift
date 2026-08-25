import Foundation
import LLM

func runSplice(_ args: [String]) {
    let at = args.firstIndex(of: "--splice")!
    let base = at + 1 < args.count ? args[at + 1] : ""
    let donor = at + 2 < args.count ? args[at + 2] : ""
    let leaves = at + 3 < args.count ? args[at + 3] : ""
    let out = at + 4 < args.count ? args[at + 4] : ""
    var status: Int32 = 0
    if base.isEmpty || donor.isEmpty || leaves.isEmpty || out.isEmpty {
        err("usage: gadeon-cli x --splice <base.gguf> <donor.gguf> "
            + "<leaf,leaf,...> <out.gguf>\n")
        status = 2
    } else {
        let want = Set(leaves.split(separator: ",").map(String.init))
        do {
            let report = try GGUFGraft.splice(base: base, donor: donor,
                                              leaves: want, to: out)
            err(String(format: "[splice] %d of %d tensors from the donor, "
                       + "%.2f GB\n", report.tower, report.text,
                       Double(report.bytes) / 1e9))
            if report.tower == 0 {
                err("[splice] FAIL -- no tensor matched \(leaves)\n")
                status = 1
            }
        } catch {
            err("[splice] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

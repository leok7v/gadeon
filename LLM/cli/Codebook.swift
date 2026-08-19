import Foundation
import LLM

func runCodebook(_ args: [String]) {
    let at = args.firstIndex(of: "--codebook")!
    let dir = at + 1 < args.count ? args[at + 1] : ""
    let hess = at + 2 < args.count ? args[at + 2] : ""
    var imat: String? = nil
    if let i = args.firstIndex(of: "--imat"), i + 1 < args.count {
        imat = args[i + 1]
    }
    var layers = [2, 11, 20]
    if let i = args.firstIndex(of: "--layers"), i + 1 < args.count {
        layers = args[i + 1].split(separator: ",").compactMap { s in Int(s) }
    }
    var status: Int32 = 0
    if dir.isEmpty || hess.isEmpty {
        err("usage: gadeon-cli x --codebook <hf-dir> <hess-dir> "
            + "[--imat dir] [--layers 2,11,20]\n")
        status = 2
    } else {
        do {
            try Q2Book.sweep(from: dir, hess: hess, imat: imat,
                             layers: layers,
                             log: { line in err(line + "\n") })
        } catch {
            err("[codebook] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

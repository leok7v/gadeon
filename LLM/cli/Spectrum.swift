import Foundation
import LLM

func runSpectrum(_ args: [String]) {
    let at = args.firstIndex(of: "--spectrum")!
    let dir = at + 1 < args.count ? args[at + 1] : ""
    let names = ((try? FileManager.default.contentsOfDirectory(atPath: dir))
        ?? []).filter { n in n.hasSuffix(".h32") }.sorted()
    if names.isEmpty {
        err("usage: gadeon-cli x --spectrum <hess-dir>\n")
        exit(2)
    }
    print("site                     K      cond    top1%    top1      pr")
    for name in names {
        let raw = FileManager.default.contents(atPath: dir + "/" + name)
            ?? Data()
        let k = Int((Double(raw.count / 4)).squareRoot().rounded())
        if k * k * 4 == raw.count {
            var h = [Float](repeating: 0, count: k * k)
            h.withUnsafeMutableBytes { dst in _ = raw.copyBytes(to: dst) }
            let s = Eigen.of(h, k)
            print(String(format: "%-22s %5d  %8.1e  %6.1f%%  %6.1f%%  %6.1f",
                         (String(name.dropLast(4)) as NSString).utf8String!,
                         s.k, s.condition, 100 * s.topShare,
                         100 * s.peakShare, s.participation))
            fflush(stdout)
        }
    }
    exit(0)
}

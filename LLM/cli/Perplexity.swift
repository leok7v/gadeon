import Accelerate
import Foundation
import LLM

func runPerplexity(_ path: String, _ args: [String]) throws {
    let at = args.firstIndex(of: "--ppl")!
    let corpus = at + 1 < args.count ? args[at + 1] : ""
    let ctx = stripped(args, "--ppl-ctx") ?? 512
    let cap = stripped(args, "--ppl-chunks") ?? Int.max
    let text = (try? String(contentsOfFile: corpus, encoding: .utf8)) ?? ""
    if text.isEmpty {
        err("usage: gadeon-cli <model.gguf> --ppl <corpus.txt> "
            + "[--ppl-ctx 512] [--ppl-chunks N]\n")
        exit(2)
    }
    let chat = try MetalChat(ggufPath: path)
    let ids = chat.tokenizer.encode(text, addSpecial: true)
    let chunks = min(ids.count / ctx, cap)
    if chunks == 0 {
        err("[ppl] corpus is \(ids.count) tokens, need \(ctx)\n")
        exit(2)
    }
    let first = ctx / 2
    let serial = args.contains("--ppl-serial")
    let vocab = chat.tokenizer.vocabCount
    var total = 0.0
    var scored = 0
    let t0 = Date()
    for c in 0..<chunks {
        let base = c * ctx
        if serial {
            chat.engine.reset()
            for i in 0..<(ctx - 1) {
                let logits = chat.engine.step(ids[base + i])
                if i + 1 >= first {
                    total += cost(logits, logits.count, ids[base + i + 1])
                    scored += 1
                }
            }
        } else {
            chat.engine.chunkCost(Array(ids[base..<(base + ctx)]),
                                  from: first) { _, want, lp in
                total += cost(lp, vocab, want)
                scored += 1
            }
        }
        let done = Double(c + 1) / Double(chunks)
        err(String(format: "\r[ppl] chunk %d/%d  ppl %.4f  %.0f%%   ",
                   c + 1, chunks, exp(total / Double(scored)), done * 100))
    }
    let ppl = exp(total / Double(scored))
    err("\n")
    let name = (path as NSString).lastPathComponent
    print(name + String(repeating: " ", count: max(1, 40 - name.count))
          + String(format: "ppl %8.4f   (%d tokens, ctx %d, %.0fs)",
                   ppl, scored, ctx, Date().timeIntervalSince(t0)))
    exit(0)
}

private func stripped(_ args: [String], _ name: String) -> Int? {
    args.firstIndex(of: name).flatMap { i in
        i + 1 < args.count ? Int(args[i + 1]) : nil
    }
}

private func cost(_ logits: UnsafePointer<Float>, _ len: Int,
                  _ want: Int32) -> Double {
    var peak: Float = 0
    let n = vDSP_Length(len)
    vDSP_maxv(logits, 1, &peak, n)
    var shifted = [Float](repeating: 0, count: len)
    var minus = -peak
    vDSP_vsadd(logits, 1, &minus, &shifted, 1, n)
    var count = Int32(len)
    vvexpf(&shifted, shifted, &count)
    var sum: Float = 0
    vDSP_sve(shifted, 1, &sum, n)
    return Double(peak) + Double(log(sum)) - Double(logits[Int(want)])
}

import Accelerate
import Foundation
import LLM

private func arg(_ args: [String], _ name: String, _ n: Int) -> String? {
    args.firstIndex(of: name).flatMap { i in
        i + n < args.count ? args[i + n] : nil
    }
}

private func count(_ args: [String], _ name: String) -> Int? {
    arg(args, name, 1).flatMap { v in Int(v) }
}

private struct Replay: Codable {
    struct Item: Codable {
        let start: Int
        let ids: [Int32]
    }
    let model: String
    let items: [Item]
}

private func gemma(_ path: String) throws -> (GemmaChat, Gemma4MetalEngine) {
    if !Gemma4Model.isGemma4(path: path) {
        err("\(path) is not a gemma-4 file; replay is gemma-only\n")
        exit(2)
    }
    let chat = try GemmaChat(ggufPath: path)
    return (chat, try Gemma4MetalEngine(chat.model))
}

func runReplayMake(_ path: String, _ args: [String]) throws {
    let src = arg(args, "--replay-make", 1) ?? ""
    let out = arg(args, "--replay-make", 2) ?? ""
    let cap = count(args, "--replay-cap") ?? 256
    let text = (try? String(contentsOfFile: src, encoding: .utf8)) ?? ""
    if text.isEmpty || out.isEmpty {
        err("usage: gadeon-cli <teacher.gguf> --replay-make <prompts.json> "
            + "<out.json> [--replay-cap 256]\n")
        exit(2)
    }
    let (chat, engine) = try gemma(path)
    let asJSON = try? JSONDecoder().decode([String].self,
                                           from: Data(text.utf8))
    let prompts = asJSON ?? text.split(separator: "\n").map(String.init)
        .filter { line in !line.trimmingCharacters(
            in: .whitespaces).isEmpty }
    var items: [Replay.Item] = []
    let t0 = Date()
    for (n, ask) in prompts.enumerated() {
        var rendered = ask
        if !chat.chatTemplate.isEmpty {
            rendered = (try? renderPrompt(
                template: chat.chatTemplate,
                messages: [AgentMessage(role: "user", content: ask)],
                tools: [], addGenerationPrompt: true,
                enableThinking: false,
                bosToken: chat.bosToken)) ?? ask
        }
        var ids = chat.encode(rendered)
        let start = ids.count
        engine.reset()
        var next = engine.extend(ids)
        var made = 0
        while made < cap && !chat.eosIds.contains(next) {
            ids.append(next)
            next = engine.decodePlain(next)
            made += 1
        }
        items.append(Replay.Item(start: start, ids: ids))
        err(String(format: "\r[replay] %d/%d  %d tokens  %.0fs   ",
                   n + 1, prompts.count, made,
                   Date().timeIntervalSince(t0)))
    }
    err("\n")
    let name = (path as NSString).lastPathComponent
    let blob = try JSONEncoder().encode(Replay(model: name, items: items))
    try blob.write(to: URL(fileURLWithPath: out))
    let total = items.reduce(0) { s, i in s + i.ids.count - i.start }
    print("[replay] \(items.count) items, \(total) scored tokens -> \(out)")
    exit(0)
}

func runReplayScore(_ path: String, _ args: [String]) throws {
    let src = arg(args, "--replay", 1) ?? ""
    let dump = arg(args, "--kld-dump", 1)
    let against = arg(args, "--kld", 1)
    let blob = try? Data(contentsOf: URL(fileURLWithPath: src))
    if blob == nil {
        err("usage: gadeon-cli <model.gguf> --replay <replay.json> "
            + "[--kld-dump <top.bin> | --kld <top.bin>]\n")
        exit(2)
    }
    let replay = try JSONDecoder().decode(Replay.self, from: blob!)
    let (chat, engine) = try gemma(path)
    let vocab = chat.vocabCount
    var teacher = Data()
    if let against {
        teacher = try Data(contentsOf: URL(fileURLWithPath: against))
        let head = teacher.withUnsafeBytes { raw in
            (raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self),
             raw.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
        }
        if head.0 != TopK.magic || Int(head.1) != TopK.k {
            err("[replay] \(against) is not a top-\(TopK.k) dump\n")
            exit(2)
        }
    }
    var sink = Data()
    if dump != nil {
        sink.append(contentsOf: withUnsafeBytes(of: TopK.magic) { b in
            Array(b)
        })
        sink.append(contentsOf: withUnsafeBytes(of: UInt32(TopK.k)) { b in
            Array(b)
        })
    }
    var tally = Divergence()
    let t0 = Date()
    for (n, item) in replay.items.enumerated() {
        engine.chunkCost(item.ids, from: item.start) { _, want, lp in
            let top = rank(lp, vocab)
            if dump != nil { append(&sink, want, lp, top, vocab) }
            if against != nil { tally.add(lp, vocab, teacher, top) }
            tally.tokens += 1
            tally.cost += cost(lp, vocab, want)
        }
        err(String(format: "\r[replay] %d/%d  %.0fs   ", n + 1,
                   replay.items.count, Date().timeIntervalSince(t0)))
    }
    err("\n")
    if let dump {
        try sink.write(to: URL(fileURLWithPath: dump))
        err("[replay] wrote \(tally.tokens) positions -> \(dump)\n")
    }
    let name = (path as NSString).lastPathComponent
    print(name + String(repeating: " ", count: max(1, 34 - name.count))
          + String(format: "ppl %8.4f  over %d tokens from %@",
                   exp(tally.cost / Double(max(tally.tokens, 1))),
                   tally.tokens, replay.model)
          + (against == nil ? "" : tally.line))
    exit(0)
}

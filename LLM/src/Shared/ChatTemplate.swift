import Foundation

// Renders a conversation through a model's jinja chat_template. Bridges
// [AgentMessage] + [ToolSpec] into the data shape the template walks and drives
// the JinjaHost callbacks against an interned node tree. `renderPrompt` is the
// pure, engine-free core that ChatSession (delta render) and the offline gates
// both call.

// An interned tree of JSON-shaped nodes backing the six JinjaHost callbacks.
// Every container is stored once and referenced by an integer handle carried
// in a JinjaValue.node; a precomputed JSON string per node answers `| tojson`.
private struct NodeArena {
    var isObj: [Bool] = []
    var objs: [[(String, JinjaValue)]] = []
    var arrs: [[JinjaValue]] = []
    var json: [String] = []

    mutating func obj(_ entries: [(String, JinjaValue)],
                      _ jsonText: String) -> JinjaValue {
        let h = isObj.count
        isObj.append(true)
        objs.append(entries)
        arrs.append([])
        json.append(jsonText)
        return .node(handle: h, tag: AgentJinjaHost.tag)
    }

    mutating func arr(_ items: [JinjaValue],
                      _ jsonText: String) -> JinjaValue {
        let h = isObj.count
        isObj.append(false)
        objs.append([])
        arrs.append(items)
        json.append(jsonText)
        return .node(handle: h, tag: AgentJinjaHost.tag)
    }

    // A parameters node whose only role is to embed a pre-serialized JSON
    // schema verbatim under `| tojson`; its structural callbacks are unused.
    mutating func raw(_ jsonText: String) -> JinjaValue {
        let h = isObj.count
        isObj.append(true)
        objs.append([])
        arrs.append([])
        json.append(jsonText)
        return .node(handle: h, tag: AgentJinjaHost.tag)
    }
}

private func jsonEscape(_ s: String) -> String {
    var out = ""
    for ch in s.unicodeScalars {
        switch ch {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default: out.unicodeScalars.append(ch)
        }
    }
    return out
}

private func jsonScalar(_ v: JinjaValue, _ arena: NodeArena) -> String {
    switch v {
    case .str(let s): return "\"" + jsonEscape(s) + "\""
    case .int(let n): return String(n)
    case .bool(let b): return b ? "true" : "false"
    case .node(let h, _): return arena.json[h]
    default: return "null"
    }
}

private func buildObj(_ pairs: [(String, JinjaValue)],
                      _ arena: inout NodeArena) -> JinjaValue {
    var parts: [String] = []
    for (k, v) in pairs {
        parts.append("\"" + jsonEscape(k) + "\": " + jsonScalar(v, arena))
    }
    return arena.obj(pairs, "{" + parts.joined(separator: ", ") + "}")
}

private func buildArr(_ items: [JinjaValue],
                      _ arena: inout NodeArena) -> JinjaValue {
    var parts: [String] = []
    for v in items {
        parts.append(jsonScalar(v, arena))
    }
    return arena.arr(items, "[" + parts.joined(separator: ", ") + "]")
}

private func buildTool(_ t: ToolSpec,
                       _ arena: inout NodeArena) -> JinjaValue {
    let params = arena.raw(t.parametersJSON)
    let fn = buildObj([
        ("name", .str(t.name)),
        ("description", .str(t.description)),
        ("parameters", params),
    ], &arena)
    return buildObj([
        ("type", .str("function")),
        ("function", fn),
    ], &arena)
}

private func buildToolCall(_ c: AgentToolCall,
                           _ arena: inout NodeArena) -> JinjaValue {
    var args: [(String, JinjaValue)] = []
    for a in c.arguments {
        args.append((a.name, .str(a.value)))
    }
    let argObj = buildObj(args, &arena)
    return buildObj([("name", .str(c.name)),
                     ("arguments", argObj)], &arena)
}

private func buildContentParts(_ parts: [ContentPart],
                               _ arena: inout NodeArena) -> JinjaValue {
    var items: [JinjaValue] = []
    for part in parts {
        var item = JinjaValue.undefined
        switch part {
        case .image:
            item = buildObj([("type", .str("image"))], &arena)
        case .text(let text):
            item = buildObj([("type", .str("text")),
                             ("text", .str(text))], &arena)
        }
        items.append(item)
    }
    return buildArr(items, &arena)
}

private func buildMessage(_ m: AgentMessage,
                          _ arena: inout NodeArena) -> JinjaValue {
    var contentVal = JinjaValue.str(m.content)
    if let parts = m.contentParts {
        contentVal = buildContentParts(parts, &arena)
    }
    var pairs: [(String, JinjaValue)] = [
        ("role", .str(m.role)),
        ("content", contentVal),
    ]
    if let reasoning = m.reasoning {
        pairs.append(("reasoning_content", .str(reasoning)))
    }
    if !m.toolCalls.isEmpty {
        var calls: [JinjaValue] = []
        for c in m.toolCalls {
            calls.append(buildToolCall(c, &arena))
        }
        pairs.append(("tool_calls", buildArr(calls, &arena)))
    }
    return buildObj(pairs, &arena)
}

// Bridges [AgentMessage] + [ToolSpec] into the data shape a Qwen chat_template
// walks (messages[].role/.content/.tool_calls[].{name,arguments};
// tools[].function.{name,description,parameters}). The engine only ever sees
// opaque node handles and calls back through these six methods.
final class AgentJinjaHost: JinjaHost {
    static let tag = 0x5157      // reserved handle tag ('QW'); != dict/builtin
    private let arena: NodeArena
    let messagesValue: JinjaValue
    let toolsValue: JinjaValue

    init(messages: [AgentMessage], tools: [ToolSpec]) {
        var a = NodeArena()
        var msgs: [JinjaValue] = []
        for m in messages {
            msgs.append(buildMessage(m, &a))
        }
        var tls: [JinjaValue] = []
        for t in tools {
            tls.append(buildTool(t, &a))
        }
        self.messagesValue = buildArr(msgs, &a)
        self.toolsValue = buildArr(tls, &a)
        self.arena = a
    }

    private func handle(_ v: JinjaValue) -> Int? {
        var result: Int? = nil
        if case .node(let h, let t) = v, t == AgentJinjaHost.tag {
            result = h
        }
        return result
    }

    func get(_ obj: JinjaValue, _ name: String) -> JinjaValue {
        var result = JinjaValue.undefined
        if let h = handle(obj), arena.isObj[h] {
            for entry in arena.objs[h] where entry.0 == name {
                result = entry.1
            }
        }
        return result
    }

    func index(_ obj: JinjaValue, _ i: Int) -> JinjaValue {
        var result = JinjaValue.undefined
        if let h = handle(obj) {
            result = arena.isObj[h]
                ? .str(arena.objs[h][i].0) : arena.arrs[h][i]
        }
        return result
    }

    func len(_ obj: JinjaValue) -> Int {
        var result = 0
        if let h = handle(obj) {
            result = arena.isObj[h]
                ? arena.objs[h].count : arena.arrs[h].count
        }
        return result
    }

    func truthy(_ v: JinjaValue) -> Bool {
        var result = false
        if let h = handle(v) {
            result = arena.isObj[h]
                ? !arena.objs[h].isEmpty : !arena.arrs[h].isEmpty
        }
        return result
    }

    func test(_ v: JinjaValue, _ name: String) -> Bool {
        var result = false
        if let h = handle(v) {
            if name == "mapping" {
                result = arena.isObj[h]
            } else if name == "sequence" || name == "iterable" {
                result = !arena.isObj[h]
            }
        }
        return result
    }

    func method(_ obj: JinjaValue, _ name: String,
                _ arg: JinjaValue) -> JinjaValue {
        var result = JinjaValue.undefined
        if name == "tojson", let h = handle(obj) {
            result = .str(arena.json[h])
        }
        return result
    }
}

// Render the full conversation through `template`. Offline-testable: no
// engine, no tokenizer -- the pure core the turn loop and its tests both call.
public func renderPrompt(template: String, messages: [AgentMessage],
                         tools: [ToolSpec], addGenerationPrompt: Bool,
                         enableThinking: Bool,
                         addVisionId: Bool = false) throws -> String {
    let host = AgentJinjaHost(messages: messages, tools: tools)
    return try jinjaRender(template, host: host, vars: [
        ("messages", host.messagesValue),
        ("tools", host.toolsValue),
        ("add_generation_prompt", .bool(addGenerationPrompt)),
        ("enable_thinking", .bool(enableThinking)),
        ("add_vision_id", .bool(addVisionId)),
    ])
}

// Whether a model reasons at all, asked of its OWN template rather than of its
// name: render the generation prompt with thinking ON and see whether it
// leaves `<think>` open. Qwen3.5 opens it and lets the model close it; the
// dense Qwen3 line bakes a CLOSED empty `<think></think>` there and never
// mentions enable_thinking, so on those the flag, the lightbulb and the
// thinking budget are all inert and must not be offered.

public func templateSupportsThinking(_ template: String) -> Bool {
    let probe = [AgentMessage(role: "user", content: "x")]
    let closed = (try? renderPrompt(
        template: template, messages: probe, tools: [],
        addGenerationPrompt: false, enableThinking: true)) ?? ""
    let full = (try? renderPrompt(
        template: template, messages: probe, tools: [],
        addGenerationPrompt: true, enableThinking: true)) ?? ""
    var result = false
    if !closed.isEmpty, full.hasPrefix(closed) {
        let gen = full.dropFirst(closed.count)
        result = gen.contains("<think>") && !gen.contains("</think>")
    }
    return result
}

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

// A tool's parameter schema, usable BOTH ways. Qwen's template serializes it
// whole (`parameters | tojson`), so the node keeps the original JSON text
// verbatim -- re-serializing would reorder keys and shift the bytes of every
// rendered system prompt, which the precooked-prefix stamp hashes. Gemma's
// template instead WALKS the schema (`params['properties'] | dictsort`,
// `value['type']`, ...), so the node also needs real entries: a node that
// answers only tojson advertises every tool with no parameters at all.
private func buildParams(_ json: String,
                         _ arena: inout NodeArena) -> JinjaValue {
    var result = arena.raw(json)
    if let obj = try? JSONSerialization.jsonObject(with: Data(json.utf8)),
       let dict = obj as? [String: Any] {
        var pairs: [(String, JinjaValue)] = []
        for (k, v) in dict {
            pairs.append((k, buildJSON(v, &arena)))
        }
        result = arena.obj(pairs, json)
    }
    return result
}

// One decoded JSON value as a Jinja value. Containers keep a re-serialized
// json text (they are only ever reached structurally, never tojson'd whole).
private func buildJSON(_ v: Any, _ arena: inout NodeArena) -> JinjaValue {
    var result = JinjaValue.none
    if let s = v as? String {
        result = .str(s)
    } else if let n = v as? NSNumber {
        result = CFGetTypeID(n) == CFBooleanGetTypeID()
            ? .bool(n.boolValue) : .int(n.intValue)
    } else if let d = v as? [String: Any] {
        var pairs: [(String, JinjaValue)] = []
        for (k, item) in d {
            pairs.append((k, buildJSON(item, &arena)))
        }
        result = buildObj(pairs, &arena)
    } else if let a = v as? [Any] {
        var items: [JinjaValue] = []
        for item in a {
            items.append(buildJSON(item, &arena))
        }
        result = buildArr(items, &arena)
    }
    return result
}

private func buildTool(_ t: ToolSpec,
                       _ arena: inout NodeArena) -> JinjaValue {
    let params = buildParams(t.parametersJSON, &arena)
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

// A tool call in BOTH shapes at once. Qwen's template reads the flat
// `tool_call.name` / `.arguments`; gemma's reads the OpenAI nesting
// `tool_call['function']['name']` and RAISES when it is absent ("arguments
// must be a JSON object"). The two key sets do not collide, so one node
// serves both and neither template sees a shape it does not expect.
private func buildToolCall(_ c: AgentToolCall,
                           _ arena: inout NodeArena) -> JinjaValue {
    var args: [(String, JinjaValue)] = []
    for a in c.arguments {
        args.append((a.name, .str(a.value)))
    }
    let argObj = buildObj(args, &arena)
    let fn = buildObj([("name", .str(c.name)),
                       ("arguments", argObj)], &arena)
    return buildObj([("name", .str(c.name)),
                     ("arguments", argObj),
                     ("type", .str("function")),
                     ("function", fn)], &arena)
}

private func buildContentParts(_ parts: [ContentPart],
                               _ arena: inout NodeArena) -> JinjaValue {
    var items: [JinjaValue] = []
    for part in parts {
        var item = JinjaValue.undefined
        switch part {
        case .image:
            item = buildObj([("type", .str("image"))], &arena)
        case .audio:
            item = buildObj([("type", .str("audio"))], &arena)
        case .video:
            item = buildObj([("type", .str("video"))], &arena)
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
        pairs.append(("reasoning", .str(reasoning)))
    }
    if let name = m.name {
        pairs.append(("name", .str(name)))
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
// `bosToken` is the SPELLING of the tokenizer's begin-of-sequence piece, for
// a template that opens the conversation by naming it (`{{- bos_token -}}`,
// which gemma-4 does unconditionally). Left empty it renders as nothing, and
// the prompt silently loses a token the model expects at position 0 -- so a
// caller that has a tokenizer should always pass it. Qwen's template never
// mentions it and is unaffected either way.
public func renderPrompt(template: String, messages: [AgentMessage],
                         tools: [ToolSpec], addGenerationPrompt: Bool,
                         enableThinking: Bool,
                         addVisionId: Bool = false,
                         bosToken: String = "") throws -> String {
    let host = AgentJinjaHost(messages: messages, tools: tools)
    return try jinjaRender(template, host: host, vars: [
        ("messages", host.messagesValue),
        ("tools", host.toolsValue),
        ("add_generation_prompt", .bool(addGenerationPrompt)),
        ("enable_thinking", .bool(enableThinking)),
        ("add_vision_id", .bool(addVisionId)),
        ("bos_token", .str(bosToken)),
    ])
}

// Whether a model reasons at all, asked of its OWN template rather than of its
// name. Two shipped shapes answer through two different signals, so ChatWire
// checks both: a template that RENDERS a reasoning channel proves it by
// rendering one (gemma-4, whose generation prompt stays silent and lets the
// model open `<|channel>thought` itself), and one that leaves its reasoning
// marker OPEN in the generation prompt proves it that way (Qwen3.5). The dense
// Qwen3 line bakes a CLOSED empty `<think></think>` and never mentions
// enable_thinking, so it fails both and the flag, the lightbulb and the
// thinking budget must not be offered.
//
// Asking only the generation prompt -- what this did before -- reported gemma
// as non-reasoning and silently disabled all three.

public func templateSupportsThinking(_ template: String) -> Bool {
    ChatWire.derive(template).reasons(template: template)
}

import Foundation

// The chat markup a model's OWN template speaks, derived by rendering that
// template with sentinel values and diffing -- never matched against a model
// name, never looked up in a table of known dialects.
//
// WHY derived. ChatSession has to find the reasoning and tool-call boundaries
// in a raw decoded byte stream, and every lineage spells them differently:
// Qwen says <think> / <tool_call>, gemma-4 says <|channel>thought /
// <|tool_call>, and gemma's pairs are not even symmetric -- it closes with
// <channel|>, not </...>. A hardcoded table costs a row per model and still
// cannot resolve the collisions between them (gemma's <|channel> against
// gpt-oss's <|channel|> differ by one byte and mean different things). The
// template already IS the authority: it renders the exact bytes the model was
// trained to emit. So rendering an assistant message that CARRIES reasoning,
// against the same message carrying none, leaves those bytes as the
// difference -- and the same subtraction reads the tool-call wire.
//
// This is the same idiom seedDelta and systemRenders already use (render with
// and without add_generation_prompt, take the suffix), generalized from the
// generation prompt to every marker the decode loop matches on.
//
// WHY a fallback. A template that renders nothing for a feature derives
// nothing: a template that strips past-turn reasoning rather than
// re-emitting it never shows its channel. Silence leaves the Qwen wire
// standing, since that is the wire such a template is overwhelmingly
// likely to speak.
struct ChatWire: Sendable {
    // The reasoning channel. Gemma keeps its channel NAME in the open marker
    // (<|channel>thought) because the same tag opens other channels; the tag
    // alone would match all of them.
    let reasoningOpen: String
    let reasoningClose: String
    // The tool-call block. Here the opener is cut back to its tag, because
    // what follows it is the call BODY and varies per call (gemma renders
    // "<|tool_call>call:NAME{", Qwen "<tool_call>\n<function=NAME>"). The tag
    // is the part every call shares, so it is the anchor.
    let toolCallOpen: String
    let toolCallClose: String
    // Turn framing, kept only to exempt its specials from the sampler's
    // repetition penalties; the turn loop never matches on it.
    let turnOpen: String
    let turnClose: String
    // Whether the reasoning markers came from the template rather than from
    // the default. Load-bearing: it decides how a generation prompt that says
    // NOTHING about reasoning is read. See startsInReasoning.
    let derivedReasoning: Bool

    static let defaultReasoningOpen = "<think>"
    static let defaultReasoningClose = "</think>"
    static let defaultToolCallOpen = "<tool_call>"
    static let defaultToolCallClose = "</tool_call>"

    // ---- derivation -----------------------------------------------------

    // Sentinels are ASCII and shaped to survive a `| trim` and any
    // strip-thinking macro, while never occurring in a real template.
    private static let sBody = "ZqBodyZq"
    private static let sReason = "ZqReasonZq"
    private static let sTool = "ZqToolZq"
    private static let sArgKey = "ZqArgKZq"
    private static let sArgVal = "ZqArgVZq"
    private static let sUser = "ZqUserZq"

    static func derive(_ template: String) -> ChatWire {
        let reason = deriveReasoning(template)
        let call = deriveToolCall(template)
        let turn = deriveTurn(template)
        return ChatWire(
            reasoningOpen: reason?.open ?? defaultReasoningOpen,
            reasoningClose: reason?.close ?? defaultReasoningClose,
            toolCallOpen: call?.open ?? defaultToolCallOpen,
            toolCallClose: call?.close ?? defaultToolCallClose,
            turnOpen: turn?.open ?? "",
            turnClose: turn?.close ?? "",
            derivedReasoning: reason != nil)
    }

    // Render an assistant turn that carries reasoning against one that does
    // not; whatever wraps the sentinel in the difference IS the channel.
    // The assistant must sit AFTER the last user message: gemma gates
    // past-turn reasoning on exactly that (it strips reasoning from turns the
    // conversation has already moved past).
    private static func deriveReasoning(
        _ t: String
    ) -> (open: String, close: String)? {
        let user = AgentMessage(role: "user", content: sUser)
        let bare = AgentMessage(role: "assistant", content: sBody)
        let with = AgentMessage(role: "assistant", content: sBody,
                                reasoning: sReason)
        let a = render(t, [user, with], think: true)
        let b = render(t, [user, bare], think: true)
        // A marker that trims to NOTHING is a failed derivation wearing a
        // success: the real Qwen3.5 template wraps reasoning in whitespace
        // alone, so the subtraction yields "" for both ends -- and an empty
        // pattern matches at every offset, so the leading-think probe fires on
        // the first token of every answer and the close is "found" at zero.
        // Measured: with reasoning OFF the decode loop believed it was inside
        // the think region for a plain answer. Report it as underived so the
        // default Qwen wire stands, which for this template is the right wire.
        return span(a, b, sReason)
            .map { pair in
                (trimWhitespace(pair.open), trimWhitespace(pair.close))
            }
            .flatMap { pair in
                pair.open.isEmpty || pair.close.isEmpty ? nil : pair
            }
    }

    // Same subtraction over a tool call. The opener keeps only its tag (up to
    // and including the first '>'), the closer only its trailing tag (from the
    // last '<'), so the call body between them is not baked into either.
    private static func deriveToolCall(
        _ t: String
    ) -> (open: String, close: String)? {
        let user = AgentMessage(role: "user", content: sUser)
        let bare = AgentMessage(role: "assistant", content: sBody)
        let call = AgentToolCall(name: sTool,
                                 arguments: [ToolArg(name: sArgKey,
                                                     value: sArgVal)])
        let with = AgentMessage(role: "assistant", content: sBody,
                                toolCalls: [call])
        let a = render(t, [user, with], think: false)
        let b = render(t, [user, bare], think: false)
        return span(a, b, sTool).map { pair in
            (headTag(pair.open), tailTag(pair.close))
        }
    }

    // Turn framing needs no diff: a lone user message renders as the framing
    // wrapped around its content, so splitting that render at the sentinel is
    // the answer. Diffing against an EMPTY-content message would put the
    // framing in the common part and leave nothing.
    private static func deriveTurn(
        _ t: String
    ) -> (open: String, close: String)? {
        let a = render(t, [AgentMessage(role: "user", content: sUser)],
                       think: false)
        var result: (open: String, close: String)? = nil
        if let r = a.range(of: sUser) {
            result = (String(a[..<r.lowerBound]),
                      String(a[r.upperBound...]))
        }
        return result
    }

    private static func render(_ t: String, _ msgs: [AgentMessage],
                               think: Bool) -> String {
        (try? renderPrompt(template: t, messages: msgs, tools: [],
                           addGenerationPrompt: false,
                           enableThinking: think)) ?? ""
    }

    // The bytes `a` inserted around `sentinel` relative to `b`: everything
    // between the two renders' common prefix and common suffix, split at the
    // sentinel. Nil when the sentinel is absent (the template rendered the
    // feature away) or nothing was inserted.
    private static func span(_ a: String, _ b: String,
                             _ sentinel: String) -> (open: String,
                                                     close: String)? {
        let x = Array(a.utf8)
        let y = Array(b.utf8)
        let s = Array(sentinel.utf8)
        var pre = 0
        while pre < x.count && pre < y.count && x[pre] == y[pre] { pre += 1 }
        var suf = 0
        while suf < x.count - pre && suf < y.count - pre
            && x[x.count - 1 - suf] == y[y.count - 1 - suf] { suf += 1 }
        // Normalize to the LEFTMOST diff. A byte shared by the end of the
        // common prefix and the start of the insertion is ambiguous, and the
        // naive walk always assigns it to the prefix -- which decapitates the
        // marker exactly when it matters, since "<tool_call>" and the
        // "<|im_end|>" that follows the block both start with '<'. Rotating
        // the window left while the two ends agree recovers the tag.
        let len = x.count - suf - pre
        while pre > 0 && len > 0 && x[pre - 1] == x[pre - 1 + len] {
            pre -= 1
            suf += 1
        }
        let mid = Array(x[pre ..< (x.count - suf)])
        var result: (open: String, close: String)? = nil
        if let at = index(mid, s) {
            result = (String(decoding: mid[..<at], as: UTF8.self),
                      String(decoding: mid[(at + s.count)...], as: UTF8.self))
        }
        return result
    }

    private static func index(_ b: [UInt8], _ pat: [UInt8]) -> Int? {
        var result: Int? = nil
        var i = 0
        let last = b.count - pat.count
        while result == nil && i <= last {
            var j = 0
            while j < pat.count && b[i + j] == pat[j] { j += 1 }
            if j == pat.count { result = i }
            i += 1
        }
        return result
    }

    private static func trimWhitespace(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // The leading tag of an opener: up to and including its first '>'. The
    // whole opener when it carries no tag at all.
    private static func headTag(_ s: String) -> String {
        let t = trimWhitespace(s)
        var result = t
        if let gt = t.firstIndex(of: ">") {
            result = String(t[...gt])
        }
        return result
    }

    // The trailing tag of a closer: from its last '<' on.
    private static func tailTag(_ s: String) -> String {
        let t = trimWhitespace(s)
        var result = t
        if let lt = t.lastIndex(of: "<") {
            result = String(t[lt...])
        }
        return result
    }

    // ---- what the turn loop asks ----------------------------------------

    // Whether this model reasons at all, asked of its own template. Two
    // independent signals, because the two shipped shapes answer differently:
    // a template that RENDERS a reasoning channel proves it (gemma), and one
    // that leaves its reasoning marker open in the generation prompt proves it
    // too (Qwen3.5) even when it strips past-turn reasoning and so derives
    // nothing. The dense Qwen3 line bakes a CLOSED empty block and never
    // mentions enable_thinking, so it fails both and must not be offered a
    // lightbulb.
    func reasons(template: String) -> Bool {
        var result = derivedReasoning
        if !result {
            let probe = [AgentMessage(role: "user", content: "x")]
            let closed = (try? renderPrompt(
                template: template, messages: probe, tools: [],
                addGenerationPrompt: false, enableThinking: true)) ?? ""
            let full = (try? renderPrompt(
                template: template, messages: probe, tools: [],
                addGenerationPrompt: true, enableThinking: true)) ?? ""
            if !closed.isEmpty, full.hasPrefix(closed) {
                let gen = String(full.dropFirst(closed.count))
                result = opensReasoning(gen)
            }
        }
        return result
    }

    // A generation prompt OPENS reasoning when it emits the open marker and
    // does not close it afterwards.
    func opensReasoning(_ gen: String) -> Bool {
        var result = false
        if let at = gen.range(of: reasoningOpen, options: .backwards) {
            result = gen.range(of: reasoningClose,
                               range: at.upperBound ..< gen.endIndex) == nil
        }
        return result
    }

    func closesReasoning(_ gen: String) -> Bool {
        gen.contains(reasoningClose)
    }

    // Whether decoding begins INSIDE the reasoning region, which is the
    // template's call and not the flag's. An explicit close or open in the
    // generation prompt decides it. SILENCE is the interesting case: on a
    // DERIVED wire it means the model opens its own channel later (gemma
    // emits a bare "<|turn>model\n" and opens <|channel>thought itself), so
    // decoding starts in content and the leading-marker scan picks the
    // channel up; on the DEFAULT wire it means the template never mentions
    // reasoning at all, where the historical reading -- thinking on implies
    // the region is open -- is what every Qwen-shaped template relies on.
    func startsInReasoning(genPrompt: String, enabled: Bool) -> Bool {
        var result = false
        if enabled {
            if closesReasoning(genPrompt) {
                result = false
            } else if opensReasoning(genPrompt) {
                result = true
            } else {
                result = !derivedReasoning
            }
        }
        return result
    }

    // The wire specials to exempt from the sampler's repetition penalties: a
    // tool round re-emits them verbatim, and penalizing them rots the markup
    // round over round. Each derived marker contributes itself plus any
    // <...>-shaped run inside it, since a tokenizer usually has the bare tag
    // as one token and the marker-with-name as several.
    var penaltyExemptCandidates: [String] {
        var out: [String] = []
        for marker in [reasoningOpen, reasoningClose, toolCallOpen,
                       toolCallClose, turnOpen, turnClose]
        where !marker.isEmpty {
            out.append(marker)
            out.append(contentsOf: ChatWire.tags(marker))
        }
        return out
    }

    // Every maximal "<...>" run in `s`.
    private static func tags(_ s: String) -> [String] {
        var out: [String] = []
        var rest = Substring(s)
        while let lt = rest.firstIndex(of: "<") {
            let tail = rest[lt...]
            if let gt = tail.firstIndex(of: ">") {
                out.append(String(tail[...gt]))
                rest = tail[tail.index(after: gt)...]
            } else {
                rest = rest[rest.endIndex...]
            }
        }
        return out
    }
}

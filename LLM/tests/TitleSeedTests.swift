import Foundation
import Testing
@testable import LLM

// What a title turn appends to the generation prompt, per template.
//
// The seed exists because naming a conversation cost 255 reasoning tokens
// against 1 of content on gemma-4, and on a capped turn produced no title at
// all. `enableThinking = false` cannot reach that: the flag rides the jinja
// template and gemma opens the channel itself as its first decoded output,
// so the KV is the only place to answer it.
//
// Three templates hand over three different states, and only the gemma
// branch had ever executed -- the other two were written from the documented
// shapes and never run. A wrong branch is SILENT: a doubled or malformed
// marker goes into the KV and surfaces only as degraded output nobody
// attributes to it.
//
// No weights and no engine here. The seed is chosen from the rendered
// generation prompt by two string predicates, so the branch is decidable
// from the template alone -- which is also why it is per-TEMPLATE and not
// per-engine, whatever backend runs afterwards.
struct TitleSeedTests {
    // Two components up from LLM/tests/<file> is LLM.
    private static func fixture(_ name: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures/" + name)
        return try String(contentsOf: url, encoding: .utf8)
    }

    // The generation prompt as ChatSession sees it: render the same turn with
    // and without it and take the suffix. Mirrors the production subtraction
    // rather than guessing at the markup, which is the template's business.
    private static func genSuffix(_ template: String) throws -> String {
        let msgs = [AgentMessage(role: "user", content: "x")]
        let closed = try renderPrompt(
            template: template, messages: msgs, tools: [],
            addGenerationPrompt: false, enableThinking: true, bosToken: "")
        let full = try renderPrompt(
            template: template, messages: msgs, tools: [],
            addGenerationPrompt: true, enableThinking: true, bosToken: "")
        return full.hasPrefix(closed)
            ? String(full.dropFirst(closed.count)) : ""
    }

    // gemma-4 says nothing about reasoning in its generation prompt, so the
    // seed has to supply the whole block.
    @Test func gemmaGetsBothMarkers() throws {
        let template = try TitleSeedTests.fixture("gemma4-chat-template.jinja")
        let wire = ChatWire.derive(template)
        let gen = try TitleSeedTests.genSuffix(template)
        #expect(!wire.opensReasoning(gen), "gen: \(gen.debugDescription)")
        #expect(!wire.closesReasoning(gen), "gen: \(gen.debugDescription)")
        #expect(ChatSession.titleSeed(gen, wire)
                == wire.reasoningOpen + wire.reasoningClose)
    }

    // Qwen3.5 OPENS the marker and leaves it open, so the seed closes it and
    // must not open a second one. This is the branch that shipped unproven.
    @Test func qwenGetsOnlyTheClosingMarker() throws {
        let template = try TitleSeedTests.fixture("qwen35-chat-template.jinja")
        let wire = ChatWire.derive(template)
        let gen = try TitleSeedTests.genSuffix(template)
        #expect(wire.opensReasoning(gen), "gen: \(gen.debugDescription)")
        #expect(!wire.closesReasoning(gen), "gen: \(gen.debugDescription)")
        #expect(ChatSession.titleSeed(gen, wire) == wire.reasoningClose)
    }

    // A template that already bakes the closed block wants nothing, and would
    // be corrupted by a second one. Built from the wire's OWN markers rather
    // than a literal, so it holds for whatever a template spells them.
    @Test func anAlreadyClosedPromptIsLeftAlone() throws {
        let template = try TitleSeedTests.fixture("qwen35-chat-template.jinja")
        let wire = ChatWire.derive(template)
        let baked = wire.reasoningOpen + wire.reasoningClose
        #expect(ChatSession.titleSeed(baked, wire) == "")
    }

    // The seed is idempotent: applying it and asking again yields nothing.
    // A turn that seeded twice would lay two blocks into the KV.
    @Test func seedingTwiceAddsNothing() throws {
        let template = try TitleSeedTests.fixture("qwen35-chat-template.jinja")
        let wire = ChatWire.derive(template)
        let gen = try TitleSeedTests.genSuffix(template)
        let once = gen + ChatSession.titleSeed(gen, wire)
        #expect(ChatSession.titleSeed(once, wire) == "")
    }
}

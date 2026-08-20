import Foundation
import Testing
@testable import LLM

// What a title turn appends to the generation prompt, per template.
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

    @Test func gemmaGetsNothing() throws {
        let template = try TitleSeedTests.fixture("gemma4-chat-template.jinja")
        let wire = ChatWire.derive(template)
        let gen = try TitleSeedTests.genSuffix(template)
        #expect(!wire.opensReasoning(gen), "gen: \(gen.debugDescription)")
        #expect(!wire.closesReasoning(gen), "gen: \(gen.debugDescription)")
        #expect(ChatSession.titleSeed(gen, wire) == "")
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

    static let toolCallRaw = "<tool_call>\n<function=calculator>\n"
        + "<parameter=expression>\n1.10 - 1.00\n</parameter>\n</function>\n"
        + "</tool_call>"

    @Test func aToolCallNeverBecomesATitle() {
        #expect(ChatSession.cleanTitle(TitleSeedTests.toolCallRaw) == "")
    }

    @Test func aToolCallNeverBecomesAFollowup() {
        #expect(ChatSession.cleanFollowup(TitleSeedTests.toolCallRaw) == "")
    }

    @Test func reasoningMarkersAreStripped() {
        #expect(ChatSession.cleanTitle("<think>hm</think>Bat and ball")
                == "hmBat and ball")
        #expect(ChatSession.cleanTitle("<|im_end|>") == "")
    }

    @Test func aTitleWithoutLettersIsRejected() {
        #expect(ChatSession.cleanTitle("1.10 - 1.00") == "")
        #expect(ChatSession.cleanTitle("42") == "")
    }

    @Test func ordinaryTitlesSurvive() {
        #expect(ChatSession.cleanTitle("Bat and ball puzzle")
                == "Bat and ball puzzle")
        #expect(ChatSession.cleanTitle("Title: Riddle Solution")
                == "Riddle Solution")
    }

    @Test func anUnmatchedAngleIsNotATag() {
        #expect(ChatSession.withoutMarkup("5 < 6 and more") == "5 < 6 and more")
    }

    @Test func aRealFollowupSurvives() {
        let q = "How did you get 0.05 from the equation?"
        #expect(ChatSession.cleanFollowup(q) == q)
    }
}

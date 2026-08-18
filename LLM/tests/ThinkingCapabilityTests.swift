import XCTest
@testable import LLM

// Whether a model reasons is asked of its own chat template, never of its
// name (see feedback-no-hardcoded-model-constants). The two real shapes:
// Qwen3.5 opens `<think>` in the generation prompt when enable_thinking is
// true and lets the model close it; the dense Qwen3 line bakes a CLOSED empty
// `<think></think>` there and never mentions enable_thinking at all, so
// nothing the user does can make it reason. The app gates its lightbulb, the
// Settings switch and the thinking budget on this, so a wrong answer here
// shows up as a control that lies.
final class ThinkingCapabilityTests: XCTestCase {

    // The Qwen3.5 shape, reduced to the branch that matters.
    private let reasoning = """
        {% for message in messages %}<|im_start|>{{ message.role }}
        {{ message.content }}<|im_end|>
        {% endfor %}{%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\\n' }}
        {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\\n' }}
        {%- else %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
        {%- endif %}
        {%- endif %}
        """

    // The dense Qwen3 shape: the think block is always closed, and
    // enable_thinking never appears.
    private let direct = """
        {% for message in messages %}<|im_start|>{{ message.role }}
        {{ message.content }}<|im_end|>
        {% endfor %}{%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}
        {%- endif %}
        """

    private let effortful = """
        {%- if enable_thinking is undefined or enable_thinking is true %}
        {%- set e = reasoning_effort|default('xhigh') %}
        {%- if e == 'high' %}{%- set e = 'xhigh' %}{%- endif %}
        {%- if e not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ e) }}
        {%- endif %}
        {{- 'effort=' ~ e }}
        {%- endif %}
        {% for message in messages %}{{ message.content }}{% endfor %}
        """

    private let probe = [AgentMessage(role: "user", content: "x")]

    private func render(_ level: String?) throws -> String {
        try renderPrompt(template: effortful, messages: probe, tools: [],
                         addGenerationPrompt: true, enableThinking: true,
                         reasoningEffort: level)
    }

    func testEffortTemplateReportsSupported() {
        XCTAssertTrue(templateTakesReasoningEffort(effortful))
    }

    func testTemplateIgnoringEffortReportsUnsupported() {
        XCTAssertFalse(templateTakesReasoningEffort(reasoning))
    }

    func testAbsentEffortTakesTheTemplateDefault() throws {
        XCTAssertTrue(try render(nil).contains("effort=xhigh"))
    }

    func testEachLevelReachesTheTemplate() throws {
        XCTAssertTrue(try render("low").contains("effort=low"))
        XCTAssertTrue(try render("medium").contains("effort=medium"))
        XCTAssertTrue(try render("high").contains("effort=xhigh"))
    }

    func testReasoningTemplateReportsThinkingSupported() {
        XCTAssertTrue(templateSupportsThinking(reasoning))
    }

    func testBakedClosedThinkTemplateReportsUnsupported() {
        XCTAssertFalse(templateSupportsThinking(direct))
    }

    // A template with no generation prompt at all cannot reason either, and
    // must not crash the probe.
    func testTemplateWithoutGenerationPromptIsUnsupported() {
        let bare = "{% for message in messages %}{{ message.content }}"
            + "{% endfor %}"
        XCTAssertFalse(templateSupportsThinking(bare))
    }

    // The SHIPPED 0.8B template, when present, must classify as reasoning --
    // the inline shapes above are a reduction, this is the real bytes.
    func testShippedQwen35TemplateIsReasoning() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let base = root.appendingPathComponent("models/Qwen3.5-0.8B")
        let fm = FileManager.default
        var found: String? = nil
        let flat = base.appendingPathComponent("chat_template.jinja")
        if fm.fileExists(atPath: flat.path) {
            found = try? String(contentsOf: flat, encoding: .utf8)
        } else {
            for sha in (try? fm.contentsOfDirectory(atPath: base.path)) ?? [] {
                let url = base.appendingPathComponent(sha)
                    .appendingPathComponent("chat_template.jinja")
                if found == nil, fm.fileExists(atPath: url.path) {
                    found = try? String(contentsOf: url, encoding: .utf8)
                }
            }
        }
        try XCTSkipIf(found == nil, "no Qwen3.5-0.8B chat template on disk")
        XCTAssertTrue(templateSupportsThinking(found!))
    }
}

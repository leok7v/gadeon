import CoreML
import Foundation

public struct AneChat: Sendable {
    public let tokenizer: Tokenizer
    public let engine: Engine
    public let modelName: String        // model dir name, for the per-turn log
    // The set's Jinja chat_template for ChatSession rendering; falls back to a
    // minimal empty-think ChatML when the set ships none.
    public let chatTemplate: String

    // Minimal ChatML with the Qwen3.5 empty-think prefix, when a set ships no
    // chat_template.jinja (reasoning-effort none; the model answers directly).
    static let fallbackChatML = """
        {% for message in messages %}<|im_start|>{{ message.role }}
        {{ message.content }}<|im_end|>
        {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
        <think>

        </think>

        {% endif %}
        """

    // ANE-only: every set loads directly on the Neural Engine. The app gates
    // chat on the full compile finishing.
    public init(modelsDir: URL, cpuOnly: Bool = false,
                onCompiledLoC: (@Sendable (Int) -> Void)? = nil) throws {
        self.tokenizer = try Tokenizer(modelsDir: modelsDir)
        self.engine = try Engine(modelsDir: modelsDir, cpuOnly: cpuOnly,
                                 onCompiledLoC: onCompiledLoC)
        self.modelName = modelsDir.lastPathComponent
        let tmpl = modelsDir.appendingPathComponent("chat_template.jinja")
        self.chatTemplate = (try? String(contentsOf: tmpl, encoding: .utf8))
            ?? AneChat.fallbackChatML
    }

    // Load the common heavy set (the 6 S=512 batched-prefill programs +
    // batched embedding, each a slow first-launch compile) OFF the actor, then
    // swap in, so a caller is decode-ready first and upgrades to fast bounded
    // prefill. The rare carry (>512) set is NOT loaded here; it compiles on
    // demand.
    public func loadHeavy() async throws {
        let e = engine
        let units = e.computeUnits
        // CPU skips the batched embedding too: it only feeds the batched
        // prefill, which CPU does not use (token-by-token instead), so loading
        // it is wasted work / memory on a low-memory device.
        if units != .cpuOnly {
            let heavy = try await Task.detached(priority: .utility) {
                try e.prepareHeavy(units)
            }.value
            await e.applyHeavy(heavy)
        }
    }

    // Block-load the batched prefill set (the mf "prefill" trunk + its tile)
    // OFF the actor, once at launch before chat, so the first prompt gets fast
    // ANE prefill not token-by-token ingest. A device that cannot build it
    // throws; the caller degrades to ingest (carryProgs stays nil).
    public func loadCarry() async throws {
        let e = engine
        let units = e.computeUnits
        let carry = try await Task.detached(priority: .utility) {
            try e.prepareCarry(units)
        }.value
        await e.applyCarry(carry)
    }

    // Autodetect + load the MTP self-speculative drafter IF the set ships it
    // (mtp_front/back + the verify trunk). A set without them is a silent no-op,
    // so this is safe to call at launch for every set. Gate spec decode on
    // engine.mtpReady().
    public func loadMTP() async {
        await engine.loadMTP()
    }

}

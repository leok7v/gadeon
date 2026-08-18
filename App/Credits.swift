import Foundation

struct Credit: Identifiable {
    let name: String
    let author: String
    let terms: String
    let source: String
    let changed: String
    var id: String { name }
    var link: URL? { URL(string: source) }
}

enum Credits {

    // Apache-2.0 5(b) requires `changed` to state the modification on every
    // model below.

    static let app = Credit(
        name: "Gadeon",
        author: "leok7v",
        terms: "GPL-3.0",
        source: "https://github.com/leok7v/gadeon",
        changed: "Runs every model below on this device. Nothing you type "
            + "leaves it except where a tool you switched on says otherwise.")

    static let all: [Credit] = [
        Credit(name: "Gemma 4",
               author: "Google DeepMind",
               terms: "Apache-2.0",
               source: "https://huggingface.co/google/gemma-4-12B-it",
               changed: "Repacked into a single GGUF. The quantization-aware "
                   + "training left 4-bit codes behind in half precision, and "
                   + "those codes are recovered as they were trained rather "
                   + "than quantized again."),
        Credit(name: "Qwen3.5",
               author: "Qwen, Alibaba",
               terms: "Apache-2.0",
               source: "https://huggingface.co/Qwen/Qwen3.5-0.8B",
               changed: "Converted to Core ML so it runs on the Neural "
                   + "Engine."),
        Credit(name: "Qwen3.8",
               author: "Qwen, Alibaba",
               terms: "Apache-2.0",
               source: "https://huggingface.co/Qwen/Qwen3.8-27B",
               changed: "Converted to Core ML so it runs on the Neural "
                   + "Engine, with its own multi-token-prediction drafter "
                   + "folded in."),
        Credit(name: "QwenPaw-Flash",
               author: "AgentScope",
               terms: "Apache-2.0",
               source: "https://huggingface.co/agentscope-ai/QwenPaw-Flash-9B",
               changed: "Converted to Core ML for the Neural Engine, with "
                   + "multi-token prediction folded in."),
        Credit(name: "Ternary Bonsai",
               author: "PrismML",
               terms: "Apache-2.0",
               source: "https://huggingface.co/prism-ml/"
                   + "Ternary-Bonsai-27B-gguf",
               changed: "Repacked into the ternary block types this app's GPU "
                   + "engine reads."),
        Credit(name: "KittenTTS",
               author: "KittenML",
               terms: "Apache-2.0",
               source: "https://huggingface.co/KittenML/kitten-tts-nano-0.2",
               changed: "The voice you hear. Reimplemented in Swift by way of "
                   + "leok7v/tts.cli, so it synthesises on device with no "
                   + "network and no system speech service."),
        Credit(name: "all-MiniLM-L6-v2",
               author: "Sentence-Transformers",
               terms: "Apache-2.0",
               source: "https://huggingface.co/sentence-transformers/"
                   + "all-MiniLM-L6-v2",
               changed: "Quantized to GGUF. It matches your question to a "
                   + "Wikipedia article on this device, which is what keeps "
                   + "the question itself off the network."),
        Credit(name: "Wikipedia",
               author: "Wikimedia Foundation and contributors",
               terms: "CC BY-SA 4.0",
               source: "https://simple.wikipedia.org",
               changed: "Article titles are indexed on device; article text "
                   + "is fetched only for the article the model chose."),
    ]

}

// A fixed empty-think ChatML prompt. Not on the app path (that uses jinja
// render); the tokenizer gate (TokenizerGateTests) encodes it and diffs the
// ids against a Python-captured reference, so it must stay byte-stable.
enum ChatML {
    static func prompt(system: String?, user: String) -> String {
        var s = ""
        if let system {
            s += "<|im_start|>system\n\(system)<|im_end|>\n"
        }
        s += "<|im_start|>user\n\(user)<|im_end|>\n"
        // Qwen3.5 empty-think prefix: suppresses thinking, direct answer
        // (matches the validated Python chat_decode.py template).
        s += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return s
    }
}

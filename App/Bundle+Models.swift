import Foundation
import LLM

var installedGB: Int {
    Int((ProcessInfo.processInfo.physicalMemory + (1 << 29)) >> 30)
}

#if DEBUG
let debugBuild = true
#else
let debugBuild = false
#endif

extension Bundle {

    private static let store: URL = {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask, appropriateFor: nil,
                                   create: true))
            ?? fm.temporaryDirectory
        return support.appendingPathComponent("models", isDirectory: true)
    }()

    static func modelStore() -> URL { store }

    static var appName: String {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName")
            as? String)
            ?? (Bundle.main.object(forInfoDictionaryKey: "CFBundleName")
                as? String)
            ?? "Gadeon"
    }

}

enum Models {

    static var all: [String] {
        let gb = installedGB
        var list: [String] = []
        if isOS {
            if gb >= 8 {
                list.append("gemma-4-E4B")
            } else {
                list.append("gemma-4-E2B")
            }
            list.append("Ternary-Bonsai-1.7B")
        } else {
            if gb >= 8 { list.append("Qwen3.5-4B") }
            if gb >= 16 { list.append("Qwen3.5-9B") }
            if gb >= 16 { list.append("Qwen3.8-27B-IQ1_S") }
            if gb >= 24 { list.append("Qwen3.8-27B-Q4_K_S") }
            if gb >= 24 { list.append("Ternary-Bonsai-27B") }
            list.append("Ternary-Bonsai-1.7B")
            list.append("gemma-4-E2B")
            list.append("gemma-4-E4B")
            if gb >= 16 { list.append("gemma-4-12B") }
        }
        return list
    }

    static var supported: Bool { !all.isEmpty }

    static let fallback = "Ternary-Bonsai-1.7B"

    static var start: String {
        let list = all
        var out = list.contains(fallback) ? fallback : (list.first ?? fallback)
        if list.contains(e2b) { out = e2b }
        if list.contains(e4b) { out = e4b }
        return out
    }

    private static let e2b = "gemma-4-E2B"
    private static let e4b = "gemma-4-E4B"

    static func display(_ name: String) -> String {
        var out = name
        switch name {
            case "Qwen3.5-4B": out = "Qwen3.5 4B"
            case "Qwen3.5-9B": out = "Qwen3.5 9B"
            case "Qwen3.8-27B-IQ1_S": out = "Qwen3.8 27B 1-bit"
            case "Qwen3.8-27B-Q4_K_S": out = "Qwen3.8 27B 4-bit"
            case "Ternary-Bonsai-27B": out = "Bonsai 27B"
            case "Ternary-Bonsai-1.7B": out = "Bonsai 1.7B"
            case "gemma-4-E2B": out = "Gemma E2B"
            case "gemma-4-E4B": out = "Gemma E4B"
            case "gemma-4-12B": out = "Gemma 12B"
            default: out = name
        }
        return out
    }

}

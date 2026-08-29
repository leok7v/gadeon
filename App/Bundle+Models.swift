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

    private static func qwen38(_ gb: Int) -> String? {
        let out: String?
        switch gb {
        case ..<9: out = nil
        case ..<16: out = "Qwen3.8-27B-IQ1_S"
        case ..<24: out = "Qwen3.8-27B-IQ2_XXS"
        case ..<32: out = "Qwen3.8-27B-IQ3_XXS"
        case 32: out = "Qwen3.8-27B-IQ4_XS"
        default: out = "Qwen3.8-27B-Q4_K_S"
        }
        return out
    }

    static var every: [String] {
        isOS
            ? ["gemma-4-E2B", "gemma-4-E4B", "Ternary-Bonsai-1.7B"]
            : ["Qwen3.5-4B", "Qwen3.5-9B",
               "Qwen3.8-27B-IQ1_S", "Qwen3.8-27B-IQ2_XXS",
               "Qwen3.8-27B-IQ3_XXS", "Qwen3.8-27B-IQ4_XS",
               "Qwen3.8-27B-Q4_K_S",
               "Ternary-Bonsai-27B", "Ternary-Bonsai-1.7B",
               "gemma-4-E2B", "gemma-4-E4B", "gemma-4-12B"]
    }

    static var downloaded: Set<String> {
        var out: Set<String> = []
        for name in every {
            if let dir = ModelCatalog.localSet(name, in: Bundle.modelStore()),
               ModelCatalog.isComplete(dir) {
                out.insert(name)
            }
        }
        return out
    }

    static var all: [String] {
        let gb = installedGB
        var band: Set<String> = []
        if isOS {
            band.insert(gb >= 8 ? "gemma-4-E4B" : "gemma-4-E2B")
            band.insert("Ternary-Bonsai-1.7B")
        } else {
            if gb >= 8 { band.insert("Qwen3.5-4B") }
            if gb >= 16 { band.insert("Qwen3.5-9B") }
            if let one = qwen38(gb) { band.insert(one) }
            band.insert("Ternary-Bonsai-27B")
            band.insert("Ternary-Bonsai-1.7B")
            band.insert("gemma-4-E4B")
            if gb >= 16 { band.insert("gemma-4-12B") }
        }
        let keep = band.union(downloaded)
        return every.filter { name in keep.contains(name) }
    }

    static func offered(unlocked: Bool) -> [String] {
        unlocked ? every : all
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

    private static func family(_ name: String) -> String {
        var out = name
        if name.hasPrefix("Qwen3.8-27B") { out = "Qwen3.8-27B" }
        return out
    }

    private static func quant(_ name: String) -> String? {
        let out: String?
        switch name {
            case "Qwen3.8-27B-IQ1_S": out = "1-bit"
            case "Qwen3.8-27B-IQ2_XXS": out = "2-bit"
            case "Qwen3.8-27B-IQ3_XXS": out = "3-bit"
            case "Qwen3.8-27B-IQ4_XS": out = "4-bit IQ"
            case "Qwen3.8-27B-Q4_K_S": out = "4-bit K"
            default: out = nil
        }
        return out
    }

    static func display(_ name: String) -> String {
        var out = name
        switch name {
            case "Qwen3.5-4B": out = "Qwen3.5 4B"
            case "Qwen3.5-9B": out = "Qwen3.5 9B"
            case "Qwen3.8-27B-IQ1_S", "Qwen3.8-27B-IQ2_XXS",
                 "Qwen3.8-27B-IQ3_XXS", "Qwen3.8-27B-IQ4_XS",
                 "Qwen3.8-27B-Q4_K_S": out = "Qwen3.8 27B"
            case "Ternary-Bonsai-27B": out = "Bonsai 27B"
            case "Ternary-Bonsai-1.7B": out = "Bonsai 1.7B"
            case "gemma-4-E2B": out = "Gemma E2B"
            case "gemma-4-E4B": out = "Gemma E4B"
            case "gemma-4-12B": out = "Gemma 12B"
            default: out = name
        }
        return out
    }

    static func qualified(_ name: String) -> String {
        var out = display(name)
        if let bits = quant(name) { out += " " + bits }
        return out
    }

    static func display(_ name: String, among visible: [String]) -> String {
        let kin = visible.filter { other in
            family(other) == family(name)
        }
        return kin.count > 1 ? qualified(name) : display(name)
    }

}

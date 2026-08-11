import Foundation
import Metal

let modernGPU: Bool = {
    MTLCreateSystemDefaultDevice()?.supportsFamily(.apple7) ?? false
}()

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
            if gb >= 4 && modernGPU && offersSmall(gb) {
                list.append("Qwen3.5-0.8B")
            }
            if gb >= 6 { list.append("QwenPaw-Flash-2B") }
            if gb >= 8 {
                list.append("gemma-4-E4B")
            } else {
                list.append("gemma-4-E2B")
            }
            list.append("Ternary-Bonsai-1.7B")
        } else {
            if offersSmall(gb) { list.append("Qwen3.5-0.8B") }
            list += ["QwenPaw-Flash-2B", "QwenPaw-Flash-4B"]
            if gb >= 16 { list.append("QwenPaw-Flash-9B") }
            if gb >= 16 { list.append("Ternary-Bonsai-27B") }
            list.append("Ternary-Bonsai-1.7B")
            list.append("gemma-4-E2B")
            list.append("gemma-4-E4B")
            if gb >= 16 { list.append("gemma-4-12B") }
        }
        return list
    }

    private static func offersSmall(_ gb: Int) -> Bool {
        gb < 8 || debugBuild
    }

    static var supported: Bool { !all.isEmpty }

    static let fallback = "Qwen3.5-0.8B"

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
        switch name {
            case "Qwen3.5-0.8B": return "Qwen3.5 0.8B"
            case "QwenPaw-Flash-2B": return "QwenPaw 2B"
            case "QwenPaw-Flash-4B": return "QwenPaw 4B"
            case "QwenPaw-Flash-9B": return "QwenPaw 9B"
            case "Ternary-Bonsai-27B": return "Bonsai 27B"
            case "Ternary-Bonsai-1.7B": return "Bonsai 1.7B"
            case "gemma-4-E2B": return "Gemma E2B"
            case "gemma-4-E4B": return "Gemma E4B"
            case "gemma-4-12B": return "Gemma 12B"
            default: return name
        }
    }

}

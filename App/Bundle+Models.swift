import Foundation

extension Bundle {
    // On-device store for downloaded sets; a set lands at models/<name>/<sha>/
    // and HubFetch marks each excluded-from-backup. The store lives in the
    // Data container's Application Support: the ANE compile cache keys on the
    // models' ABSOLUTE paths, so the path must be stable for the cache to stay
    // warm. The Data container is PRESERVED across upgrade installs (Apple DTS)
    // and only regenerates on a delete -- whereas the app-group container,
    // where the store briefly lived, is spuriously regenerated on update by an
    // iOS 16/26 bug (FB11844942), churning the path and re-keying the cache.
    // Application Support is not purged like Caches, so the weights persist.
    private static let store: URL = {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask, appropriateFor: nil,
                                   create: true))
            ?? fm.temporaryDirectory
        return support.appendingPathComponent("models", isDirectory: true)
    }()

    static func modelStore() -> URL { store }

    // The app's display name, for the onboarding title (iOS) and window title
    // (macOS) -- read from the bundle, never hardcoded.
    static var appName: String {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName")
            as? String)
            ?? (Bundle.main.object(forInfoDictionaryKey: "CFBundleName")
                as? String)
            ?? "Gadeon"
    }
}

// The models the toolbar picker offers, gated by installed RAM (see `all`).
// Every set downloads on demand from the Hub (ModelCatalog); the bundle ships
// no weights.
enum Models {
    // Availability by installed RAM, rounded to the nearest GiB (iOS reports a
    // little under the marketing size). iOS: >=4GB adds the 0.8B, >=6GB the
    // 2B, and the 1.7B is offered on every device. macOS: 0.8B/2B/4B, plus 9B
    // at >=16GB.
    static var all: [String] {
        let gb = (ProcessInfo.processInfo.physicalMemory + (1 << 29)) >> 30
        var list: [String] = []
        if isOS {
            if gb >= 4 { list.append("Qwen3.5-0.8B") }
            if gb >= 6 { list.append("QwenPaw-Flash-2B") }
            // The dense 1.7B ternary GGUF (~442 MB) runs on the GPU (Metal),
            // not the ANE, so it needs neither the RAM the CoreML sets want
            // nor an ANE new enough to lower the decode graph -- it is the one
            // model a 3 GB H12-class iPhone can run at all.
            list.append("Ternary-Bonsai-1.7B")
        } else {
            list = ["Qwen3.5-0.8B", "QwenPaw-Flash-2B", "QwenPaw-Flash-4B"]
            if gb >= 16 { list.append("QwenPaw-Flash-9B") }
            // The 27B ternary GGUF is only ~7 GB (Q2_0) + ~78 MB GDN state +
            // lazy paged KV, so it fits the same >=16 GB gate as the 9B. It
            // downloads on demand from the Hub like the CoreML sets.
            if gb >= 16 { list.append("Ternary-Bonsai-27B") }
            // The small ternary GGUF is offered on any Mac for now.
            list.append("Ternary-Bonsai-1.7B")
        }
        return list
    }

    // False when a device is offered no model at all: the app shows a
    // not-supported notice and downloads nothing. Every device reached today
    // gets at least the GPU-run 1.7B, so this is a backstop for a future tier
    // that ships nothing rather than a live gate.
    static var supported: Bool { !all.isEmpty }

    // The base Neural Engine model, and the identity the tiling / sample-prompt
    // rules key off. NOT necessarily runnable here -- see `start`.
    static let fallback = "Qwen3.5-0.8B"

    // The model a fresh install selects, and the sanitizer for a persisted
    // choice: the base model when this device is offered it, else the first
    // one it IS offered. A 3 GB H12-class iPhone is the case that forces the
    // distinction -- handing it `fallback` downloads 0.7 GB and then fails
    // every decode-graph compile, since that Neural Engine cannot lower them.
    static var start: String {
        let list = all
        return list.contains(fallback) ? fallback : (list.first ?? fallback)
    }

    // Cosmetic label for the title bar / picker. The internal id (used above and
    // as the store path / catalog / precook key) is UNCHANGED; the full name
    // still shows in Settings > Models.
    static func display(_ name: String) -> String {
        switch name {
        case "Qwen3.5-0.8B": return "Qwen3.5 0.8B"
        case "QwenPaw-Flash-2B": return "QwenPaw 2B"
        case "QwenPaw-Flash-4B": return "QwenPaw 4B"
        case "QwenPaw-Flash-9B": return "QwenPaw 9B"
        case "Ternary-Bonsai-27B": return "Bonsai 27B"
        case "Ternary-Bonsai-1.7B": return "Bonsai 1.7B"
        default: return name
        }
    }
}

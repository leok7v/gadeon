import Foundation
import Metal

// Whether this GPU is A14 / M1 or newer (Metal's apple7 family), and the best
// available proxy for the NEURAL ENGINE generation, which no API names: the
// A13's H12 ANE cannot compile the 0.8B decode graph at all, and A13 is
// exactly what fails apple7. The same flag decides whether the gemma towers
// can run, for an unrelated reason -- see MetalContext.matrixUnits.
//
// Computed once. `all` is read on every picker render, and creating a device
// to ask is not free.
let modernGPU: Bool = {
    MTLCreateSystemDefaultDevice()?.supportsFamily(.apple7) ?? false
}()

// Installed RAM in GiB, rounded to nearest: iOS reports a little under the
// marketing size, so a 3 GB phone answers 2.67. ONE definition, so the model
// tiers and the speech floor cannot come to different conclusions about what
// a device is.

var installedGB: Int {
    Int((ProcessInfo.processInfo.physicalMemory + (1 << 29)) >> 30)
}

// The ONE `#if DEBUG` in the app, here beside the other two questions the
// model tiers ask about the machine they are running on. A build condition is
// not platform-varying, so it wants one definition rather than the paired
// *-iOS / *-macOS declarations that carry things which genuinely differ.
#if DEBUG
let debugBuild = true
#else
let debugBuild = false
#endif

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

// The models the toolbar picker offers (see `all`). Every set downloads on
// demand from the Hub (ModelCatalog); the bundle ships no weights.
enum Models {
    // Availability by installed RAM, rounded to the nearest GiB (iOS reports a
    // little under the marketing size). iOS: >=4GB AND an A14 adds the 0.8B,
    // >=6GB the 2B, the 1.7B is offered on every device, and exactly one
    // gemma -- E4B at >=8GB, E2B below. macOS: 2B/4B, plus 9B and the 27B at
    // >=16GB, and both gemmas on anything. The 0.8B rides `offersSmall` on
    // both.
    static var all: [String] {
        let gb = installedGB
        var list: [String] = []
        if isOS {
            // RAM alone is NOT the test here. An iPhone 11 Pro Max is an A13
            // with 4 GB, so it clears the tier -- and its H12 Neural Engine
            // cannot compile this model's decode graph at all, so offering it
            // buys a multi-hundred-MB download and then "could not be
            // prepared", twice, with Try Again unable to help.
            if gb >= 4 && modernGPU && offersSmall(gb) {
                list.append("Qwen3.5-0.8B")
            }
            if gb >= 6 { list.append("QwenPaw-Flash-2B") }
            // EXACTLY ONE gemma per device, so the picker never offers a
            // choice between two sizes of the same model. E4B's 3.8 GB of
            // weights do not run on a 4 GB phone -- tried on an iPhone13,1 and
            // it does not come up -- where E2B is measured to, so E2B is what
            // every smaller device gets.
            //
            // WIRED is the number that decides this and no per-process one can
            // see it: an earlier build died on a 12 mini at 121 MB of
            // footprint with 2226 MB of reported headroom, killed for
            // vm-pageshortage while the machine held 3050 MB wired and 30 MB
            // free. E2B fits there only because the two embedding tables are
            // gathered on the CPU, so per_layer_token_embd's window is never
            // bound and never wired.
            //
            // The threshold is the tier the app COMPUTES, not the marketing
            // size, and the two disagree: an iPad16,1 reports 7.72 GiB and
            // rounds to 8, while an iPhone16,1 of the same nominal 8 GB
            // reports 7.4999 and rounds to 7. So a 15 Pro takes the E2B arm
            // deliberately -- read the tier off Footprint's device line before
            // moving this number.
            //
            // E2B rides every device below that on no further gate. It was
            // once gated on the GPU's matrix units, because the batched
            // prefill is built on the simdgroup GEMM and asking an older GPU
            // to build that pipeline does not fail -- it takes the shader
            // compiler service down. The engine falls back to the
            // token-by-token forward there now, so that gate guards nothing:
            // an iPhone12,8 (SE, 2.88 GiB, A13, no matrix units) runs a
            // tool-calling turn end to end at 9.1 t/s prefill.
            if gb >= 8 {
                list.append("gemma-4-E4B")
            } else {
                list.append("gemma-4-E2B")
            }
            // The dense 1.7B ternary GGUF (~442 MB) runs on the GPU (Metal),
            // not the ANE, so it needs neither the RAM the CoreML sets want
            // nor an ANE new enough to lower the decode graph -- it is the one
            // model a 3 GB H12-class iPhone can run at all.
            list.append("Ternary-Bonsai-1.7B")
        } else {
            if offersSmall(gb) { list.append("Qwen3.5-0.8B") }
            list += ["QwenPaw-Flash-2B", "QwenPaw-Flash-4B"]
            if gb >= 16 { list.append("QwenPaw-Flash-9B") }
            // The 27B ternary GGUF is only ~7 GB (Q2_0) + ~78 MB GDN state +
            // lazy paged KV, so it fits the same >=16 GB gate as the 9B. It
            // downloads on demand from the Hub like the CoreML sets.
            if gb >= 16 { list.append("Ternary-Bonsai-27B") }
            // The small ternary GGUF is offered on any Mac for now.
            list.append("Ternary-Bonsai-1.7B")
            // Both gemmas: 2.7 and 3.8 GB on the GPU, so any Mac runs either,
            // and unlike iOS there is room to hold the pair and compare them.
            list.append("gemma-4-E2B")
            list.append("gemma-4-E4B")
            // The 12B takes the same >=16 GB gate as the 27B: 6.8 GB of
            // weights the GPU wires, plus its KV. That KV is smaller than 48
            // layers at 3840 suggests -- its eight full-attention layers keep
            // one 512-wide kv head whose key and value share a projection,
            // and the other forty window at 1024.
            if gb >= 16 { list.append("gemma-4-12B") }
        }
        return list
    }

    // The 0.8B is the SMALL set: quick to download, quick to load, and weak
    // enough that a device with room for anything else is better served by it.
    // So it is offered where nothing better fits -- and in a Debug build,
    // where being the fastest set to get running IS what it is for.
    //
    // A machine that stops being offered it loses the download too:
    // pruneUnavailable erases a set the tiers no longer name, which is what
    // keeps a gating change from stranding gigabytes with no route off the
    // device. Alternating Debug and Release on one machine therefore
    // re-downloads it each way.
    private static func offersSmall(_ gb: Int) -> Bool {
        gb < 8 || debugBuild
    }

    // False when a device is offered no model at all, in which case nothing is
    // downloaded. A backstop for a future tier that ships nothing rather than
    // a live gate: the App Store cannot offer the app below iOS 18, which
    // needs an A12, and every device that clears that bar gets at least the
    // GPU-run 1.7B.
    static var supported: Bool { !all.isEmpty }

    // The base Neural Engine model, and the identity the tiling / sample-prompt
    // rules key off. NOT necessarily runnable here -- see `start`.
    static let fallback = "Qwen3.5-0.8B"

    // The model a fresh install selects, and the sanitizer for a persisted
    // choice that names something this device is not offered.
    //
    // A gemma leads, the larger one where it is offered: it is the model this
    // app is FOR -- the only one that hears and watches as well as reads, and
    // the only one whose weights are a single file rather than an ANE set the
    // device must compile first. `all` offers at most one of the two on iOS,
    // so this picks the Mac's larger one and follows the device's gate
    // everywhere else. The lines under it are what a tier offering neither
    // would land on, and they prefer `fallback` to `list.first` because a 3 GB
    // H12-class iPhone must not be handed a CoreML set whose decode graph its
    // Neural Engine cannot lower.
    static var start: String {
        let list = all
        var out = list.contains(fallback) ? fallback : (list.first ?? fallback)
        if list.contains(e2b) { out = e2b }
        if list.contains(e4b) { out = e4b }
        return out
    }

    private static let e2b = "gemma-4-E2B"
    private static let e4b = "gemma-4-E4B"

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
        case "gemma-4-E2B": return "Gemma E2B"
        case "gemma-4-E4B": return "Gemma E4B"
        case "gemma-4-12B": return "Gemma 12B"
        default: return name
        }
    }
}

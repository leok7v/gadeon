import CoreML
import Foundation
import LLM

// Standalone probe / bench modes: each runs before (or instead
// of) the chat path and exits the process itself. Bodies are
// verbatim main.swift blocks; the dispatcher in main.swift
// keeps their original order.

// Minimal single-turn ChatML wrap (no system prompt, no tool schemas) for the
// greedy probe: the smallest templated prompt that still elicits a direct answer,
// so a slow set's correctness is checkable argmax-deterministically in ~1 block.
// The empty <think></think> is the reasoning-effort-none direct-answer form.
func probeWrap(_ user: String) -> String {
    "<|im_start|>user\n\(user)<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
}

// Bonsai-27B vision tower gate: run the Swift ViT (SIMD/Accelerate) over the
// exact pixels the numpy reference preprocessed and cosine-compare its merged
// embeddings, no LM load. Reference + pixels come from
// scripts/convert/qwen35/bonsai27b_vit_ref.py.
//   gadeon-cli --vit mmproj.gguf --vit-pixels pixels.bin --vit-ref ref.bin
@MainActor func probeVit() throws {
    if let vtIdx = rawArgs.firstIndex(of: "--vit") {
        let mmproj = vtIdx + 1 < rawArgs.count ? rawArgs[vtIdx + 1] : ""
        func pathArg(_ flag: String) -> String? {
            rawArgs.firstIndex(of: flag).flatMap { i in
                i + 1 < rawArgs.count ? rawArgs[i + 1] : nil
            }
        }
        let vit = try ViT(path: mmproj)
        let c = vit.cfg
        err("[vit] \(c.layers) blocks, \(c.embd) wide, \(c.imageSize)px, "
            + "\(c.mergedTokens) merged tokens -> \(c.projDim)\n")
        let pixPath = pathArg("--vit-pixels")
        var pixels = [Float](repeating: 0, count: c.imageSize * c.imageSize * 3)
        if let pixPath {
            let data = try Data(contentsOf: URL(fileURLWithPath: pixPath))
            precondition(data.count == pixels.count * 4,
                         "pixels.bin size mismatch: \(data.count)")
            _ = pixels.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        }
        let t0 = Date()
        let out = vit.forward(pixels: pixels)
        err(String(format: "[vit] forward %.2fs\n", Date().timeIntervalSince(t0)))
        // Cross-gate the GPU tower against the CPU forward (the oracle) on the
        // same pixels: f16 weights are the only delta, so cos must be ~1. The
        // second forward times the resident steady state (no load/dequant).
        if let mvit = try? MetalViT(path: mmproj) {
            let m0 = Date()
            let mout = mvit.forward(pixels: pixels)
            let m1 = Date()
            _ = mvit.forward(pixels: pixels)
            err(String(format: "[vit] metal forward %.2fs (warm %.2fs)\n",
                       m1.timeIntervalSince(m0),
                       Date().timeIntervalSince(m1)))
            var dot: Float = 0, nc: Float = 0, nm: Float = 0
            for i in 0 ..< out.count {
                dot += out[i] * mout[i]
                nc += out[i] * out[i]
                nm += mout[i] * mout[i]
            }
            let mcos = dot / (nc.squareRoot() * nm.squareRoot())
            print(String(format: "VIT metal-vs-cpu cos=%.7f %@", mcos,
                         mcos > 0.999 ? "MATCH" : "MISMATCH"))
            if mcos <= 0.999 { exit(1) }
        } else {
            err("[vit] no Metal device; GPU cross-gate skipped\n")
        }
        if let refPath = pathArg("--vit-ref") {
            let data = try Data(contentsOf: URL(fileURLWithPath: refPath))
            var ref = [Float](repeating: 0, count: data.count / 4)
            _ = ref.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            precondition(ref.count == out.count,
                         "ref count \(ref.count) != out \(out.count)")
            var dot: Float = 0, no: Float = 0, nr: Float = 0
            for i in 0 ..< out.count {
                dot += out[i] * ref[i]
                no += out[i] * out[i]
                nr += ref[i] * ref[i]
            }
            let cos = dot / (no.squareRoot() * nr.squareRoot())
            var worst: Float = 0
            for i in 0 ..< out.count { worst = max(worst, abs(out[i] - ref[i])) }
            print(String(format: "VIT cos=%.7f maxdiff=%.5f %@", cos, worst,
                         cos > 0.9999 ? "MATCH" : "MISMATCH"))
            exit(cos > 0.9999 ? 0 : 1)
        }
        print("VIT ok (no ref given): \(out.count) values")
        exit(0)
    }
}

// Byte-gate the Swift preprocessor against HF's pixel_values (fp16), no model
// load. A vertical flip or patch-order slip shows up on the non-uniform probe.
@MainActor func probeVLPreprocess() throws {
    if let png = vpPng, let bin = vpBin {
        let data = try Data(contentsOf: URL(fileURLWithPath: png))
        let patches = try VisionPreprocess.patches(data, VisionGrid.canonical)
        let ref = try Data(contentsOf: URL(fileURLWithPath: bin))
        var maxDiff: Float = 0
        patches.withUnsafeBytes { pb in
            let pp = pb.bindMemory(to: Float16.self)
            ref.withUnsafeBytes { rb in
                let rp = rb.bindMemory(to: UInt16.self)
                for i in 0 ..< min(patches.count, rp.count) {
                    let d = abs(Float(pp[i]) - Float(Float16(bitPattern: rp[i])))
                    maxDiff = max(maxDiff, d)
                }
            }
        }
        err("[vl-preprocess] \(patches.count) vals maxdiff=\(maxDiff) "
            + (maxDiff < 0.02 ? "MATCH\n" : "MISMATCH\n"))
        exit(maxDiff < 0.02 ? 0 : 1)
    }
}

// Network-tool probes (no model): run one safe tool directly, for bring-up.
@MainActor func probeNet() async {
    if let wi = rawArgs.firstIndex(of: "--web") {
        let q = rawArgs[(wi + 1)...].first { !$0.hasPrefix("--") } ?? ""
        print(await Tools.websearch(q, count: 5))
        exit(0)
    }
    if let ni = rawArgs.firstIndex(of: "--news") {
        let topic = rawArgs[(ni + 1)...].first { !$0.hasPrefix("--") }
        print(await Tools.news(topic))
        exit(0)
    }
    if let fi = rawArgs.firstIndex(of: "--fetch") {
        let u = rawArgs[(fi + 1)...].first { !$0.hasPrefix("--") } ?? ""
        print(await Tools.fetch(u, limit: 1200, offset: 0))
        exit(0)
    }
    if let wi = rawArgs.firstIndex(of: "--weather") {
        let loc = rawArgs[(wi + 1)...].first { !$0.hasPrefix("--") } ?? ""
        print(await Tools.weather(loc))
        exit(0)
    }
}

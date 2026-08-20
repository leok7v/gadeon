// One tensor, one kernel, many iterations, reported as effective GB/s. The
// whole-model bench cannot separate a kernel from prefill, thermals and page
// cache; this reads the weight bytes a kernel MUST move and divides by the
// time it took, so "how far from bandwidth" is answerable directly.
import Foundation
import Metal

public enum MetalKernelBench {

    public static func run(ggufPath: String) throws -> String {
        let model = try BonsaiModel(path: ggufPath)
        let ctx = try MetalContext(model.gguf)
        try ctx.prewarm()
        let c = model.cfg
        var picks: [(String, GGUFTensor)] = []
        for name in ["blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                     "blk.0.attn_qkv.weight", "blk.0.ssm_out.weight"] {
            if let t = model.gguf.maybe(name) { picks.append((name, t)) }
        }
        picks.append(("lm_head", model.output))
        var out = "kernel bench: "
            + URL(fileURLWithPath: ggufPath).lastPathComponent + "\n"
        out += pad("tensor", 24) + pad("kernel", 10) + pad("N", 4)
            + pad("ms", 10) + "GB/s\n"
        for (name, t) in picks {
            let k = t.dims[0], m = t.dims[1]
            let bytes = GGUF.rowByteCount(t.type, k) * m
            let x = ctx.makeF32(8 * k)
            let y = ctx.makeF32(8 * m)
            seed(x, 8 * k)
            // The first timed dispatch of a tensor pays its page-in.
            _ = time(ctx, t, x, y, n: 1, tile: false, model: model)
            for n in [1, 2, 3, 4, 5] {
                for kind in ["auto", "tile"] {
                    let ms = time(ctx, t, x, y, n: n, tile: kind == "tile",
                                  model: model)
                    if ms > 0 {
                        let gbs = Double(bytes) / (ms / 1000) / 1e9
                        out += pad(name, 24) + pad(kind, 10) + pad("\(n)", 4)
                            + pad(String(format: "%.3f", ms), 10)
                            + String(format: "%.1f", gbs) + "\n"
                    }
                }
            }
        }
        _ = c
        out += occupancy(ctx)
        return out
    }

    // maxTotalThreadsPerThreadgroup falls as register use rises.
    private static func occupancy(_ ctx: MetalContext) -> String {
        var out = "\nregister pressure (maxThreads/threadgroup, "
            + "1024 = uncontended)\n"
        var names = ["q2_e8_gemv", "q4_0_gemv"]
        for r in 2...5 {
            names.append("q2_e8_gemm_nb_r\(r)")
            names.append("q4_0_gemm_nb_r\(r)")
        }
        for n in names {
            if let p = try? ctx.pipeline(n) {
                out += pad(n, 24) + "\(p.maxTotalThreadsPerThreadgroup)\n"
            }
        }
        return out
    }

    private static func pad(_ s: String, _ n: Int) -> String {
        s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
    }

    private static func seed(_ b: MTLBuffer, _ n: Int) {
        let p = b.contents().assumingMemoryBound(to: Float.self)
        var s: UInt64 = 0x2545_F491_4F6C_DD1D
        for i in 0..<n {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            p[i] = Float(Int32(truncatingIfNeeded: s)) / 2.147e9
        }
    }

    // Iterations ride ONE command buffer so the measurement is GPU time for
    // the kernel, not per-commit latency.
    private static func time(_ ctx: MetalContext, _ t: GGUFTensor,
                             _ x: MTLBuffer, _ y: MTLBuffer, n: Int,
                             tile: Bool, model: BonsaiModel) -> Double {
        let reps = 20
        let off = ctx.window(UInt64(t.base - model.gguf.map))
        var result = -1.0
        let narrowOK = (t.type == .q2_e8 || t.type == .q4_0)
            && n > 1 && n <= MetalEnc.narrowMax
        if tile || narrowOK || n == 1 {
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            let f = MetalEnc(ctx: ctx, e: e)
            for _ in 0..<reps {
                if n == 1 {
                    f.gemv(t, x: x, out: y, off: off)
                } else if tile {
                    f.gemmTile(t, X: x, out: y, off: off, N: n)
                } else {
                    f.gemmNarrow(t, X: x, out: y, off: off, N: n)
                }
            }
            e.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if cb.error == nil {
                result = (cb.gpuEndTime - cb.gpuStartTime) * 1000 / Double(reps)
            }
        }
        return result
    }
}

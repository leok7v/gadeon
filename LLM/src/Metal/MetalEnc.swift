// Typed one-token dispatch surface: each method encodes ONE kernel onto the
// shared command encoder with the buffer bindings and kargs its MSL
// counterpart
// expects. Kept separate from MetalEngine so the forward loop reads as the op
// sequence (gemv / rmsnorm / gdnScan / ...) and the binding detail lives here.
// All buffers are hazard-tracked, so the engine safely reuses scratch across
// the sequential dispatches on one encoder.
//
// The `try! ctx.pipeline(name)` below is sound because MetalContext.prewarm
// built every buildable pipeline during engine init: the lookup here is a
// dictionary hit, so it can only fail on a name this library does not carry --
// a typo, not a runtime condition, and one that fails on the first dispatch of
// every run rather than intermittently.
import Foundation
import Metal

// Which slot of a GDN state ring a batched kernel reads, and how many slots
// it may write forward into. `inPlace` is one slot, i.e. the plain path.
struct StateRing {
    let slot0: Int
    let slots: Int
    static let inPlace = StateRing(slot0: 0, slots: 1)
}

struct MetalEnc {
    let ctx: MetalContext
    let e: MTLComputeCommandEncoder

    // LLM_GPU_LABELS=1 wraps each dispatch in a debug group named for its
    // kernel, so an Instruments Metal capture names every GPU interval (they
    // all
    // run through one unlabeled compute encoder otherwise). No numeric effect.
    static let labels =
        ProcessInfo.processInfo.environment["LLM_GPU_LABELS"] == "1"
    private func push(_ n: String) {
        if MetalEnc.labels { e.pushDebugGroup(n) }
    }
    private func pop() {
        if MetalEnc.labels { e.popDebugGroup() }
    }

    // ---- generic dispatch shapes ----------------------------------------
    // One thread per element (non-uniform grid; the kernel guards the tail).
    private func grid1D(_ name: String, _ n: Int,
                        _ bind: (MTLComputeCommandEncoder) -> Void) {
        push(name)
        let pipe = try! ctx.pipeline(name)
        e.setComputePipelineState(pipe)
        bind(e)
        let w = min(256, pipe.maxTotalThreadsPerThreadgroup)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                          threadsPerThreadgroup:
                            MTLSize(width: w, height: 1, depth: 1))
        pop()
    }

    // `groups` threadgroups of `tpg` threads; `tgmem` bytes of threadgroup(0).
    private func groups(_ name: String, _ groups: Int, tpg: Int,
                        tgmem: Int = 0,
                        _ bind: (MTLComputeCommandEncoder) -> Void) {
        push(name)
        let pipe = try! ctx.pipeline(name)
        e.setComputePipelineState(pipe)
        bind(e)
        // Threadgroup-memory length must be a multiple of 16 bytes. Guard the
        // rounded request against the device limit so an oversized allocation
        // fails HERE with a clear message instead of as an opaque encoder
        // fault (a T-scaled request used to blow this past ~8K context).
        if tgmem > 0 {
            let need = (tgmem + 15) & ~15
            precondition(need <= ctx.device.maxThreadgroupMemoryLength,
                "\(name): threadgroup memory \(need)B exceeds device max "
                + "\(ctx.device.maxThreadgroupMemoryLength)B")
            e.setThreadgroupMemoryLength(need, index: 0)
        }
        e.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        pop()
    }

    // ---- mat-vec: out[M] = W @ x  (W ne0=K, ne1=M at byte off) -----------
    // Dispatched on the TENSOR's type, not the caller's: a gemma forward mixes
    // Q4_0 attention with Q2_0 wide MLP, so the engine names a tensor and this
    // picks the kernel. Mirrors GQ.matvec.
    func gemv(_ w: GGUFTensor, x: MTLBuffer, out: MTLBuffer, off: WeightRef,
              xOff: Int = 0, outOff: Int = 0) {
        let k = w.dims[0], m = w.dims[1]
        var a = GemvArgs(woff: off.local, K: UInt32(k), M: UInt32(m))
        let name: String
        switch w.type {
        case .q2_0: name = "q2_0_gemv"
        case .q2_e8: name = "q2_e8_gemv"
        case .q4_0: name = "q4_0_gemv"
        case .q8_0: name = "q8_0_gemv"
        case .bf16: name = "bf16_gemv"
        case .f16: name = "f16_gemv"
        case .f32: name = "f32_gemv"
        default:
            fatalError("MetalEnc.gemv: no kernel for \(w.type) (\(w.name))")
        }
        // one simdgroup (32 threads) per 8 output rows (NR0=8 in the kernel).
        let rows = 8
        groups(name, (m + rows - 1) / rows, tpg: 32) { e in
            e.setBuffer(off.buf, offset: 0, index: 0)
            e.setBuffer(x, offset: xOff, index: 1)
            e.setBuffer(out, offset: outOff, index: 2)
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 3)
            if w.type == .q2_e8 {
                e.setBuffer(ctx.e8pTable(), offset: 0, index: 4)
            }
        }
    }

    // out[N,M] = X[N,K] @ W for a tensor with no batched kernel. The
    // unquantized per-layer projection is the only one in gemma, and a
    // row loop still rides the caller's ONE command buffer -- the round trip
    // is what prefill was paying, not the dispatch.
    func gemmRows(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer,
                  off: WeightRef, N: Int) {
        let k = w.dims[0], m = w.dims[1]
        for j in 0 ..< N {
            gemv(w, x: X, out: out, off: off, xOff: j * k * 4,
                 outOff: j * m * 4)
        }
    }

    // ---- Q2_0 batched mat-mat: out[N,M] = X[N,K] @ W  (prefill) ----------
    // Grid (ceil(N/8), M) simdgroups; the kernel amortizes each weight row
    // across a tile of 8 token-columns.
    // Half tiles (smem 6144 in-bounds / 8192 for a partial tile) beat the f32
    // kernel's 12288 by raising resident-threadgroup occupancy: prefill is
    // compute-bound, so the win is more simdgroups in flight, not fewer bytes.
    // MEASURED on an M3, interleaved A/B with a cooldown before every run
    // (uncooled runs throttle and invert the result): 27B 42.7 -> 47.6 t/s
    // (+11.5%, four reps 1.104-1.127), 1.7B 602 -> 638 t/s (+6.0%). Parity
    // holds -- every token-level self-test matches the SIMD oracle, and the
    // GEMM sits at 1.9e-4 RELATIVE error, inside fp16's own 4.9e-4 epsilon.
    // LLM_F16_TILES=0 restores the f32 tiles for an A/B. It governs all three
    // weight types, and its REACH is wider than the name suggests: every
    // gemma projection goes through `linear(X:N:)`, so it swaps the vision
    // and audio towers onto f32 tiles as well as the text prefill.
    // The q4_0/q8_0 f32 twins were written to test whether fp16 tiles were
    // what kept batched prefill from reproducing the per-token path. They are
    // NOT -- measured, both paths land equally close to HF's own logits
    // (0.99875 batched / 0.99890 with f32 tiles / 0.99901 per-token) -- and
    // f32 costs 6% of prefill. See gemmaBatchGate for why exact agreement is
    // unreachable in the first place. Kept as the A/B instrument that
    // settled it.
    static let f16Tiles =
        ProcessInfo.processInfo.environment["LLM_F16_TILES"] != "0"

    // LLM_ATTN_MM=0 takes the batched attention back to the scalar body, which
    // is the A/B instrument: the two compute the same thing by different
    // means, and only a measurement says which is faster on a given part.
    static let attnMatrix =
        ProcessInfo.processInfo.environment["LLM_ATTN_MM"] != "0"

    // Per TYPE because the two disagree; --kernel-bench has the numbers.
    // LLM_NARROW_MAX=0 restores the tile everywhere.
    static let narrowMax = min(4,
        Int(ProcessInfo.processInfo.environment["LLM_NARROW_MAX"] ?? "") ?? 4)
    static let narrowMaxQ4 = min(narrowMax, 2)

    func gemmNarrow(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer,
                    off: WeightRef, N: Int) {
        let k = w.dims[0], m = w.dims[1]
        var a = GemvArgs(woff: off.local, K: UInt32(k), M: UInt32(m))
        let rows = 4
        let e8 = w.type == .q2_e8
        let name = (e8 ? "q2_e8_gemm_nb_r" : "q4_0_gemm_nb_r") + "\(N)"
        groups(name, (m + rows - 1) / rows, tpg: 32) { e in
            e.setBuffer(off.buf, offset: 0, index: 0)
            e.setBuffer(X, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 3)
            if e8 { e.setBuffer(ctx.e8pTable(), offset: 0, index: 4) }
        }
    }

    func gemm(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer, off: WeightRef,
              N: Int) {
        let k = w.dims[0], m = w.dims[1]
        var a = GemvArgs(woff: off.local, K: UInt32(k), M: UInt32(m))
        var nn = UInt32(N)
        let cap = w.type == .q4_0 ? MetalEnc.narrowMaxQ4 : MetalEnc.narrowMax
        let narrow = (w.type == .q2_e8 || w.type == .q4_0)
            && N > 1 && N <= cap
        // The half-tile spill path needs 8192B f32 temp; a fully in-bounds
        // tile
        // (M multiple of 64, N of 32) never spills, so 6144 suffices.
        let fullTile = m % 64 == 0 && N % 32 == 0
        let name: String
        switch w.type {
        case .q2_0:
            name = MetalEnc.f16Tiles ? "q2_0_gemm_mm_h" : "q2_0_gemm_mm"
        case .q4_0:
            name = MetalEnc.f16Tiles ? "q4_0_gemm_mm_h" : "q4_0_gemm_mm"
        case .q8_0:
            name = MetalEnc.f16Tiles ? "q8_0_gemm_mm_h" : "q8_0_gemm_mm"
        case .q2_e8:
            name = MetalEnc.f16Tiles ? "q2_e8_gemm_mm_h" : "q2_e8_gemm_mm"
        case .f32, .f16, .bf16:
            name = ""
        default:
            fatalError("MetalEnc.gemm: no kernel for \(w.type) (\(w.name))")
        }
        if narrow {
            gemmNarrow(w, X: X, out: out, off: off, N: N)
        } else if name.isEmpty {
            gemmRows(w, X: X, out: out, off: off, N: N)
        } else {
            gemmTiled(name, w, X: X, out: out, off: off, N: N, a: &a, nn: &nn,
                      fullTile: fullTile)
        }
    }

    // Force the prefill tile regardless of width, for the A/B in
    // MetalKernelBench.
    func gemmTile(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer,
                  off: WeightRef, N: Int) {
        let k = w.dims[0], m = w.dims[1]
        var a = GemvArgs(woff: off.local, K: UInt32(k), M: UInt32(m))
        var nn = UInt32(N)
        let fullTile = m % 64 == 0 && N % 32 == 0
        let name: String
        switch w.type {
        case .q2_0: name = MetalEnc.f16Tiles ? "q2_0_gemm_mm_h" : "q2_0_gemm_mm"
        case .q4_0: name = MetalEnc.f16Tiles ? "q4_0_gemm_mm_h" : "q4_0_gemm_mm"
        case .q8_0: name = MetalEnc.f16Tiles ? "q8_0_gemm_mm_h" : "q8_0_gemm_mm"
        case .q2_e8:
            name = MetalEnc.f16Tiles ? "q2_e8_gemm_mm_h" : "q2_e8_gemm_mm"
        default: name = ""
        }
        if !name.isEmpty {
            gemmTiled(name, w, X: X, out: out, off: off, N: N, a: &a, nn: &nn,
                      fullTile: fullTile)
        }
    }

    private func gemmTiled(_ name: String, _ w: GGUFTensor, X: MTLBuffer,
                           out: MTLBuffer, off: WeightRef, N: Int,
                           a: inout GemvArgs, nn: inout UInt32,
                           fullTile: Bool) {
        let m = w.dims[1]
        // Every kernel above is a simdgroup-matrix one, and prewarm skips
        // exactly those on a GPU that cannot build them -- so arriving here
        // without matrix units means a capability gate upstream let an
        // attachment through. Said plainly HERE, because the alternative is
        // the `try!` below dying inside the Metal compiler with nothing in
        // the message pointing at the cause.
        precondition(ctx.matrixUnits,
                     "MetalEnc.gemm: \(name) needs simdgroup matrix units; "
                     + "this GPU has none, so its caller should have been "
                     + "gated (see Gemma4MetalBackend.supportsSoftTokens)")
        let tgmem = MetalEnc.f16Tiles ? (fullTile ? 6144 : 8192) : 12288
        push(name)
        let pipe = try! ctx.pipeline(name)
        e.setComputePipelineState(pipe)
        e.setBuffer(off.buf, offset: 0, index: 0)
        e.setBuffer(X, offset: 0, index: 1)
        e.setBuffer(out, offset: 0, index: 2)
        e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 3)
        e.setBytes(&nn, length: 4, index: 4)
        if w.type == .q2_e8 {
            e.setBuffer(ctx.e8pTable(), offset: 0, index: 5)
        }
        e.setThreadgroupMemoryLength(tgmem, index: 0)
        e.dispatchThreadgroups(
            MTLSize(width: (N + 31) / 32, height: (m + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        pop()
    }

    // ---- row dequant: out[n] = dequant(weight span at off) ---------------
    // `outOff` is a BYTE offset into `out`, so a batched caller can dequant
    // straight into row r of an [N][n] buffer.
    func dequantRow(weightOff: WeightRef, out: MTLBuffer, n: Int,
                    type: GGUFType = .q2_0, outOff: Int = 0) {
        var a = GemvArgs(woff: weightOff.local, K: UInt32(n), M: UInt32(n))
        let name: String
        switch type {
        case .q2_0: name = "q2_0_dequant_row"
        case .q4_0: name = "q4_0_dequant_row"
        case .q8_0: name = "q8_0_dequant_row"
        case .bf16: name = "bf16_dequant_row"
        case .f16: name = "f16_dequant_row"
        case .f32: name = "f32_dequant_row"
        default:
            fatalError("MetalEnc.dequantRow: no kernel for \(type)")
        }
        grid1D(name, n) { e in
            e.setBuffer(weightOff.buf, offset: 0, index: 0)
            e.setBuffer(out, offset: outOff, index: 1)
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 2)
        }
    }

    func argmaxRows(x: MTLBuffer, out: MTLBuffer, n: Int, rows: Int) {
        var nn = UInt32(n)
        groups("argmax_rows", rows, tpg: 256, tgmem: 0) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(out, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
        }
    }

    // ---- norms -----------------------------------------------------------
    func rmsnorm(x: MTLBuffer, weightOff: WeightRef, out: MTLBuffer, n: Int,
                 eps: Float, xOff: Int = 0, outOff: Int = 0) {
        var a = NormArgs(woff: weightOff.local, n: UInt32(n), eps: eps)
        groups("rmsnorm", 1, tpg: 256, tgmem: 128) { e in
            e.setBuffer(x, offset: xOff, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBuffer(out, offset: outOff, index: 2)
            e.setBytes(&a, length: MemoryLayout<NormArgs>.stride, index: 3)
        }
    }

    // Batched whole-vector rmsnorm to a separate output (attn_norm/post_norm
    // in
    // prefill): rows tokens, each of length n, x -> y.
    func rmsnormBatch(x: MTLBuffer, weightOff: WeightRef, y: MTLBuffer, n: Int,
                      rows: Int, eps: Float) {
        var a = NormArgs(woff: weightOff.local, n: UInt32(n), eps: eps)
        groups("rmsnorm_batch", rows, tpg: 256, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBuffer(y, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<NormArgs>.stride, index: 3)
        }
    }

    func rmsnormRows(x: MTLBuffer, xoff: UInt32, d: Int, rows: Int,
                     weightOff: WeightRef, eps: Float) {
        var a = RowArgs(woff: weightOff.local, d: UInt32(d), xoff: xoff,
                        eps: eps)
        groups("rmsnorm_rows", rows, tpg: 128, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<RowArgs>.stride, index: 2)
        }
    }

    func l2normRows(x: MTLBuffer, xoff: UInt32, d: Int, rows: Int,
                    eps: Float) {
        var a = RowArgs(woff: 0, d: UInt32(d), xoff: xoff, eps: eps)
        groups("l2norm_rows", rows, tpg: 128, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RowArgs>.stride, index: 1)
        }
    }

    // Batched l2norm: `rowsPerTok` rows of length d for each of `tokens`
    // tokens,
    // token n based at n*tokStride (q|k across a whole prefill chunk in ONE
    // dispatch, vs 2*tokens tiny per-token l2normRows calls).
    func l2normRowsBatch(x: MTLBuffer, d: Int, rowsPerTok: Int, tokStride: Int,
                         tokens: Int, eps: Float) {
        var a = RowBatchArgs(d: UInt32(d), rowsPerTok: UInt32(rowsPerTok),
                             tokStride: UInt32(tokStride), eps: eps)
        groups("l2norm_rows_batch", tokens * rowsPerTok, tpg: 128,
               tgmem: 128) {
            e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RowBatchArgs>.stride, index: 1)
        }
    }

    // ---- gemma norms: BF16 weights, and the scale-free V norm ------------
    func rmsnormBF16(x: MTLBuffer, weightOff: WeightRef, out: MTLBuffer,
                     n: Int, eps: Float) {
        var a = NormArgs(woff: weightOff.local, n: UInt32(n), eps: eps)
        groups("rmsnorm_bf16", 1, tpg: 256, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<NormArgs>.stride, index: 3)
        }
    }

    func rmsnormRowsBF16(x: MTLBuffer, xoff: UInt32, d: Int, rows: Int,
                         weightOff: WeightRef, eps: Float) {
        var a = RowArgs(woff: weightOff.local, d: UInt32(d), xoff: xoff,
                        eps: eps)
        groups("rmsnorm_rows_bf16", rows, tpg: 128, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<RowArgs>.stride, index: 2)
        }
    }

    func rmsnormRowsNoWeight(x: MTLBuffer, xoff: UInt32, d: Int, rows: Int,
                             eps: Float) {
        var a = RowArgs(woff: 0, d: UInt32(d), xoff: xoff, eps: eps)
        groups("rmsnorm_rows_noweight", rows, tpg: 128, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RowArgs>.stride, index: 1)
        }
    }

    // ---- gemma elementwise glue ------------------------------------------
    func scaleInPlace(x: MTLBuffer, n: Int, s: Float, off: Int = 0) {
        var nn = UInt32(n)
        var ss = s
        grid1D("scale_inplace", n) { e in
            e.setBuffer(x, offset: off, index: 0)
            e.setBytes(&nn, length: 4, index: 1)
            e.setBytes(&ss, length: 4, index: 2)
        }
    }

    func addScaled(a: MTLBuffer, b: MTLBuffer, n: Int, s: Float) {
        var nn = UInt32(n)
        var ss = s
        grid1D("add_scaled", n) { e in
            e.setBuffer(a, offset: 0, index: 0)
            e.setBuffer(b, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
            e.setBytes(&ss, length: 4, index: 3)
        }
    }

    func geluMul(a: MTLBuffer, b: MTLBuffer, n: Int, bOff: Int = 0) {
        var g = GeluMulArgs(n: UInt32(n), boff: UInt32(bOff))
        grid1D("gelu_mul", n) { e in
            e.setBuffer(a, offset: 0, index: 0)
            e.setBuffer(b, offset: 0, index: 1)
            e.setBytes(&g, length: MemoryLayout<GeluMulArgs>.stride, index: 2)
        }
    }

    func softcap(x: MTLBuffer, n: Int, cap: Float) {
        var nn = UInt32(n)
        var cc = cap
        grid1D("softcap", n) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&nn, length: 4, index: 1)
            e.setBytes(&cc, length: 4, index: 2)
        }
    }

    func rmsnormBatchBF16(x: MTLBuffer, weightOff: WeightRef, y: MTLBuffer,
                          n: Int, rows: Int, eps: Float) {
        var a = NormArgs(woff: weightOff.local, n: UInt32(n), eps: eps)
        groups("rmsnorm_batch_bf16", rows, tpg: 256, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBuffer(y, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<NormArgs>.stride, index: 3)
        }
    }

    func gemmaVisionRope(x: MTLBuffer, cos: MTLBuffer, sin: MTLBuffer,
                         rowStride: Int, headDim: Int, nHead: Int, N: Int) {
        var a = GVRopeArgs(rowStride: UInt32(rowStride), off: 0,
                           headDim: UInt32(headDim), nHead: UInt32(nHead),
                           N: UInt32(N))
        grid1D("gemma_vit_rope", N * nHead * (headDim / 2)) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(cos, offset: 0, index: 1)
            e.setBuffer(sin, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<GVRopeArgs>.stride, index: 3)
        }
    }

    func gemmaVisionAttn(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
                         mask: MTLBuffer, out: MTLBuffer, n: Int, hd: Int,
                         nHead: Int) {
        var a = GVAttnArgs(n: UInt32(n), hd: UInt32(hd),
                           nHead: UInt32(nHead))
        groups("gemma_vit_attn", n * nHead, tpg: 32,
               tgmem: hd * MemoryLayout<Float>.stride) { e in
            e.setBuffer(q, offset: 0, index: 0)
            e.setBuffer(k, offset: 0, index: 1)
            e.setBuffer(v, offset: 0, index: 2)
            e.setBuffer(mask, offset: 0, index: 3)
            e.setBuffer(out, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<GVAttnArgs>.stride, index: 5)
        }
    }

    // ---- the tower's own activation --------------------------------------
    // Dispatched on the value the file names, the same way gemv dispatches on
    // a tensor's type, so swapping two implemented curves stays a data
    // change.
    func activateMul(_ act: GemmaActivation, a: MTLBuffer, b: MTLBuffer,
                     n: Int, bOff: Int = 0) {
        switch act {
        case .geluTanh: geluMul(a: a, b: b, n: n, bOff: bOff)
        case .silu:
            precondition(bOff == 0, "silu_mul takes no b offset")
            siluMul(a: a, b: b, n: n)
        }
    }

    // The batched per-layer input: each token's [n] gate against its own row
    // of a [bStride]-wide table at a fixed layer offset.
    func activateMulRows(_ act: GemmaActivation, a: MTLBuffer, b: MTLBuffer,
                         n: Int, rows: Int, aStride: Int, bStride: Int,
                         bOff: Int) {
        precondition(act == .geluTanh,
                     "gelu_mul_rows is the only strided form; \(act) would "
                     + "need its own kernel")
        var g = GeluMulRowsArgs(n: UInt32(n), rows: UInt32(rows),
                                aStride: UInt32(aStride),
                                bStride: UInt32(bStride), boff: UInt32(bOff))
        grid1D("gelu_mul_rows", rows * n) { e in
            e.setBuffer(a, offset: 0, index: 0)
            e.setBuffer(b, offset: 0, index: 1)
            e.setBytes(&g, length: MemoryLayout<GeluMulRowsArgs>.stride,
                       index: 2)
        }
    }

    func activateInPlace(_ act: GemmaActivation, x: MTLBuffer, n: Int) {
        switch act {
        case .geluTanh: gelu(x: x, n: n)
        case .silu: siluInPlace(x: x, n: n)
        }
    }

    // ---- static range quantization ---------------------------------------
    // A scale of 0 marks an uncalibrated linear, where the clamp is the
    // identity -- so it is SKIPPED rather than dispatched, and no pass over
    // the buffer happens at all.
    func srqInPlace(x: MTLBuffer, n: Int, s: Float) {
        if s != 0 {
            var a = SrqArgs(n: UInt32(n), s: s)
            grid1D("srq_inplace", n) { e in
                e.setBuffer(x, offset: 0, index: 0)
                e.setBytes(&a, length: MemoryLayout<SrqArgs>.stride, index: 1)
            }
        }
    }

    func srqTo(src: MTLBuffer, dst: MTLBuffer, n: Int, s: Float) {
        var a = SrqArgs(n: UInt32(n), s: s)
        grid1D("srq_to", n) { e in
            e.setBuffer(src, offset: 0, index: 0)
            e.setBuffer(dst, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<SrqArgs>.stride, index: 2)
        }
    }

    // One QUANTIZED projection with the trained clamp around it. The input is
    // clamped into `scratch` rather than in place because it is usually a
    // shared buffer -- one norm feeds q, k and v -- and the output is clamped
    // in place because it belongs to this linear alone.
    func linear(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer, off: WeightRef,
                N: Int, srq: SRQ, scratch: MTLBuffer) {
        var input = X
        if srq.input != 0 {
            srqTo(src: X, dst: scratch, n: N * w.dims[0], s: srq.input)
            input = scratch
        }
        gemm(w, X: input, out: out, off: off, N: N)
        srqInPlace(x: out, n: N * w.dims[1], s: srq.output)
    }

    func linear(_ w: GGUFTensor, x: MTLBuffer, out: MTLBuffer, off: WeightRef,
                srq: SRQ, scratch: MTLBuffer) {
        var input = x
        if srq.input != 0 {
            srqTo(src: x, dst: scratch, n: w.dims[0], s: srq.input)
            input = scratch
        }
        gemv(w, x: input, out: out, off: off)
        srqInPlace(x: out, n: w.dims[1], s: srq.output)
    }

    // ---- gemma audio (Conformer tower) -----------------------------------
    func siluInPlace(x: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("silu_inplace", n) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&nn, length: 4, index: 1)
        }
    }

    func fmaInPlace(x: MTLBuffer, y: MTLBuffer, n: Int, s: Float) {
        var a = FmaArgs(n: UInt32(n), s: s)
        grid1D("fma_inplace", n) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(y, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<FmaArgs>.stride, index: 2)
        }
    }

    func gluRows(src: MTLBuffer, dst: MTLBuffer, rows: Int, width: Int) {
        var a = GluArgs(n: UInt32(rows), e: UInt32(width))
        grid1D("glu_rows", rows * width) { e in
            e.setBuffer(src, offset: 0, index: 0)
            e.setBuffer(dst, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<GluArgs>.stride, index: 2)
        }
    }

    func depthwiseCausal(x: MTLBuffer, w: MTLBuffer, out: MTLBuffer,
                         rows: Int, width: Int, k: Int) {
        var a = DwArgs(n: UInt32(rows), e: UInt32(width), k: UInt32(k))
        grid1D("depthwise_causal", rows * width) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(w, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<DwArgs>.stride, index: 3)
        }
    }

    func scaleHeadDims(x: MTLBuffer, scales: MTLBuffer, rows: Int,
                       width: Int, hd: Int) {
        var a = HeadScaleArgs(n: UInt32(rows), e: UInt32(width),
                              hd: UInt32(hd))
        grid1D("scale_head_dims", rows * width) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(scales, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<HeadScaleArgs>.stride,
                       index: 2)
        }
    }

    func gemmaAudioAttn(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
                        relk: MTLBuffer, out: MTLBuffer, rows: Int,
                        width: Int, heads: Int, hd: Int, chunk: Int,
                        context: Int, past: Int, cap: Float) {
        precondition(context <= 32,
                     "gemma_audio_attn stages one window's scores in a "
                     + "32-slot thread array")
        var a = AAttnArgs(n: UInt32(rows), e: UInt32(width),
                          heads: UInt32(heads), hd: UInt32(hd),
                          chunk: UInt32(chunk), ctx: UInt32(context),
                          past: UInt32(past), cap: cap)
        grid1D("gemma_audio_attn", rows * heads) { e in
            e.setBuffer(q, offset: 0, index: 0)
            e.setBuffer(k, offset: 0, index: 1)
            e.setBuffer(v, offset: 0, index: 2)
            e.setBuffer(relk, offset: 0, index: 3)
            e.setBuffer(out, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<AAttnArgs>.stride, index: 5)
        }
    }

    // ---- gemma rope: an explicit rotated-pair count ----------------------
    func ropeGemma(x: MTLBuffer, headDim: Int, nHead: Int, rotated: Int,
                   base: Float, pos: Int) {
        var a = RopeGemmaArgs(headDim: UInt32(headDim), nHead: UInt32(nHead),
                              rotated: UInt32(rotated), base: base,
                              pos: UInt32(pos))
        grid1D("rope_gemma", nHead * rotated) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RopeGemmaArgs>.stride,
                       index: 1)
        }
    }

    // ---- partial NEOX rope ----------------------------------------------
    func rope(x: MTLBuffer, headDim: Int, nHead: Int, nRot: Int, base: Float,
              pos: Int) {
        var a = RopeArgs(headDim: UInt32(headDim), nHead: UInt32(nHead),
                         nRot: UInt32(nRot), base: base, pos: UInt32(pos))
        grid1D("rope_neox", nHead * (nRot / 2)) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RopeArgs>.stride, index: 1)
        }
    }

    // ---- elementwise glue ------------------------------------------------
    func siluMul(a: MTLBuffer, b: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("silu_mul", n) { e in
            e.setBuffer(a, offset: 0, index: 0)
            e.setBuffer(b, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
        }
    }

    func mulSilu(a: MTLBuffer, b: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("mul_silu", n) { e in
            e.setBuffer(a, offset: 0, index: 0)
            e.setBuffer(b, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
        }
    }

    func accumOuter(h: MTLBuffer, src: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        push("accum_outer")
        let pipe = try! ctx.pipeline("accum_outer")
        e.setComputePipelineState(pipe)
        e.setBuffer(h, offset: 0, index: 0)
        e.setBuffer(src, offset: 0, index: 1)
        e.setBytes(&nn, length: 4, index: 2)
        e.dispatchThreads(MTLSize(width: n, height: n, depth: 1),
                          threadsPerThreadgroup:
                            MTLSize(width: 32, height: 8, depth: 1))
        pop()
    }

    func accumSq(dst: MTLBuffer, src: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("accum_sq", n) { e in
            e.setBuffer(dst, offset: 0, index: 0)
            e.setBuffer(src, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
        }
    }

    func add(x: MTLBuffer, y: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("add_inplace", n) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(y, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
        }
    }

    // ---- GDN -------------------------------------------------------------
    // `count` = nV for one token, N*nV batched (dt/aNeg indexed by gid%nV).
    // The one dispatch reading TWO weights, so it binds two windows: ssm_dt
    // and ssm_a are separate tensors and nothing puts them in the same one.
    func gdnGate(bPre: MTLBuffer, aPre: MTLBuffer, dtOff: WeightRef,
                 aOff: WeightRef, beta: MTLBuffer, g: MTLBuffer, nV: Int,
                 count: Int) {
        var a = GateArgs(dtOff: dtOff.local, aOff: aOff.local, nV: UInt32(nV))
        var total = UInt32(count)
        grid1D("gdn_gate", count) { e in
            e.setBuffer(bPre, offset: 0, index: 0)
            e.setBuffer(aPre, offset: 0, index: 1)
            e.setBuffer(dtOff.buf, offset: 0, index: 2)
            e.setBuffer(aOff.buf, offset: 0, index: 3)
            e.setBuffer(beta, offset: 0, index: 4)
            e.setBuffer(g, offset: 0, index: 5)
            e.setBytes(&a, length: MemoryLayout<GateArgs>.stride, index: 6)
            e.setBytes(&total, length: 4, index: 7)
        }
    }

    func gdnConv(qkvMix: MTLBuffer, convState: MTLBuffer, cwOff: WeightRef,
                 out: MTLBuffer, convDim: Int, dConv: Int) {
        var a = ConvArgs(cwOff: cwOff.local, convDim: UInt32(convDim),
                         dConv: UInt32(dConv))
        grid1D("gdn_conv", convDim) { e in
            e.setBuffer(qkvMix, offset: 0, index: 0)
            e.setBuffer(convState, offset: 0, index: 1)
            e.setBuffer(cwOff.buf, offset: 0, index: 2)
            e.setBuffer(out, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<ConvArgs>.stride, index: 4)
        }
    }

    // convOut holds [q(keyDim) | k(keyDim) | v(valueDim)]; bind the three
    // spans
    // by byte offset (keyDim*4 is 256-aligned, safe for setBuffer offset).
    func gdnScan(convOut: MTLBuffer, keyDim: Int, g: MTLBuffer,
                 beta: MTLBuffer,
                 S: MTLBuffer, o: MTLBuffer, nV: Int, nK: Int, dS: Int,
                 qScale: Float) {
        var a = ScanArgs(nV: UInt32(nV), nK: UInt32(nK), dS: UInt32(dS),
                         qScale: qScale)
        let stride = keyDim * MemoryLayout<Float>.stride
        grid1D("gdn_scan", nV * dS) { e in
            e.setBuffer(convOut, offset: 0, index: 0)
            e.setBuffer(convOut, offset: stride, index: 1)
            e.setBuffer(convOut, offset: 2 * stride, index: 2)
            e.setBuffer(g, offset: 0, index: 3)
            e.setBuffer(beta, offset: 0, index: 4)
            e.setBuffer(S, offset: 0, index: 5)
            e.setBuffer(o, offset: 0, index: 6)
            e.setBytes(&a, length: MemoryLayout<ScanArgs>.stride, index: 7)
        }
    }

    // ---- attention -------------------------------------------------------
    func splitQGate(qFull: MTLBuffer, q: MTLBuffer, gate: MTLBuffer, hd: Int,
                    nH: Int) {
        var a = SplitArgs(hd: UInt32(hd), nH: UInt32(nH))
        grid1D("split_qgate", hd * nH) { e in
            e.setBuffer(qFull, offset: 0, index: 0)
            e.setBuffer(q, offset: 0, index: 1)
            e.setBuffer(gate, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<SplitArgs>.stride, index: 3)
        }
    }

    func kvAppend(kCur: MTLBuffer, vCur: MTLBuffer, K: MTLBuffer, V: MTLBuffer,
                  kvDim: Int, pos: Int) {
        var a = KVArgs(kvDim: UInt32(kvDim), pos: UInt32(pos))
        grid1D("kv_append", kvDim) { e in
            e.setBuffer(kCur, offset: 0, index: 0)
            e.setBuffer(vCur, offset: 0, index: 1)
            e.setBuffer(K, offset: 0, index: 2)
            e.setBuffer(V, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<KVArgs>.stride, index: 4)
        }
    }

    // Paged attention: kAddr/vAddr are the bindless page-address tables, pages
    // are the live page buffers to make resident. A barrier orders the tail
    // append (directly bound, hazard-tracked) before the bindless page reads.
    func attnPaged(q: MTLBuffer, kAddr: MTLBuffer, vAddr: MTLBuffer,
                   pages: [MTLResource], gate: MTLBuffer, out: MTLBuffer,
                   hd: Int, nH: Int, nKV: Int, T: Int, kvDim: Int, P: Int,
                   scale: Float, gated: Int, lo: Int = 0) {
        precondition(hd % 4 == 0 && hd <= 512,
                     "attn_head: half4 V slices need hd % 4 == 0, and the "
                     + "four-slice accumulator caps head dim at 512")
        var a = AttnArgs(hd: UInt32(hd), nH: UInt32(nH), nKV: UInt32(nKV),
                         T: UInt32(T), kvDim: UInt32(kvDim), P: UInt32(P),
                         scale: scale, gated: UInt32(gated), lo: UInt32(lo))
        e.memoryBarrier(scope: .buffers)
        e.useResources(pages, usage: .read)
        // Flash attention: threadgroup memory is qs[hd] + redM/redL[4] +
        // redAcc[4*hd] = 5*hd + 8 floats, independent of the context length T.
        let tgmem = (5 * hd + 8) * MemoryLayout<Float>.stride
        groups("attn_head", nH, tpg: 128, tgmem: tgmem) {
            e in
            e.setBuffer(q, offset: 0, index: 0)
            e.setBuffer(kAddr, offset: 0, index: 1)
            e.setBuffer(vAddr, offset: 0, index: 2)
            e.setBuffer(gate, offset: 0, index: 3)
            e.setBuffer(out, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<AttnArgs>.stride, index: 5)
        }
    }

    // ---- batched (prefill) dispatches ------------------------------------
    func embedBatch(ids: MTLBuffer, weightOff: WeightRef, out: MTLBuffer,
                    nEmbd: Int, N: Int, type: GGUFType = .q2_0) {
        var a = EmbedArgs(woff: weightOff.local, K: UInt32(nEmbd),
                          M: UInt32(N),
                          rowBytes: UInt32(GGUF.rowByteCount(type, nEmbd)))
        let name: String
        switch type {
        case .q2_0: name = "embed_batch"
        case .q4_0: name = "q4_0_embed_batch"
        case .q8_0: name = "q8_0_embed_batch"
        case .bf16: name = "bf16_embed_batch"
        case .f16: name = "f16_embed_batch"
        case .f32: name = "f32_embed_batch"
        default:
            fatalError("MetalEnc.embedBatch: no kernel for \(type)")
        }
        grid1D(name, N * nEmbd) { e in
            e.setBuffer(weightOff.buf, offset: 0, index: 0)
            e.setBuffer(ids, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<EmbedArgs>.stride, index: 3)
        }
    }

    func gdnConvBatch(qkvMixN: MTLBuffer, convState: MTLBuffer,
                      cwOff: WeightRef, outN: MTLBuffer, convDim: Int,
                      dConv: Int, N: Int, ring: StateRing = .inPlace) {
        var a = ConvBatchArgs(cwOff: cwOff.local, convDim: UInt32(convDim),
                              dConv: UInt32(dConv), N: UInt32(N),
                              slot0: UInt32(ring.slot0),
                              slots: UInt32(ring.slots),
                              stateElems: UInt32(convDim * (dConv - 1)))
        grid1D("gdn_conv_batch", convDim) { e in
            e.setBuffer(qkvMixN, offset: 0, index: 0)
            e.setBuffer(convState, offset: 0, index: 1)
            e.setBuffer(cwOff.buf, offset: 0, index: 2)
            e.setBuffer(outN, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<ConvBatchArgs>.stride,
                       index: 4)
        }
    }

    // The register-resident simdgroup-cooperative scan (state held across the
    // N-loop) is the default; LLM_GDN_SCAN2=0 falls back to the per-column
    // device-memory scan for an A/B.
    static let gdnScan2 =
        ProcessInfo.processInfo.environment["LLM_GDN_SCAN2"] != "0"

    func gdnScanBatch(convOutN: MTLBuffer, keyDim: Int, valueDim: Int,
                      convDim: Int, gN: MTLBuffer, betaN: MTLBuffer,
                      S: MTLBuffer, oN: MTLBuffer, nV: Int, nK: Int, dS: Int,
                      qScale: Float, N: Int, ring: StateRing = .inPlace) {
        var a = ScanBatchArgs(nV: UInt32(nV), nK: UInt32(nK), dS: UInt32(dS),
                              qScale: qScale, N: UInt32(N),
                              convDim: UInt32(convDim), keyDim: UInt32(keyDim),
                              valueDim: UInt32(valueDim),
                              slot0: UInt32(ring.slot0),
                              slots: UInt32(ring.slots),
                              stateElems: UInt32(nV * dS * dS))
        let bind = { (e: MTLComputeCommandEncoder) in
            e.setBuffer(convOutN, offset: 0, index: 0)
            e.setBuffer(gN, offset: 0, index: 1)
            e.setBuffer(betaN, offset: 0, index: 2)
            e.setBuffer(S, offset: 0, index: 3)
            e.setBuffer(oN, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<ScanBatchArgs>.stride,
                       index: 5)
        }
        // scan2 holds the dS x dS state resident in 32 lanes x KS=dS/32
        // register slices, so it needs dS a multiple of 32 and <= 256
        // (ls[8]); any other
        // state size falls back to the per-column device-memory scan.
        if MetalEnc.gdnScan2 && dS % 32 == 0 && dS <= 256 {
            // A threadgroup's 4 simdgroups cover 4 output columns, so the grid
            // spans dS/4 column-groups x nV heads, 128 threads (4 simdgroups)
            // each.
            push("gdn_scan_batch2")
            let pipe = try! ctx.pipeline("gdn_scan_batch2")
            e.setComputePipelineState(pipe)
            bind(e)
            e.dispatchThreadgroups(
                MTLSize(width: dS / 4, height: nV, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1,
                                               depth: 1))
            pop()
        } else {
            grid1D("gdn_scan_batch", nV * dS, bind)
        }
    }

    func splitQGateBatch(qFullN: MTLBuffer, qN: MTLBuffer, gateN: MTLBuffer,
                         hd: Int, nH: Int, N: Int) {
        var a = SplitArgs(hd: UInt32(hd), nH: UInt32(nH))
        var nn = UInt32(N)
        grid1D("split_qgate_batch", N * hd * nH) { e in
            e.setBuffer(qFullN, offset: 0, index: 0)
            e.setBuffer(qN, offset: 0, index: 1)
            e.setBuffer(gateN, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<SplitArgs>.stride, index: 3)
            e.setBytes(&nn, length: 4, index: 4)
        }
    }

    // Gemma's rope for a whole chunk. Its pairing is (j, j + headDim/2) with
    // only the first `rotated` pairs real, which ropeBatch does not reproduce.
    func ropeGemmaBatch(x: MTLBuffer, headDim: Int, nHead: Int, rotated: Int,
                        base: Float, basePos: Int, N: Int) {
        var a = RopeGemmaBatchArgs(headDim: UInt32(headDim),
                                   nHead: UInt32(nHead),
                                   rotated: UInt32(rotated), base: base,
                                   basePos: UInt32(basePos), N: UInt32(N))
        grid1D("rope_gemma_batch", N * nHead * rotated) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RopeGemmaBatchArgs>.stride,
                       index: 1)
        }
    }

    func ropeBatch(x: MTLBuffer, headDim: Int, nHead: Int, nRot: Int,
                   base: Float, basePos: Int, N: Int) {
        var a = RopeBatchArgs(headDim: UInt32(headDim), nHead: UInt32(nHead),
                              nRot: UInt32(nRot), base: base,
                              basePos: UInt32(basePos), N: UInt32(N))
        grid1D("rope_batch", N * nHead * (nRot / 2)) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RopeBatchArgs>.stride,
                       index: 1)
        }
    }

    // Interleaved M-RoPE with per-token 3D positions ([N][3] i32); text
    // (t==h==w) degenerates to ropeBatch, image spans diverge.
    func ropeMBatch(x: MTLBuffer, pos3: MTLBuffer, headDim: Int, nHead: Int,
                    nRot: Int, base: Float, N: Int) {
        var a = RopeMBatchArgs(headDim: UInt32(headDim), nHead: UInt32(nHead),
                               nRot: UInt32(nRot), base: base, N: UInt32(N))
        grid1D("rope_mrope_batch", N * nHead * (nRot / 2)) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(pos3, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<RopeMBatchArgs>.stride,
                       index: 2)
        }
    }

    func kvAppendBatch(kCurN: MTLBuffer, vCurN: MTLBuffer, kAddr: MTLBuffer,
                       vAddr: MTLBuffer, pages: [MTLResource], kvDim: Int,
                       basePos: Int, P: Int, N: Int) {
        var a = KVBatchArgs(kvDim: UInt32(kvDim), basePos: UInt32(basePos),
                            P: UInt32(P), N: UInt32(N))
        e.useResources(pages, usage: .write)
        grid1D("kv_append_batch", N * kvDim) { e in
            e.setBuffer(kCurN, offset: 0, index: 0)
            e.setBuffer(vCurN, offset: 0, index: 1)
            e.setBuffer(kAddr, offset: 0, index: 2)
            e.setBuffer(vAddr, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<KVBatchArgs>.stride, index: 4)
        }
    }

    // ---- ViT (vision tower) dispatches -----------------------------------
    // f16-weight simdgroup GEMM: out[N,M] = X[N,K] @ W[M,K]^T. W is one of
    // MetalViT's dequanted half buffers (ggml-native [out][in] order); the
    // kernel guards the K tail, so any ViT K works.
    func f16Gemm(_ w: MTLBuffer, X: MTLBuffer, out: MTLBuffer, K: Int, M: Int,
                 N: Int) {
        var a = F16WArgs(K: UInt32(K), M: UInt32(M))
        var nn = UInt32(N)
        // The bounds-checked spill path reuses shmem as f32 temp (8192 B); a
        // fully in-bounds tile needs only the half tiles (6144 B).
        let fullTile = M % 64 == 0 && N % 32 == 0
        // As in `gemm`: a simdgroup-matrix kernel, so prewarm skips it where
        // the hardware cannot build it. The guard is three frames up in
        // MetalBackend.extendVision, which refuses without matrix units
        // before MetalViT is ever constructed -- said here because that is
        // far enough away to be edited out without anyone noticing.
        precondition(ctx.matrixUnits,
                     "MetalEnc.f16Gemm: f16w_gemm_mm needs simdgroup matrix "
                     + "units; this GPU has none, so MetalViT should never "
                     + "have been built (see MetalBackend.supportsVision)")
        push("f16w_gemm_mm")
        let pipe = try! ctx.pipeline("f16w_gemm_mm")
        e.setComputePipelineState(pipe)
        e.setBuffer(w, offset: 0, index: 0)
        e.setBuffer(X, offset: 0, index: 1)
        e.setBuffer(out, offset: 0, index: 2)
        e.setBytes(&a, length: MemoryLayout<F16WArgs>.stride, index: 3)
        e.setBytes(&nn, length: 4, index: 4)
        e.setThreadgroupMemoryLength(fullTile ? 6144 : 8192, index: 0)
        e.dispatchThreadgroups(
            MTLSize(width: (N + 31) / 32, height: (M + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        pop()
    }

    // Pre-LN layernorm with bias, x -> y (x survives for the residual).
    func layerNorm(x: MTLBuffer, w: MTLBuffer, b: MTLBuffer, y: MTLBuffer,
                   n: Int, rows: Int, eps: Float) {
        var a = LNArgs(n: UInt32(n), eps: eps)
        groups("vit_layernorm", rows, tpg: 256, tgmem: 256) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(w, offset: 0, index: 1)
            e.setBuffer(b, offset: 0, index: 2)
            e.setBuffer(y, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<LNArgs>.stride, index: 4)
        }
    }

    func addBiasRows(x: MTLBuffer, bias: MTLBuffer, m: Int, rows: Int) {
        var mm = UInt32(m)
        var total = UInt32(rows * m)
        grid1D("add_bias_rows", rows * m) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(bias, offset: 0, index: 1)
            e.setBytes(&mm, length: 4, index: 2)
            e.setBytes(&total, length: 4, index: 3)
        }
    }

    func gelu(x: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("gelu_tanh", n) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&nn, length: 4, index: 1)
        }
    }

    // Vision M-RoPE on the q (off 0) or k (off e) span of the fused qkv rows,
    // from the host-precomputed per-slot tables.
    func visionRope(x: MTLBuffer, cos: MTLBuffer, sin: MTLBuffer,
                    rowStride: Int, off: Int, headDim: Int, nHead: Int,
                    N: Int) {
        var a = VRopeArgs(rowStride: UInt32(rowStride), off: UInt32(off),
                          headDim: UInt32(headDim), nHead: UInt32(nHead),
                          N: UInt32(N))
        grid1D("vit_rope", N * nHead * (headDim / 2)) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(cos, offset: 0, index: 1)
            e.setBuffer(sin, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<VRopeArgs>.stride, index: 3)
        }
    }

    // Flash-style bidirectional attention over the fused qkv rows: one
    // threadgroup per (head, tile of 16 queries), each simdgroup one query,
    // the K/V tile (TK=32 keys) + the tile's queries staged in threadgroup
    // memory (TK=32 keys, f32). The 4 x 32 lane slice caps hd at 128.
    func visionAttn(qkv: MTLBuffer, out: MTLBuffer, n: Int, embd: Int,
                    hd: Int, nHead: Int, scale: Float) {
        precondition(hd <= 128, "vit_attn lane slice caps head dim at 128")
        var a = VAttnArgs(n: UInt32(n), e: UInt32(embd), hd: UInt32(hd),
                          nHead: UInt32(nHead), scale: scale)
        let tq = 16
        push("vit_attn")
        let pipe = try! ctx.pipeline("vit_attn")
        e.setComputePipelineState(pipe)
        e.setBuffer(qkv, offset: 0, index: 0)
        e.setBuffer(out, offset: 0, index: 1)
        e.setBytes(&a, length: MemoryLayout<VAttnArgs>.stride, index: 2)
        e.setThreadgroupMemoryLength(
            ((2 * 32 + tq) * hd * MemoryLayout<Float>.stride + 15) & ~15,
            index: 0)
        e.dispatchThreadgroups(
            MTLSize(width: (n + tq - 1) / tq, height: nHead, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tq * 32, height: 1,
                                           depth: 1))
        pop()
    }

    // `window` bounds how far back each query row reads (0 = the whole
    // history). Gemma windows 28 of its 35 layers at 512, and in a batch every
    // row has its own start, so the bound is computed per row in the kernel
    // rather than passed as one `lo`.
    // `blocks` is the per-row vision-block table (two absolute uints per chunk
    // row); nil means this caller has none and takes the shared empty one. The
    // matrix kernel reads a whole FA_Q=8 tile of rows whatever N is, so the
    // table has to answer 8 rows past the batch. [vision-block]
    func attnBatch(qN: MTLBuffer, kAddr: MTLBuffer, vAddr: MTLBuffer,
                   pages: [MTLResource], gateN: MTLBuffer, outN: MTLBuffer,
                   hd: Int, nH: Int, nKV: Int, kvDim: Int, P: Int,
                   scale: Float,
                   basePos: Int, N: Int, gated: Int, window: Int = 0,
                   blocks: MTLBuffer? = nil,
                   matrix: Bool = MetalEnc.attnMatrix) {
        precondition(hd % 4 == 0 && hd <= 512,
                     "attn_batch: half4 V slices need hd % 4 == 0, and the "
                     + "four-slice accumulator caps head dim at 512")
        var a = AttnBatchArgs(hd: UInt32(hd), nH: UInt32(nH), nKV: UInt32(nKV),
                              kvDim: UInt32(kvDim), P: UInt32(P), scale: scale,
                              basePos: UInt32(basePos), N: UInt32(N),
                              gated: UInt32(gated), window: UInt32(window))
        let blk = blocks ?? ctx.noBlocks(N + 8)
        e.memoryBarrier(scope: .buffers)
        e.useResources(pages, usage: .read)
        // The matrix-unit kernel where the GPU and the geometry allow it: it
        // needs the 8x8 units (absent below Apple7) and a head dim that splits
        // into 8-wide tiles four ways, one slice per simdgroup.
        //
        // AND a page that holds a whole multiple of 8 positions. The kernel
        // reads an 8-key block as ONE strided simdgroup_load, which is only
        // contiguous inside a single page -- the self-test's P=4 pool proves
        // it, and got 0/12 before this term was added. Every shipping pool is
        // 512. Everything else takes the scalar body. [flash-attn-mm]
        let mm = matrix && ctx.matrixUnits
            && hd % 8 == 0 && (hd / 4) % 8 == 0 && P % 8 == 0
        if mm {
            // qs[8*hd half] | s[8*32 f32] | p[8*32 half] | m[8] | l[8]
            // | corr[64 half]
            let tgmem = 8 * hd * 2 + 256 * 4 + 256 * 2 + 8 * 4 + 8 * 4 + 64 * 2
            let tiles = (N + 7) / 8
            groups("attn_batch_mm", tiles * nH, tpg: 128, tgmem: tgmem) { e in
                e.setBuffer(qN, offset: 0, index: 0)
                e.setBuffer(kAddr, offset: 0, index: 1)
                e.setBuffer(vAddr, offset: 0, index: 2)
                e.setBuffer(gateN, offset: 0, index: 3)
                e.setBuffer(outN, offset: 0, index: 4)
                e.setBytes(&a, length: MemoryLayout<AttnBatchArgs>.stride,
                           index: 5)
                e.setBuffer(blk, offset: 0, index: 6)
            }
        } else {
            // Flash attention: threadgroup memory is O(hd) (qs + redM/redL +
            // redAcc), independent of the causal length -- so a long prefill
            // chunk no longer risks blowing the threadgroup budget.
            let tgmem = (5 * hd + 8) * MemoryLayout<Float>.stride
            groups("attn_batch", N * nH, tpg: 128, tgmem: tgmem) { e in
                e.setBuffer(qN, offset: 0, index: 0)
                e.setBuffer(kAddr, offset: 0, index: 1)
                e.setBuffer(vAddr, offset: 0, index: 2)
                e.setBuffer(gateN, offset: 0, index: 3)
                e.setBuffer(outN, offset: 0, index: 4)
                e.setBytes(&a, length: MemoryLayout<AttnBatchArgs>.stride,
                           index: 5)
                e.setBuffer(blk, offset: 0, index: 6)
            }
        }
    }
}

// Swift mirrors of the MSL kargs structs (same field order/types). Weight byte
// offsets are UInt64 (the GGUF weight buffer exceeds 4 GB) and lead each
// struct
// for 8-byte alignment.
struct NormArgs { var woff: UInt64; var n: UInt32; var eps: Float }
struct RowArgs {
    var woff: UInt64; var d: UInt32; var xoff: UInt32; var eps: Float
}
struct RowBatchArgs {
    var d: UInt32; var rowsPerTok: UInt32; var tokStride: UInt32
    var eps: Float
}
struct RopeGemmaBatchArgs {
    var headDim: UInt32; var nHead: UInt32; var rotated: UInt32
    var base: Float; var basePos: UInt32; var N: UInt32
}
struct RopeArgs {
    var headDim: UInt32; var nHead: UInt32; var nRot: UInt32
    var base: Float; var pos: UInt32
}
struct GeluMulArgs { var n: UInt32; var boff: UInt32 }
struct GeluMulRowsArgs {
    var n: UInt32; var rows: UInt32; var aStride: UInt32
    var bStride: UInt32; var boff: UInt32
}
struct GVAttnArgs { var n: UInt32; var hd: UInt32; var nHead: UInt32 }
struct GVRopeArgs {
    var rowStride: UInt32; var off: UInt32; var headDim: UInt32
    var nHead: UInt32; var N: UInt32
}
struct RopeGemmaArgs {
    var headDim: UInt32; var nHead: UInt32; var rotated: UInt32
    var base: Float; var pos: UInt32
}
struct FmaArgs { var n: UInt32; var s: Float }
struct SrqArgs { var n: UInt32; var s: Float }
struct GluArgs { var n: UInt32; var e: UInt32 }
struct DwArgs { var n: UInt32; var e: UInt32; var k: UInt32 }
struct HeadScaleArgs { var n: UInt32; var e: UInt32; var hd: UInt32 }
struct AAttnArgs {
    var n: UInt32; var e: UInt32; var heads: UInt32; var hd: UInt32
    var chunk: UInt32; var ctx: UInt32; var past: UInt32; var cap: Float
}
struct GateArgs { var dtOff: UInt64; var aOff: UInt64; var nV: UInt32 }
struct ConvArgs { var cwOff: UInt64; var convDim: UInt32; var dConv: UInt32 }
struct ScanArgs {
    var nV: UInt32; var nK: UInt32; var dS: UInt32; var qScale: Float
}
struct SplitArgs { var hd: UInt32; var nH: UInt32 }
struct AttnArgs {
    var hd: UInt32; var nH: UInt32; var nKV: UInt32
    var T: UInt32; var kvDim: UInt32; var P: UInt32; var scale: Float
    var gated: UInt32; var lo: UInt32
}
struct KVArgs { var kvDim: UInt32; var pos: UInt32 }
struct ConvBatchArgs {
    var cwOff: UInt64; var convDim: UInt32; var dConv: UInt32; var N: UInt32
    var slot0: UInt32; var slots: UInt32; var stateElems: UInt32
}
struct ScanBatchArgs {
    var nV: UInt32; var nK: UInt32; var dS: UInt32; var qScale: Float
    var N: UInt32; var convDim: UInt32; var keyDim: UInt32
    var valueDim: UInt32
    var slot0: UInt32; var slots: UInt32; var stateElems: UInt32
}
struct RopeBatchArgs {
    var headDim: UInt32; var nHead: UInt32; var nRot: UInt32
    var base: Float; var basePos: UInt32; var N: UInt32
}
struct RopeMBatchArgs {
    var headDim: UInt32; var nHead: UInt32; var nRot: UInt32
    var base: Float; var N: UInt32
}
struct KVBatchArgs {
    var kvDim: UInt32; var basePos: UInt32; var P: UInt32; var N: UInt32
}
struct AttnBatchArgs {
    var hd: UInt32; var nH: UInt32; var nKV: UInt32; var kvDim: UInt32
    var P: UInt32; var scale: Float; var basePos: UInt32; var N: UInt32
    var gated: UInt32; var window: UInt32
}
struct EmbedArgs {
    var woff: UInt64; var K: UInt32; var M: UInt32; var rowBytes: UInt32
}
struct F16WArgs { var K: UInt32; var M: UInt32 }
struct LNArgs { var n: UInt32; var eps: Float }
struct VRopeArgs {
    var rowStride: UInt32; var off: UInt32; var headDim: UInt32
    var nHead: UInt32; var N: UInt32
}
struct VAttnArgs {
    var n: UInt32; var e: UInt32; var hd: UInt32; var nHead: UInt32
    var scale: Float
}

// Typed one-token dispatch surface: each method encodes ONE kernel onto the
// shared command encoder with the buffer bindings and kargs its MSL counterpart
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

struct MetalEnc {
    let ctx: MetalContext
    let e: MTLComputeCommandEncoder

    // LLM_GPU_LABELS=1 wraps each dispatch in a debug group named for its
    // kernel, so an Instruments Metal capture names every GPU interval (they all
    // run through one unlabeled compute encoder otherwise). No numeric effect.
    static let labels =
        ProcessInfo.processInfo.environment["LLM_GPU_LABELS"] == "1"
    private func push(_ n: String) { if MetalEnc.labels { e.pushDebugGroup(n) } }
    private func pop() { if MetalEnc.labels { e.popDebugGroup() } }

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
    private func groups(_ name: String, _ groups: Int, tpg: Int, tgmem: Int = 0,
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

    // ---- Q2_0 mat-vec: out[M] = W @ x  (W ne0=K, ne1=M at byte off) -----
    func gemv(_ w: GGUFTensor, x: MTLBuffer, out: MTLBuffer, off: UInt64,
              xOff: Int = 0) {
        let k = w.dims[0], m = w.dims[1]
        var a = GemvArgs(woff: off, K: UInt32(k), M: UInt32(m))
        // one simdgroup (32 threads) per 8 output rows (NR0=8 in the kernel).
        groups("q2_0_gemv", (m + 7) / 8, tpg: 32) { e in
            e.setBuffer(ctx.weights, offset: 0, index: 0)
            e.setBuffer(x, offset: xOff, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 3)
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
    // LLM_F16_TILES=0 restores the f32 tiles for an A/B.
    static let f16Tiles =
        ProcessInfo.processInfo.environment["LLM_F16_TILES"] != "0"

    func gemm(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer, off: UInt64,
              N: Int) {
        let k = w.dims[0], m = w.dims[1]
        var a = GemvArgs(woff: off, K: UInt32(k), M: UInt32(m))
        var nn = UInt32(N)
        // The half-tile spill path needs 8192B f32 temp; a fully in-bounds tile
        // (M multiple of 64, N of 32) never spills, so 6144 suffices.
        let fullTile = m % 64 == 0 && N % 32 == 0
        let name = MetalEnc.f16Tiles ? "q2_0_gemm_mm_h" : "q2_0_gemm_mm"
        let tgmem = MetalEnc.f16Tiles ? (fullTile ? 6144 : 8192) : 12288
        push(name)
        let pipe = try! ctx.pipeline(name)
        e.setComputePipelineState(pipe)
        e.setBuffer(ctx.weights, offset: 0, index: 0)
        e.setBuffer(X, offset: 0, index: 1)
        e.setBuffer(out, offset: 0, index: 2)
        e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 3)
        e.setBytes(&nn, length: 4, index: 4)
        e.setThreadgroupMemoryLength(tgmem, index: 0)
        e.dispatchThreadgroups(
            MTLSize(width: (N + 31) / 32, height: (m + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        pop()
    }

    // ---- Q2_0 row dequant: out[n] = dequant(weight row at off) -----------
    func dequantRow(weightOff: UInt64, out: MTLBuffer, n: Int) {
        var a = GemvArgs(woff: weightOff, K: UInt32(n), M: UInt32(n))
        grid1D("q2_0_dequant_row", n) { e in
            e.setBuffer(ctx.weights, offset: 0, index: 0)
            e.setBuffer(out, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 2)
        }
    }

    // ---- norms -----------------------------------------------------------
    func rmsnorm(x: MTLBuffer, weightOff: UInt64, out: MTLBuffer, n: Int,
                 eps: Float) {
        var a = NormArgs(woff: weightOff, n: UInt32(n), eps: eps)
        groups("rmsnorm", 1, tpg: 256, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(ctx.weights, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<NormArgs>.stride, index: 3)
        }
    }

    // Batched whole-vector rmsnorm to a separate output (attn_norm/post_norm in
    // prefill): rows tokens, each of length n, x -> y.
    func rmsnormBatch(x: MTLBuffer, weightOff: UInt64, y: MTLBuffer, n: Int,
                      rows: Int, eps: Float) {
        var a = NormArgs(woff: weightOff, n: UInt32(n), eps: eps)
        groups("rmsnorm_batch", rows, tpg: 256, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(ctx.weights, offset: 0, index: 1)
            e.setBuffer(y, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<NormArgs>.stride, index: 3)
        }
    }

    func rmsnormRows(x: MTLBuffer, xoff: UInt32, d: Int, rows: Int,
                     weightOff: UInt64, eps: Float) {
        var a = RowArgs(woff: weightOff, d: UInt32(d), xoff: xoff, eps: eps)
        groups("rmsnorm_rows", rows, tpg: 128, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(ctx.weights, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<RowArgs>.stride, index: 2)
        }
    }

    func l2normRows(x: MTLBuffer, xoff: UInt32, d: Int, rows: Int, eps: Float) {
        var a = RowArgs(woff: 0, d: UInt32(d), xoff: xoff, eps: eps)
        groups("l2norm_rows", rows, tpg: 128, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RowArgs>.stride, index: 1)
        }
    }

    // Batched l2norm: `rowsPerTok` rows of length d for each of `tokens` tokens,
    // token n based at n*tokStride (q|k across a whole prefill chunk in ONE
    // dispatch, vs 2*tokens tiny per-token l2normRows calls).
    func l2normRowsBatch(x: MTLBuffer, d: Int, rowsPerTok: Int, tokStride: Int,
                         tokens: Int, eps: Float) {
        var a = RowBatchArgs(d: UInt32(d), rowsPerTok: UInt32(rowsPerTok),
                             tokStride: UInt32(tokStride), eps: eps)
        groups("l2norm_rows_batch", tokens * rowsPerTok, tpg: 128, tgmem: 128) {
            e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RowBatchArgs>.stride, index: 1)
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
    func gdnGate(bPre: MTLBuffer, aPre: MTLBuffer, dtOff: UInt64, aOff: UInt64,
                 beta: MTLBuffer, g: MTLBuffer, nV: Int, count: Int) {
        var a = GateArgs(dtOff: dtOff, aOff: aOff, nV: UInt32(nV))
        var total = UInt32(count)
        grid1D("gdn_gate", count) { e in
            e.setBuffer(bPre, offset: 0, index: 0)
            e.setBuffer(aPre, offset: 0, index: 1)
            e.setBuffer(ctx.weights, offset: 0, index: 2)
            e.setBuffer(beta, offset: 0, index: 3)
            e.setBuffer(g, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<GateArgs>.stride, index: 5)
            e.setBytes(&total, length: 4, index: 6)
        }
    }

    func gdnConv(qkvMix: MTLBuffer, convState: MTLBuffer, cwOff: UInt64,
                 out: MTLBuffer, convDim: Int, dConv: Int) {
        var a = ConvArgs(cwOff: cwOff, convDim: UInt32(convDim),
                         dConv: UInt32(dConv))
        grid1D("gdn_conv", convDim) { e in
            e.setBuffer(qkvMix, offset: 0, index: 0)
            e.setBuffer(convState, offset: 0, index: 1)
            e.setBuffer(ctx.weights, offset: 0, index: 2)
            e.setBuffer(out, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<ConvArgs>.stride, index: 4)
        }
    }

    // convOut holds [q(keyDim) | k(keyDim) | v(valueDim)]; bind the three spans
    // by byte offset (keyDim*4 is 256-aligned, safe for setBuffer offset).
    func gdnScan(convOut: MTLBuffer, keyDim: Int, g: MTLBuffer, beta: MTLBuffer,
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
                   scale: Float, gated: Int) {
        precondition(hd % 4 == 0 && hd <= 256,
                     "attn_head: half4 V slices need hd % 4 == 0, and the "
                     + "two-slice accumulator caps head dim at 256")
        var a = AttnArgs(hd: UInt32(hd), nH: UInt32(nH), nKV: UInt32(nKV),
                         T: UInt32(T), kvDim: UInt32(kvDim), P: UInt32(P),
                         scale: scale, gated: UInt32(gated))
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
    func embedBatch(ids: MTLBuffer, weightOff: UInt64, out: MTLBuffer,
                    nEmbd: Int, N: Int) {
        var a = GemvArgs(woff: weightOff, K: UInt32(nEmbd), M: UInt32(N))
        grid1D("embed_batch", N * nEmbd) { e in
            e.setBuffer(ctx.weights, offset: 0, index: 0)
            e.setBuffer(ids, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 3)
        }
    }

    func gdnConvBatch(qkvMixN: MTLBuffer, convState: MTLBuffer, cwOff: UInt64,
                      outN: MTLBuffer, convDim: Int, dConv: Int, N: Int) {
        var a = ConvBatchArgs(cwOff: cwOff, convDim: UInt32(convDim),
                              dConv: UInt32(dConv), N: UInt32(N))
        grid1D("gdn_conv_batch", convDim) { e in
            e.setBuffer(qkvMixN, offset: 0, index: 0)
            e.setBuffer(convState, offset: 0, index: 1)
            e.setBuffer(ctx.weights, offset: 0, index: 2)
            e.setBuffer(outN, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<ConvBatchArgs>.stride, index: 4)
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
                      qScale: Float, N: Int) {
        var a = ScanBatchArgs(nV: UInt32(nV), nK: UInt32(nK), dS: UInt32(dS),
                              qScale: qScale, N: UInt32(N),
                              convDim: UInt32(convDim), keyDim: UInt32(keyDim),
                              valueDim: UInt32(valueDim))
        let bind = { (e: MTLComputeCommandEncoder) in
            e.setBuffer(convOutN, offset: 0, index: 0)
            e.setBuffer(gN, offset: 0, index: 1)
            e.setBuffer(betaN, offset: 0, index: 2)
            e.setBuffer(S, offset: 0, index: 3)
            e.setBuffer(oN, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<ScanBatchArgs>.stride, index: 5)
        }
        // scan2 holds the dS x dS state resident in 32 lanes x KS=dS/32 register
        // slices, so it needs dS a multiple of 32 and <= 256 (ls[8]); any other
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
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
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

    func ropeBatch(x: MTLBuffer, headDim: Int, nHead: Int, nRot: Int,
                   base: Float, basePos: Int, N: Int) {
        var a = RopeBatchArgs(headDim: UInt32(headDim), nHead: UInt32(nHead),
                              nRot: UInt32(nRot), base: base,
                              basePos: UInt32(basePos), N: UInt32(N))
        grid1D("rope_batch", N * nHead * (nRot / 2)) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RopeBatchArgs>.stride, index: 1)
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

    func attnBatch(qN: MTLBuffer, kAddr: MTLBuffer, vAddr: MTLBuffer,
                   pages: [MTLResource], gateN: MTLBuffer, outN: MTLBuffer,
                   hd: Int, nH: Int, nKV: Int, kvDim: Int, P: Int, scale: Float,
                   basePos: Int, N: Int, gated: Int) {
        precondition(hd % 4 == 0 && hd <= 256,
                     "attn_batch: half4 V slices need hd % 4 == 0, and the "
                     + "two-slice accumulator caps head dim at 256")
        var a = AttnBatchArgs(hd: UInt32(hd), nH: UInt32(nH), nKV: UInt32(nKV),
                              kvDim: UInt32(kvDim), P: UInt32(P), scale: scale,
                              basePos: UInt32(basePos), N: UInt32(N),
                              gated: UInt32(gated))
        e.memoryBarrier(scope: .buffers)
        e.useResources(pages, usage: .read)
        // Flash attention: threadgroup memory is O(hd) (qs + redM/redL +
        // redAcc), independent of the causal length -- so a long prefill chunk
        // no longer risks blowing the threadgroup budget.
        let tgmem = (5 * hd + 8) * MemoryLayout<Float>.stride
        groups("attn_batch", N * nH, tpg: 128, tgmem: tgmem) { e in
            e.setBuffer(qN, offset: 0, index: 0)
            e.setBuffer(kAddr, offset: 0, index: 1)
            e.setBuffer(vAddr, offset: 0, index: 2)
            e.setBuffer(gateN, offset: 0, index: 3)
            e.setBuffer(outN, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<AttnBatchArgs>.stride, index: 5)
        }
    }
}

// Swift mirrors of the MSL kargs structs (same field order/types). Weight byte
// offsets are UInt64 (the GGUF weight buffer exceeds 4 GB) and lead each struct
// for 8-byte alignment.
struct NormArgs { var woff: UInt64; var n: UInt32; var eps: Float }
struct RowArgs { var woff: UInt64; var d: UInt32; var xoff: UInt32; var eps: Float }
struct RowBatchArgs {
    var d: UInt32; var rowsPerTok: UInt32; var tokStride: UInt32; var eps: Float
}
struct RopeArgs {
    var headDim: UInt32; var nHead: UInt32; var nRot: UInt32
    var base: Float; var pos: UInt32
}
struct GateArgs { var dtOff: UInt64; var aOff: UInt64; var nV: UInt32 }
struct ConvArgs { var cwOff: UInt64; var convDim: UInt32; var dConv: UInt32 }
struct ScanArgs { var nV: UInt32; var nK: UInt32; var dS: UInt32; var qScale: Float }
struct SplitArgs { var hd: UInt32; var nH: UInt32 }
struct AttnArgs {
    var hd: UInt32; var nH: UInt32; var nKV: UInt32
    var T: UInt32; var kvDim: UInt32; var P: UInt32; var scale: Float
    var gated: UInt32
}
struct KVArgs { var kvDim: UInt32; var pos: UInt32 }
struct ConvBatchArgs {
    var cwOff: UInt64; var convDim: UInt32; var dConv: UInt32; var N: UInt32
}
struct ScanBatchArgs {
    var nV: UInt32; var nK: UInt32; var dS: UInt32; var qScale: Float
    var N: UInt32; var convDim: UInt32; var keyDim: UInt32; var valueDim: UInt32
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
    var gated: UInt32
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

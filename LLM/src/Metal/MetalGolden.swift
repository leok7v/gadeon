// Byte-exact golden buffers for the Metal kernels: dump each kernel's output
// on fixed inputs, then assert the SAME BYTES after a refactor.
//
// WHY bytes and not cosine. A consolidation claims the kernel performs the
// identical sequence of floating-point operations, and identical operations
// give identical floats -- so the honest acceptance test for "same behaviour,
// less code" is equality, not tolerance. Cosine cannot see a one-ulp change,
// and on the gemma path it cannot see much more: SRQ's rint() is a step
// function, which puts a ~0.99 noise floor under every comparison and would
// swallow a real indexing bug. A byte compare has no floor. If a refactor is
// NOT bit-identical that is a finding to explain -- most likely the source
// shape changed which multiplies the compiler contracts into an FMA -- rather
// than a tolerance to widen.
//
// `gadeon-cli <gguf> --metal-golden <dir>` WRITES the goldens when a case has
// no file yet and COMPARES otherwise, so the workflow is: capture on the
// baseline commit, re-run after each consolidation, expect every line MATCH.
//
// Inputs are deterministic (a hashed LCG, no Date/random) and weights are REAL
// tensors out of the file, because the bugs these kernels have are layout bugs
// and synthetic weights of the right shape would not expose one.
import Foundation
import Metal

public enum MetalGolden {
    // Deterministic pseudo-random f32 in [-1, 1), matching MetalSelfTest.fill
    // so a case can be reproduced from either harness.
    private static func fill(_ n: Int, seed: UInt64) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        var s = seed &+ 0x9E37_79B9_7F4A_7C15
        for i in 0..<n {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let bits = UInt32(truncatingIfNeeded: s >> 33)
            out[i] = Float(bits) / Float(UInt32.max) * 2 - 1
        }
        return out
    }

    // One kernel dispatch and the buffer whose bytes are the golden. Each case
    // rides its OWN command buffer so a GPU fault names the case that caused
    // it rather than the whole batch.
    private struct Case {
        let name: String
        let out: MTLBuffer
        let count: Int
        let encode: (MetalEnc) -> Void
    }

    public static func run(ggufPath: String, dir: String) throws -> String {
        let g = try GGUF(path: ggufPath)
        let ctx = try MetalContext(g)
        try ctx.prewarm()
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)

        // Tensor picks are by TYPE rather than by name, so one harness covers
        // both lineages: the ternary file has f32 norms and Q2_0 matrices, the
        // gemma file adds BF16 norms, Q4_0/Q8_0 matrices, and f32 norms of its
        // own in the audio tower. Sorted keys keep the pick deterministic.
        func first(_ ok: (GGUFTensor) -> Bool) -> GGUFTensor? {
            var found: GGUFTensor? = nil
            for name in g.tensors.keys.sorted() where found == nil {
                let t = g.tensors[name]!
                if ok(t) { found = t }
            }
            return found
        }
        func vector(_ ty: GGUFType) -> GGUFTensor? {
            first { t in
                t.type == ty && t.dims.count == 1 && t.dims[0] >= 256
            }
        }
        // Bounded on both axes: the lm_head is 262144 rows and a golden of it
        // buys nothing a 4096-row matrix does not.
        func matrix(_ ty: GGUFType) -> GGUFTensor? {
            first { t in
                t.type == ty && t.dims.count == 2
                    && t.dims[0] >= 256 && t.dims[1] >= 256
                    && t.dims[1] <= 8192
            }
        }
        func off(_ t: GGUFTensor) -> WeightRef {
            ctx.window(UInt64(t.base - g.map))
        }

        var cases: [Case] = []
        let eps: Float = 1e-6

        // ---- norms ------------------------------------------------------
        // Both weight regimes where the file carries them. The row forms take
        // d = 128 over 4 rows so a weight of any width works.
        for (ty, tag) in [(GGUFType.f32, "f32"), (.bf16, "bf16")] {
            if let w = vector(ty) {
                let n = min(w.dims[0], 2048)
                let rows = 3
                let x1 = ctx.makeF32(fill(n, seed: 11))
                let o1 = ctx.makeF32(n)
                cases.append(Case(name: "rmsnorm_\(tag)", out: o1,
                                  count: n) { f in
                    if ty == .bf16 {
                        f.rmsnormBF16(x: x1, weightOff: off(w), out: o1,
                                      n: n, eps: eps)
                    } else {
                        f.rmsnorm(x: x1, weightOff: off(w), out: o1, n: n,
                                  eps: eps)
                    }
                })
                let x2 = ctx.makeF32(fill(rows * n, seed: 12))
                let o2 = ctx.makeF32(rows * n)
                cases.append(Case(name: "rmsnorm_batch_\(tag)", out: o2,
                                  count: rows * n) { f in
                    if ty == .bf16 {
                        f.rmsnormBatchBF16(x: x2, weightOff: off(w), y: o2,
                                           n: n, rows: rows, eps: eps)
                    } else {
                        f.rmsnormBatch(x: x2, weightOff: off(w), y: o2, n: n,
                                       rows: rows, eps: eps)
                    }
                })
                let d = 128, r = 4
                let x3 = ctx.makeF32(fill(d * r, seed: 13))
                cases.append(Case(name: "rmsnorm_rows_\(tag)", out: x3,
                                  count: d * r) { f in
                    if ty == .bf16 {
                        f.rmsnormRowsBF16(x: x3, xoff: 0, d: d, rows: r,
                                          weightOff: off(w), eps: eps)
                    } else {
                        f.rmsnormRows(x: x3, xoff: 0, d: d, rows: r,
                                      weightOff: off(w), eps: eps)
                    }
                })
            }
        }
        // The weightless reductions. rmsnorm_rows_noweight divides by the RMS
        // and l2norm_rows by the L2 norm, which differ by the 1/sqrt(d) and
        // are the pair most likely to be conflated by a shared helper.
        let d = 128, r = 4
        let xNo = ctx.makeF32(fill(d * r, seed: 14))
        cases.append(Case(name: "rmsnorm_rows_noweight", out: xNo,
                          count: d * r) { f in
            f.rmsnormRowsNoWeight(x: xNo, xoff: 0, d: d, rows: r, eps: eps)
        })
        let xL2 = ctx.makeF32(fill(d * r, seed: 15))
        cases.append(Case(name: "l2norm_rows", out: xL2, count: d * r) { f in
            f.l2normRows(x: xL2, xoff: 0, d: d, rows: r, eps: eps)
        })
        // Strided: 3 tokens x 2 rows of 128 inside a 512-wide token stride, so
        // the token/row index split is exercised rather than a flat run.
        let toks = 3, perTok = 2, stride = 512
        let xL2B = ctx.makeF32(fill(toks * stride, seed: 16))
        cases.append(Case(name: "l2norm_rows_batch", out: xL2B,
                          count: toks * stride) { f in
            f.l2normRowsBatch(x: xL2B, d: d, rowsPerTok: perTok,
                              tokStride: stride, tokens: toks, eps: eps)
        })
        // The ViT LayerNorm carries its weight and bias as plain buffers, so
        // it needs no tensor of any type and runs for every file.
        let lnN = 256, lnRows = 3
        let lnX = ctx.makeF32(fill(lnRows * lnN, seed: 17))
        let lnW = ctx.makeF32(fill(lnN, seed: 18))
        let lnB = ctx.makeF32(fill(lnN, seed: 19))
        let lnY = ctx.makeF32(lnRows * lnN)
        cases.append(Case(name: "vit_layernorm", out: lnY,
                          count: lnRows * lnN) { f in
            f.layerNorm(x: lnX, w: lnW, b: lnB, y: lnY, n: lnN,
                        rows: lnRows, eps: eps)
        })

        // ---- ropes ------------------------------------------------------
        // Both pairings, both regimes. rope_neox pairs (i, i + nRot/2) and
        // rope_gemma pairs (j, j + headDim/2) with only the first `rotated`
        // pairs real -- a helper that unified them on the wrong rule passes
        // one of these and fails the other.
        let hd = 128, nH = 2
        let rNeox = ctx.makeF32(fill(hd * nH, seed: 20))
        cases.append(Case(name: "rope_neox", out: rNeox,
                          count: hd * nH) { f in
            f.rope(x: rNeox, headDim: hd, nHead: nH, nRot: 64, base: 1e6,
                   pos: 37)
        })
        let nBat = 3
        let rBat = ctx.makeF32(fill(nBat * hd * nH, seed: 21))
        cases.append(Case(name: "rope_batch", out: rBat,
                          count: nBat * hd * nH) { f in
            f.ropeBatch(x: rBat, headDim: hd, nHead: nH, nRot: 64, base: 1e6,
                        basePos: 5, N: nBat)
        })
        // rotated < headDim/2 is the proportional regime the full-attention
        // layers use; a kernel that rotates every pair still fills the buffer.
        let rGem = ctx.makeF32(fill(hd * nH, seed: 22))
        cases.append(Case(name: "rope_gemma", out: rGem,
                          count: hd * nH) { f in
            f.ropeGemma(x: rGem, headDim: hd, nHead: nH, rotated: 16,
                        base: 1e4, pos: 37)
        })
        let rGemB = ctx.makeF32(fill(nBat * hd * nH, seed: 23))
        cases.append(Case(name: "rope_gemma_batch", out: rGemB,
                          count: nBat * hd * nH) { f in
            f.ropeGemmaBatch(x: rGemB, headDim: hd, nHead: nH, rotated: 16,
                             base: 1e4, basePos: 5, N: nBat)
        })
        // M-RoPE with DISTINCT t/h/w per token: equal components degenerate to
        // rope_batch, so a golden built from text positions would not pin the
        // interleave at all.
        var pos3: [Int32] = []
        for n in 0..<nBat {
            pos3 += [Int32(n + 1), Int32(n + 4), Int32(n + 9)]
        }
        let mrX = ctx.makeF32(fill(nBat * hd * nH, seed: 24))
        let mrP = ctx.makeF32(nBat * 3)
        pos3.withUnsafeBytes { src in
            mrP.contents().copyMemory(from: src.baseAddress!,
                                      byteCount: src.count)
        }
        cases.append(Case(name: "rope_mrope_batch", out: mrX,
                          count: nBat * hd * nH) { f in
            f.ropeMBatch(x: mrX, pos3: mrP, headDim: hd, nHead: nH, nRot: 64,
                         base: 1e6, N: nBat)
        })
        // The two vision ropes read host-built cos/sin tables. vit_rope pairs
        // across the whole head; gemma_vit_rope pairs WITHIN each axis half.
        let vN = 4
        let vX = ctx.makeF32(fill(vN * hd * nH, seed: 25))
        let vCos = ctx.makeF32(fill(vN * hd, seed: 26))
        let vSin = ctx.makeF32(fill(vN * hd, seed: 27))
        cases.append(Case(name: "vit_rope", out: vX,
                          count: vN * hd * nH) { f in
            f.visionRope(x: vX, cos: vCos, sin: vSin, rowStride: hd * nH,
                         off: 0, headDim: hd, nHead: nH, N: vN)
        })
        let gvX = ctx.makeF32(fill(vN * hd * nH, seed: 28))
        cases.append(Case(name: "gemma_vit_rope", out: gvX,
                          count: vN * hd * nH) { f in
            f.gemmaVisionRope(x: gvX, cos: vCos, sin: vSin,
                              rowStride: hd * nH, headDim: hd, nHead: nH,
                              N: vN)
        })

        // ---- gemv / gemm, per weight type the file carries ---------------
        // The GEMM runs at N=8 and N=33: 33 spills past the 32-column tile, so
        // the bounds-checked threadgroup path runs too. That path is written
        // once per kernel today and is exactly what a template must preserve.
        for ty in [GGUFType.q2_0, .q2_x, .q4_0, .q8_0, .bf16, .f32] {
            if let w = matrix(ty) {
                let K = w.dims[0], M = w.dims[1]
                let tag = "\(ty)"
                let xv = ctx.makeF32(fill(K, seed: 30))
                let ov = ctx.makeF32(M)
                cases.append(Case(name: "gemv_\(tag)", out: ov,
                                  count: M) { f in
                    f.gemv(w, x: xv, out: ov, off: off(w))
                })
                // Only the quantized types have a batched kernel; bf16 and f32
                // go through gemmRows, which is a gemv loop and would only
                // re-measure the case above.
                if ty == .q2_0 || ty == .q2_x || ty == .q4_0 || ty == .q8_0 {
                    for n in [8, 33] {
                        let xm = ctx.makeF32(fill(n * K, seed: 31))
                        let om = ctx.makeF32(n * M)
                        cases.append(Case(name: "gemm_\(tag)_n\(n)", out: om,
                                          count: n * M) { f in
                            f.gemm(w, X: xm, out: om, off: off(w), N: n)
                        })
                    }
                }
            }
        }

        // ---- paged attention --------------------------------------------
        // The two most complex kernels in the file and near-identical twins,
        // so they are exactly where a shared body could diverge unnoticed.
        // A tiny page size (P=8) forces the bindless multi-page gather; hd
        // 256 and 512 are the two gemma geometries and the second is what
        // the four-wide V accumulator was widened for; a non-zero `lo` and a
        // non-zero window exercise the read bound that makes eviction
        // unnecessary; basePos > 0 puts the batch mid-KV, where every
        // absolute index is basePos-relative.
        for hd in [256, 512] {
            let nH = 2, nKV = 1, T = 40, P = 8
            let pool = MetalKVPool(device: ctx.device, P: P, kvDim: hd * nKV)
            pool.appendBatch(T)
            let src = fill(2 * T * hd, seed: 40)
            for t in 0..<T {
                let kp = pool.kPages[t / P].contents()
                    .assumingMemoryBound(to: Float16.self)
                let vp = pool.vPages[t / P].contents()
                    .assumingMemoryBound(to: Float16.self)
                for i in 0..<hd {
                    kp[(t % P) * hd + i] = Float16(src[t * hd + i])
                    vp[(t % P) * hd + i] = Float16(src[T * hd + t * hd + i])
                }
            }
            pool.refreshTable()
            let gate = ctx.makeF32(fill(hd * nH, seed: 41))
            // gated=1 is the ternary hybrid's sigmoid output gate; gemma
            // passes 0 and never reads the buffer. Both arms run here.
            for (lo, gated) in [(0, 0), (12, 1)] {
                let q = ctx.makeF32(fill(hd * nH, seed: 42))
                let o = ctx.makeF32(hd * nH)
                cases.append(Case(name: "attn_head_hd\(hd)_lo\(lo)_g\(gated)",
                                  out: o, count: hd * nH) { f in
                    f.attnPaged(q: q, kAddr: pool.kAddr, vAddr: pool.vAddr,
                                pages: pool.residentPages, gate: gate,
                                out: o, hd: hd, nH: nH, nKV: nKV, T: T,
                                kvDim: hd * nKV, P: P, scale: 0.125,
                                gated: gated, lo: lo)
                })
            }
            let nq = 5
            let gateN = ctx.makeF32(fill(nq * hd * nH, seed: 43))
            for (basePos, window) in [(0, 0), (17, 16)] {
                let qN = ctx.makeF32(fill(nq * hd * nH, seed: 44))
                let oN = ctx.makeF32(nq * hd * nH)
                let nm = "attn_batch_hd\(hd)_p\(basePos)_w\(window)"
                cases.append(Case(name: nm, out: oN,
                                  count: nq * hd * nH) { f in
                    f.attnBatch(qN: qN, kAddr: pool.kAddr, vAddr: pool.vAddr,
                                pages: pool.residentPages, gateN: gateN,
                                outN: oN, hd: hd, nH: nH, nKV: nKV,
                                kvDim: hd * nKV, P: P, scale: 0.125,
                                basePos: basePos, N: nq, gated: 0,
                                window: window)
                })
            }
        }

        // ---- the two tower attentions -----------------------------------
        // Both are CCN-17 bodies with no coverage until now. vit_attn needs no
        // weights at all (it reads a fused qkv buffer), so it runs for every
        // file; gemma_audio_attn needs the tower's geometry, so it runs only
        // where the file carries one.
        //
        // Synthetic q/k/v is right HERE where a real tensor is not: these
        // kernels take plain float buffers from the host rather than decoding
        // a weight layout, and a golden compares a kernel against ITSELF
        // rather than against an oracle.
        let aN = 40, vHd = 64, vHeads = 2
        let vE = vHd * vHeads
        let vQkv = ctx.makeF32(fill(aN * 3 * vE, seed: 50))
        let vOut = ctx.makeF32(aN * vE)
        cases.append(Case(name: "vit_attn", out: vOut,
                          count: aN * vE) { f in
            f.visionAttn(qkv: vQkv, out: vOut, n: aN, embd: vE, hd: vHd,
                         nHead: vHeads, scale: 0.125)
        })
        // Gate on the tower's TENSORS, not on try?: Gemma4AudioConfig is
        // declared throws but force-unwraps its metadata, so on a file
        // without an audio tower it fatals rather than throwing.
        if g.maybe("a.blk.0.attn_q.weight") != nil,
           let ac = try? Gemma4AudioConfig(g) {
            // n must span SEVERAL blocks: the extra-key bug this kernel had
            // is invisible in the first one, where the stray key falls before
            // position zero and is masked anyway.
            let an = 4 * ac.chunk
            let ae = ac.embd
            let aq = ctx.makeF32(fill(an * ae, seed: 51))
            let ak = ctx.makeF32(fill(an * ae, seed: 52))
            let av = ctx.makeF32(fill(an * ae, seed: 53))
            let arel = ctx.makeF32(fill(ac.context * ae, seed: 54))
            let aout = ctx.makeF32(an * ae)
            cases.append(Case(name: "gemma_audio_attn", out: aout,
                              count: an * ae) { f in
                f.gemmaAudioAttn(q: aq, k: ak, v: av, relk: arel, out: aout,
                                 rows: an, width: ae, heads: ac.heads,
                                 hd: ac.headDim, chunk: ac.chunk,
                                 context: ac.context, past: ac.pastHorizon,
                                 cap: ac.logitCap)
            })
        }

        // ---- run, then write or compare ---------------------------------
        var lines = ["  weights in \(ctx.windowMB.count) windows "
                     + "\(ctx.windowMB) MB, GPU caps one at "
                     + "\(ctx.device.maxBufferLength / 1_048_576) MB"]
        var failed = 0
        var wrote = 0
        for c in cases {
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            c.encode(MetalEnc(ctx: ctx, e: e))
            e.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            precondition(cb.error == nil,
                         "\(c.name): \(cb.error!)")
            let bytes = c.count * MemoryLayout<Float>.stride
            let got = Data(bytes: c.out.contents(), count: bytes)
            let file = (dir as NSString)
                .appendingPathComponent("\(c.name).bin")
            let old = try? Data(contentsOf: URL(fileURLWithPath: file))
            if old == nil {
                try got.write(to: URL(fileURLWithPath: file))
                wrote += 1
                lines.append("  WROTE  \(c.name) (\(c.count) floats)")
            } else if old! == got {
                lines.append("  MATCH  \(c.name)")
            } else {
                failed += 1
                lines.append("  DIFFER \(c.name)  \(describe(old!, got))")
            }
        }
        let verdict = failed == 0
            ? (wrote > 0 ? "GOLDEN WRITTEN (\(wrote) new)" : "GOLDEN MATCH")
            : "GOLDEN DIFFER (\(failed) of \(cases.count))"
        return lines.joined(separator: "\n") + "\n" + verdict
    }

    // Where the two buffers first part company, and by how much -- an ulp-sized
    // delta reads as a contraction change, an O(1) one as a real bug.
    private static func describe(_ a: Data, _ b: Data) -> String {
        var out = "sizes \(a.count) vs \(b.count)"
        if a.count == b.count {
            let n = a.count / MemoryLayout<Float>.stride
            let fa = a.withUnsafeBytes { p in
                Array(p.bindMemory(to: Float.self).prefix(n))
            }
            let fb = b.withUnsafeBytes { p in
                Array(p.bindMemory(to: Float.self).prefix(n))
            }
            var firstAt = -1
            var worst: Float = 0
            var count = 0
            for i in 0..<n where fa[i].bitPattern != fb[i].bitPattern {
                if firstAt < 0 { firstAt = i }
                worst = max(worst, abs(fa[i] - fb[i]))
                count += 1
            }
            out = "\(count)/\(n) floats differ, first at \(firstAt) "
                + "(\(fa[max(firstAt, 0)]) vs \(fb[max(firstAt, 0)])), "
                + "maxAbs \(worst)"
        }
        return out
    }
}

// The GGUF block types the gemma-4 repack emits that the ternary path never
// needed: Q4_0 (attention, low MLP, the PLE table), Q8_0 (vision tower), and
// BF16 (every norm). Q2_0 already lives in Quant.swift and is reused as-is.
//
// Layouts, byte for byte as scripts/convert/gemma4gguf/repack_qat.py writes
// them (and as repack_probe.py proves against the QAT checkpoint):
//   Q4_0  32 weights / 18 bytes  { f16 d; u8 qs[16] }  byte j -> elements j
//         (low nibble) and j+16 (high), w = (q - 8) * d
//   Q8_0  32 weights / 34 bytes  { f16 d; i8 qs[32] }  w = q * d
//   BF16  the top 16 bits of the f32 bit pattern
//
// WHY a second matvec family rather than dequant-then-GEMM: the weights are
// the bandwidth, so every read stays quantized and expands in registers, the
// same shape Q2_0.matvec already has.
import Foundation

enum Q4_0 {
    static let qk = 32
    static let blockBytes = 18

    // Dequantize `n` contiguous weights (n % 32 == 0) starting at `base`.
    static func dequant(_ base: UnsafeRawPointer, count n: Int,
                        into out: UnsafeMutablePointer<Float>) {
        var p = base
        var o = 0
        for _ in 0..<(n / qk) {
            let d = Float(p.loadUnaligned(as: Float16.self))
            for j in 0..<16 {
                let b = p.load(fromByteOffset: 2 + j, as: UInt8.self)
                out[o + j] = (Float(b & 0x0F) - 8) * d
                out[o + j + 16] = (Float(b >> 4) - 8) * d
            }
            p += blockBytes
            o += qk
        }
    }
}

enum Q8_0 {
    static let qk = 32
    static let blockBytes = 34

    static func dequant(_ base: UnsafeRawPointer, count n: Int,
                        into out: UnsafeMutablePointer<Float>) {
        var p = base
        var o = 0
        for _ in 0..<(n / qk) {
            let d = Float(p.loadUnaligned(as: Float16.self))
            for j in 0..<qk {
                let q = p.load(fromByteOffset: 2 + j, as: Int8.self)
                out[o + j] = Float(q) * d
            }
            p += blockBytes
            o += qk
        }
    }
}

// Type-dispatching weight reads. A gemma-4 GGUF mixes four types across the
// same forward (Q4_0 attention, Q2_0 wide MLP, Q8_0 vision, BF16 norms), so
// the engine asks for a tensor, not for a format.
enum GQ {

    // Bytes one logical row of `t` occupies, ne0 wide.
    static func rowBytes(_ t: GGUFTensor) -> Int {
        GGUF.rowByteCount(t.type, t.dims[0])
    }

    // y[m] = sum_k W[m,k] * x[k]; W has ne0 = K (input), ne1 = M (rows).
    // Rows are independent and the closure only READS the weight + activation
    // pointers, so the parallel fill is race-free -- the same argument
    // Q2_0.matvec's nonisolated(unsafe) makes.
    static func matvec(_ w: GGUFTensor, x: [Float], out: inout [Float]) {
        switch w.type {
        case .q2_0:
            Q2_0.matvec(w, x: x, out: &out)
        case .q4_0:
            rowParallel(w, x: x, out: &out) { p, xp, k in
                var acc: Float = 0
                let d = Float(p.loadUnaligned(as: Float16.self))
                for j in 0..<16 {
                    let b = p.load(fromByteOffset: 2 + j, as: UInt8.self)
                    acc += (Float(b & 0x0F) - 8) * xp[k + j]
                    acc += (Float(b >> 4) - 8) * xp[k + j + 16]
                }
                return acc * d
            }
        case .q8_0:
            rowParallel(w, x: x, out: &out) { p, xp, k in
                var acc: Float = 0
                let d = Float(p.loadUnaligned(as: Float16.self))
                for j in 0..<32 {
                    let q = p.load(fromByteOffset: 2 + j, as: Int8.self)
                    acc += Float(q) * xp[k + j]
                }
                return acc * d
            }
        case .bf16:
            denseMatvec(w, x: x, out: &out)
        case .f16:
            denseMatvec(w, x: x, out: &out)
        case .f32:
            denseMatvec(w, x: x, out: &out)
        default:
            fatalError("GQ.matvec: unsupported type \(w.type) for \(w.name)")
        }
    }

    // Y[n][m] = sum_k X[n][k] * W[m][k], for N activation rows at once.
    //
    // The point is DEQUANT REUSE and not the parallelism, which the mat-vec
    // already had: unpacking a 32-weight block costs a mask, a shift, an
    // int-to-float and a subtract per weight, and a mat-vec throws that work
    // away after ONE dot product. A 294-token prefill therefore unpacked
    // every weight in the model 294 times. Here a block is unpacked once and
    // serves all N rows, so the expensive half is divided by N.
    static func matmul(_ w: GGUFTensor, X: [Float], N: Int,
                       out: inout [Float]) {
        let K = w.dims[0]
        let M = w.dims[1]
        switch w.type {
        case .q4_0, .q8_0:
            blockMatmul(w, X: X, N: N, K: K, M: M, out: &out)
        case .bf16, .f16, .f32:
            denseMatmul(w, X: X, N: N, K: K, M: M, out: &out)
        default:
            // A type with no batched twin (Q2_0) still answers correctly one
            // row at a time; only the reuse is lost.
            var row = [Float](repeating: 0, count: K)
            var acc = [Float](repeating: 0, count: M)
            for n in 0..<N {
                for i in 0..<K { row[i] = X[n * K + i] }
                matvec(w, x: row, out: &acc)
                for m in 0..<M { out[n * M + m] = acc[m] }
            }
        }
    }

    private static func blockMatmul(_ w: GGUFTensor, X: [Float], N: Int,
                                    K: Int, M: Int, out: inout [Float]) {
        let q4 = w.type == .q4_0
        let per = q4 ? Q4_0.qk : Q8_0.qk
        let size = q4 ? Q4_0.blockBytes : Q8_0.blockBytes
        let nblk = K / per
        let stride = nblk * size
        // X transposed to [K][N] so the innermost accumulate walks N
        // CONTIGUOUSLY against a broadcast weight, which is the shape that
        // vectorizes. Row-major X strides by K there and does not. The
        // transpose is N*K work feeding N*K*M.
        var XT = [Float](repeating: 0, count: N * K)
        X.withUnsafeBufferPointer { xb in
            XT.withUnsafeMutableBufferPointer { tb in
                for n in 0..<N {
                    for k in 0..<K { tb[k * N + n] = xb[n * K + k] }
                }
            }
        }
        XT.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                nonisolated(unsafe) let base = w.base
                nonisolated(unsafe) let xt = xb.baseAddress!
                nonisolated(unsafe) let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: M) { m in
                    // Stack scratch: one unpacked block and this row's N
                    // accumulators. Allocating per m would cost more than the
                    // arithmetic it holds.
                    withUnsafeTemporaryAllocation(of: Float.self,
                                                  capacity: per + N) { tmp in
                        let blk = tmp.baseAddress!
                        let acc = blk + per
                        for n in 0..<N { acc[n] = 0 }
                        var p = base + m * stride
                        var k = 0
                        for _ in 0..<nblk {
                            if q4 {
                                Q4_0.dequant(p, count: per, into: blk)
                            } else {
                                Q8_0.dequant(p, count: per, into: blk)
                            }
                            for j in 0..<per {
                                let wj = blk[j]
                                let col = xt + (k + j) * N
                                for n in 0..<N { acc[n] += wj * col[n] }
                            }
                            p += size
                            k += per
                        }
                        for n in 0..<N { op[n * M + m] = acc[n] }
                    }
                }
            }
        }
    }

    private static func denseMatmul(_ w: GGUFTensor, X: [Float], N: Int,
                                    K: Int, M: Int, out: inout [Float]) {
        let stride = rowBytes(w)
        let ty = w.type
        X.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                nonisolated(unsafe) let base = w.base
                nonisolated(unsafe) let xp = xb.baseAddress!
                nonisolated(unsafe) let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: M) { m in
                    let p = base + m * stride
                    withUnsafeTemporaryAllocation(of: Float.self,
                                                  capacity: N) { acc in
                        for n in 0..<N { acc[n] = 0 }
                        for i in 0..<K {
                            let e = element(p, ty, i)
                            for n in 0..<N { acc[n] += e * xp[n * K + i] }
                        }
                        for n in 0..<N { op[n * M + m] = acc[n] }
                    }
                }
            }
        }
    }

    // One block-quantized mat-vec, `block` supplying a block's dot product
    // from its base pointer, the activation pointer, and the block's first
    // element index.
    private static func rowParallel(
        _ w: GGUFTensor, x: [Float], out: inout [Float],
        _ block: @escaping @Sendable (UnsafeRawPointer,
                                      UnsafePointer<Float>, Int) -> Float
    ) {
        let K = w.dims[0]
        let M = w.dims[1]
        let per = w.type == .q4_0 ? Q4_0.qk : Q8_0.qk
        let size = w.type == .q4_0 ? Q4_0.blockBytes : Q8_0.blockBytes
        let nblk = K / per
        let stride = nblk * size
        x.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                nonisolated(unsafe) let base = w.base
                nonisolated(unsafe) let xp = xb.baseAddress!
                nonisolated(unsafe) let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: M) { m in
                    var acc: Float = 0
                    var p = base + m * stride
                    var k = 0
                    for _ in 0..<nblk {
                        acc += block(p, xp, k)
                        p += size
                        k += per
                    }
                    op[m] = acc
                }
            }
        }
    }

    // Unquantized weights (the QAT's modules_to_not_convert, and every norm)
    // still occasionally appear as a matrix; expand per row rather than hold
    // a second f32 copy of the whole tensor.
    private static func denseMatvec(_ w: GGUFTensor, x: [Float],
                                    out: inout [Float]) {
        let K = w.dims[0]
        let M = w.dims[1]
        let stride = rowBytes(w)
        let ty = w.type
        x.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                nonisolated(unsafe) let base = w.base
                nonisolated(unsafe) let xp = xb.baseAddress!
                nonisolated(unsafe) let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: M) { m in
                    let p = base + m * stride
                    var acc: Float = 0
                    for i in 0..<K { acc += element(p, ty, i) * xp[i] }
                    op[m] = acc
                }
            }
        }
    }

    private static func element(_ p: UnsafeRawPointer, _ ty: GGUFType,
                                _ i: Int) -> Float {
        let v: Float
        switch ty {
        case .f32:
            v = p.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
        case .f16:
            v = Float(p.loadUnaligned(fromByteOffset: i * 2,
                                      as: Float16.self))
        default:
            let bits = p.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
            v = Float(bitPattern: UInt32(bits) << 16)
        }
        return v
    }

    // `count` weights of row `row`, starting `from` elements in. The gemma
    // embedding tables are gathered this way: token_embd one whole row, the
    // PLE table one 256-wide layer slice out of its 8960-wide row. `from` and
    // `count` must land on block boundaries, which they do -- 256 and 1536
    // are multiples of both 32 and 128.
    static func dequantSpan(_ t: GGUFTensor, row: Int, from: Int, count: Int,
                            into out: inout [Float]) {
        out.withUnsafeMutableBufferPointer { ob in
            gather(t, row: row, from: from, count: count,
                   into: ob.baseAddress!)
        }
    }

    // The same span straight into caller memory. The GPU engine gathers its
    // two embedding tables through this rather than through a kernel: they
    // are read one ROW at a time and never by a GEMM, so keeping them out of
    // an MTLBuffer keeps them out of the pages a command buffer WIRES.
    static func gather(_ t: GGUFTensor, row: Int, from: Int, count: Int,
                       into o: UnsafeMutablePointer<Float>) {
        let base = t.base + row * rowBytes(t)
        switch t.type {
        case .q2_0:
            Q2_0.dequant(base + from / Q2_0.qk * Q2_0.blockBytes,
                         count: count, into: o)
        case .q4_0:
            Q4_0.dequant(base + from / Q4_0.qk * Q4_0.blockBytes,
                         count: count, into: o)
        case .q8_0:
            Q8_0.dequant(base + from / Q8_0.qk * Q8_0.blockBytes,
                         count: count, into: o)
        default:
            for i in 0..<count { o[i] = element(base, t.type, from + i) }
        }
    }
}

// Static range quantization: the QAT trained with every quantized linear's
// input and output ROUNDED to a per-linear scale, and the audio tower does
// not work without it -- the same weights unclamped answer "the content is
// unclear" where the clamped ones transcribe.
//
// A missing scale marks an uncalibrated linear (the checkpoint stores 0 and
// the repack omits it), where the clamp is the identity.

struct SRQ {
    let input: Float
    let output: Float

    static let none = SRQ(input: 0, output: 0)

    init(input: Float, output: Float) {
        self.input = input
        self.output = output
    }

    // LLM_SRQ=0 disables every clamp. This is NOT a runtime option -- the
    // model does not work without it (the audio tower answers "the content is
    // unclear" where the clamped one transcribes), and nothing ships with it
    // off. It exists so a GATE can compare two code paths without rint()
    // standing between them.
    //
    // WHY that matters: rint is a step function, so it turns ANY reordering of
    // a reduction into a whole-quantization-step difference. That puts a ~0.99
    // noise floor under every similarity comparison, which is wide enough to
    // hide a real indexing bug -- a wrong window start, a bad page slot. With
    // the clamp off, two f32 paths computing the same thing agree to fp
    // rounding, so the same comparison resolves ~1e-6 and a structural defect
    // has nowhere left to hide.
    static let enabled =
        ProcessInfo.processInfo.environment["LLM_SRQ"] != "0"

    init(_ g: GGUF, _ weight: String) {
        let base = "gemma4.srq." + weight.replacingOccurrences(
            of: ".weight", with: "")
        // A scale of 0 is already the identity everywhere it is consumed
        // (SRQ.apply, srq_inplace, srq_to and MetalEnc.linear all skip it), so
        // zeroing here disables the clamp without a second code path.
        input = SRQ.enabled ? Float(g.double(base + ".in") ?? 0) : 0
        output = SRQ.enabled ? Float(g.double(base + ".out") ?? 0) : 0
    }

    // clamp(round(x / s), -128, 127) * s. torch.round is round-half-to-EVEN,
    // so anything else drifts on exact .5 -- which the trained scales hit.
    static func apply(_ x: inout [Float], _ s: Float) {
        if s != 0 {
            let inv = 1 / s
            for i in 0..<x.count {
                let q = (x[i] * inv).rounded(.toNearestOrEven)
                x[i] = min(max(q, -128), 127) * s
            }
        }
    }
}

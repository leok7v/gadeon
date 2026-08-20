import Accelerate
// Q2_0 (ggml type 42, PrismML "Q2_0_g128") dequant + ternary mat-vec.
//
// Block (34 bytes, 128 weights): { f16 d; u8 qs[32] }  -- scale FIRST.
// 2-bit codes, 4 per byte, LSB-first; weight = (code - 1) * d
// (00:-1 01:0 10:+1 11:+2). d = absmax >= 0; the sign lives in the code.
// Layout + dequant verified against the real file and llama.cpp dequantize_row_q2_0.
import Foundation

enum Q2_0 {
    static let qk = 128
    static let blockBytes = 34

    // Dequantize one contiguous run of `n` weights (n % 128 == 0) to f32.
    static func dequant(_ base: UnsafeRawPointer, count n: Int, into out: UnsafeMutablePointer<Float>) {
        let nb = n / qk
        var p = base
        var o = 0
        for _ in 0..<nb {
            let d = Float(p.loadUnaligned(as: Float16.self))
            let qs = p + 2
            for j in 0..<qk {
                let byte = qs.load(fromByteOffset: j >> 2, as: UInt8.self)
                let code = Int((byte >> UInt8((j & 3) << 1)) & 0x03)
                out[o + j] = Float(code - 1) * d
            }
            p += blockBytes
            o += qk
        }
    }

    // y[m] = sum_k W[m,k] * x[k], W a Q2_0 tensor with ne0 = K (input), ne1 = M (rows).
    // Per row: K/128 blocks; per block dot = d * sum((code-1) * x). Uses the
    // bit-decomposition dot = d*(acc_lo + 2*acc_hi - sumy) from the fork's Metal
    // kernel to keep it multiply-light.
    static func matvec(_ w: GGUFTensor, x: [Float], out: inout [Float]) {
        let K = w.dims[0]
        let M = w.dims[1]
        precondition(K % qk == 0)
        let nblk = K / qk
        let rowBytes = nblk * blockBytes
        // The concurrentPerform closure is @Sendable; each iteration `m` writes a
        // distinct output row and only READS the (immutable) weight + activation
        // pointers, so the parallel fill is race-free. `nonisolated(unsafe)`
        // asserts that to the strict-concurrency checker (which cannot prove the
        // per-index disjointness of raw pointers).
        x.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                nonisolated(unsafe) let base = w.base
                nonisolated(unsafe) let xp = xb.baseAddress!
                nonisolated(unsafe) let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: M) { m in
                    var acc: Float = 0
                    var p = base + m * rowBytes
                    var k = 0
                    for _ in 0..<nblk {
                        let d = Float(p.loadUnaligned(as: Float16.self))
                        let qs = p + 2
                        // sum over the 128 block weights: (code-1)*x = code*x - x
                        var lo: Float = 0, hi: Float = 0, sy: Float = 0
                        for j in 0..<qk {
                            let byte = qs.load(fromByteOffset: j >> 2, as: UInt8.self)
                            let code = (byte >> UInt8((j & 3) << 1)) & 0x03
                            let xv = xp[k + j]
                            sy += xv
                            if code & 1 != 0 { lo += xv }
                            if code & 2 != 0 { hi += xv }
                        }
                        acc += d * (lo + 2 * hi - sy)
                        p += blockBytes
                        k += qk
                    }
                    op[m] = acc
                }
            }
        }
    }
}

enum Q2E8Row {
    static let qk = 128
    static let blockBytes = 34
    static let perCode = E8P.dim

    static func dequant(_ base: UnsafeRawPointer, count n: Int,
                        into out: UnsafeMutablePointer<Float>) {
        let nb = n / qk
        let codes = base + nb * 2
        for ib in 0..<nb {
            let d = Float((base + ib * 2).loadUnaligned(as: Float16.self))
            for u in 0..<(qk / perCode) {
                let c = (codes + ib * (qk / perCode) * 2 + u * 2)
                    .loadUnaligned(as: UInt16.self)
                let v = E8P.unpack(c)
                for i in 0..<perCode {
                    out[ib * qk + u * perCode + i] = d * v[i]
                }
            }
        }
    }

    static func matvec(_ w: GGUFTensor, x: [Float], out: inout [Float]) {
        let K = w.dims[0]
        let M = w.dims[1]
        precondition(K % qk == 0)
        let nblk = K / qk
        let rowBytes = nblk * blockBytes
        let lanes = qk / perCode
        x.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                nonisolated(unsafe) let base = w.base
                nonisolated(unsafe) let xp = xb.baseAddress!
                nonisolated(unsafe) let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: M) { m in
                    let rp = base + m * rowBytes
                    let codes = rp + nblk * 2
                    var acc: Float = 0
                    for ib in 0..<nblk {
                        let d = Float((rp + ib * 2)
                            .loadUnaligned(as: Float16.self))
                        var dot: Float = 0
                        for u in 0..<lanes {
                            let c = (codes + ib * lanes * 2 + u * 2)
                                .loadUnaligned(as: UInt16.self)
                            let v = E8P.unpack(c)
                            let at = ib * qk + u * perCode
                            for i in 0..<perCode { dot += v[i] * xp[at + i] }
                        }
                        acc += d * dot
                    }
                    op[m] = acc
                }
            }
        }
    }
}

enum QB {
    static func matvec(_ w: GGUFTensor, x: [Float], out: inout [Float]) {
        switch w.type {
        case .q2_e8: Q2E8Row.matvec(w, x: x, out: &out)
        default: GQ.matvec(w, x: x, out: &out)
        }
    }

    static func dequant(_ w: GGUFTensor, row: Int, count n: Int,
                        into out: UnsafeMutablePointer<Float>) {
        switch w.type {
        case .q2_e8:
            Q2E8Row.dequant(w.base + row * GQ.rowBytes(w), count: n, into: out)
        default:
            GQ.gather(w, row: row, from: 0, count: n, into: out)
        }
    }
}

// Dense tensor -> [Float], for the f32/f16/bf16 tensors the ViT consumes
// whole (weights become cblas GEMM operands). bf16 -> f32 is a 16-bit left
// shift into the top of the IEEE 754 single bit pattern.
enum Dense {
    // ggml stores [out][in] row-major; a plain vDSP_mmul wants [in][out].
    // Block-quantized rows go through GQ.dequantSpan, which knows every
    // type the gemma repack emits; the unquantized ones are a straight read.
    static func transposedFloats(_ t: GGUFTensor, _ inDim: Int,
                                 _ outDim: Int) -> [Float] {
        var flat: [Float]
        switch t.type {
        case .q2_0, .q4_0:
            flat = [Float](repeating: 0, count: inDim * outDim)
            var row = [Float](repeating: 0, count: inDim)
            for m in 0..<outDim {
                GQ.dequantSpan(t, row: m, from: 0, count: inDim, into: &row)
                for i in 0..<inDim { flat[m * inDim + i] = row[i] }
            }
        default:
            flat = floats(t)
        }
        var out = [Float](repeating: 0, count: flat.count)
        vDSP_mtrans(flat, 1, &out, 1, vDSP_Length(inDim), vDSP_Length(outDim))
        return out
    }

    // The seam between a mapped GGUF tensor and the strided f32 ops. F32 is
    // WRAPPED IN PLACE over the mapping, so the GGUF has to outlive every
    // tensor made this way; F16 widens into the arena. A block-quantized type
    // returns nil rather than a guess -- it has no f32 image to wrap, and the
    // caller knows whether dequantizing one is affordable.
    static func tensor(_ t: GGUFTensor, into arena: Arena) -> Tensor? {
        var out: Tensor? = nil
        var ne: [Int64] = [1, 1, 1, 1]
        for d in 0..<t.dims.count { ne[d] = Int64(t.dims[d]) }
        let nDims = Int32(t.dims.count)
        if t.type == .f32 {
            let data = UnsafeMutableRawPointer(mutating: t.base)
                .assumingMemoryBound(to: Float.self)
            out = tensorWrapNd(arena, nDims, data, ne)
        } else if t.type == .f16 {
            let o = tensorNewNd(arena, nDims, ne)
            let total = Int(tensorNelements(o))
            for i in 0..<total {
                let h = (t.base + i * 2).loadUnaligned(as: UInt16.self)
                o.data[i] = Float(Float16(bitPattern: h))
            }
            out = o
        }
        if let o = out { tensorSetName(o, t.name) }
        return out
    }

    static func floats(_ t: GGUFTensor) -> [Float] {
        let n = t.count
        return [Float](unsafeUninitializedCapacity: n) { buf, cnt in
            switch t.type {
            case .f32:
                memcpy(buf.baseAddress!, t.base, n * 4)
            case .f16:
                let src = t.base.assumingMemoryBound(to: Float16.self)
                for i in 0..<n { buf[i] = Float(src[i]) }
            case .bf16:
                let src = t.base.assumingMemoryBound(to: UInt16.self)
                for i in 0..<n {
                    buf[i] = Float(bitPattern: UInt32(src[i]) << 16)
                }
            case .q8_0:
                // block_q8_0: { f16 d; i8 qs[32] }, weight = d * qs[j]
                var p = t.base
                var o = 0
                for _ in 0..<(n / 32) {
                    let d = Float(p.loadUnaligned(as: Float16.self))
                    for j in 0..<32 {
                        let q = p.load(fromByteOffset: 2 + j, as: Int8.self)
                        buf[o + j] = d * Float(q)
                    }
                    p += 34
                    o += 32
                }
            default:
                fatalError("Dense.floats: unsupported type \(t.type) "
                    + "for \(t.name)")
            }
            cnt = n
        }
    }
}

// F32 tensor helpers (norms, ssm_a/dt/conv1d, all small params are F32).
enum F32T {
    static func array(_ t: GGUFTensor) -> [Float] {
        precondition(t.type == .f32)
        let n = t.count
        return [Float](unsafeUninitializedCapacity: n) { buf, cnt in
            memcpy(buf.baseAddress!, t.base, n * 4); cnt = n
        }
    }
    static func ptr(_ t: GGUFTensor) -> UnsafePointer<Float> {
        precondition(t.type == .f32)
        return t.base.assumingMemoryBound(to: Float.self)
    }
}

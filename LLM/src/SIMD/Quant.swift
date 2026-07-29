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

// Dense tensor -> [Float], for the f32/f16/bf16 tensors the ViT consumes
// whole (weights become cblas GEMM operands). bf16 -> f32 is a 16-bit left
// shift into the top of the IEEE 754 single bit pattern.
enum Dense {
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

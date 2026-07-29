// The Metal Shading Language kernels for the Bonsai (Qwen3.5 GDN hybrid) GPU
// backend, compiled to default.metallib at BUILD time: the Xcode framework
// target compiles this file natively; the SwiftPM build runs the MetalBuild
// plugin (LLM/plugins/MetalBuild). MetalContext loads the library from its
// bundle -- no runtime source compile.
//
// Every kernel reproduces the numerics of the pure-Swift SIMD reference engine
// (LLM/src/SIMD), which is itself at end-to-end parity with llama.cpp. The
// SIMD engine is the op-by-op oracle: each kernel below has a named Swift
// counterpart it must match (Q2_0.matvec, Kern.rmsnorm, GDN.step, ...).
//
// Q2_0 block (from PrismML-Eng/llama.cpp, GGML_TYPE_Q2_0): 34 bytes, 128
// weights, { half d; uchar qs[32] }. Codes are 2-bit, 4 per byte, LSB-first;
// weight = (code - 1) * d. Blocks are addressed by manual byte arithmetic
// (stride 34, d at +0, qs at +2) to avoid any MSL struct-packing assumption.
//
// Weights live in one big no-copy buffer (the whole mmap'd GGUF); every kernel
// that reads a weight takes a byte offset into it. Activation and state buffers
// are shared-storage f32 (unified memory). All dispatches for one token run on
// one command encoder, so Metal's hazard tracking serializes the dependent
// steps and scratch buffers are safely reused across layers.

#include <metal_stdlib>
using namespace metal;

inline float siluf(float x)    { return x / (1.0f + exp(-x)); }
inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float softplusf(float x){ return max(x, 0.0f) + log(1.0f + exp(-fabs(x))); }

// ---- Q2_0 ternary mat-vec: out[m] = sum_k W[m,k] * x[k] -----------------
// W ne0=K (input, fastest), ne1=M (rows), based at byte offset woff. Each
// simdgroup handles NR0=8 output rows; TPB=8 lanes cooperate on one 128-weight
// block (their 32 qs bytes read as 4 consecutive bytes each -> coalesced weight
// stream). Per-block dot d*(lo + 2*hi - sumy) = d*sum((code-1)*x), matching
// Q2_0.matvec; simd_sum reduces the 32 lanes per row. woff is 64-bit: the
// weight buffer is the whole GGUF (>4 GB) so tensor byte offsets exceed uint.
//
// Decode is latency-bound (measured 8.1 t/s = 65% of the 12.4 t/s memory wall,
// only 12% of compute peak), so each lane keeps TWO blocks in flight per
// iteration in DISTINCT scalar arrays (a loop-indexed yl[u][i] spills to local
// memory and regresses; UN=2 is the sweet spot, UN=4 blows the register budget
// to 5.7 t/s): +8% -> 8.7 t/s. The scalar lo/hi select-add beats every
// alternative tried here -- float4-dot, scalar-fma, and NSG=4 all regressed.
struct GemvArgs { ulong woff; uint K; uint M; };

kernel void q2_0_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 128;
    const ushort TPB = 8, SW = 16, STEP = 32 / TPB;   // 4 blocks in flight
    const ulong rowBytes = (ulong) nblk * 34;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort il  = (tiisg % TPB) * SW;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    // Two blocks (ib0, ib1) per iteration, held in DISTINCT scalar arrays (not a
    // 2D array -- a loop-indexed yl[u][i] spills to local memory and regresses):
    // both blocks' activations + all weight reads issue before either decode,
    // doubling in-flight loads. UN=2 is the sweet spot (x4 blew the register
    // budget: 5.7 t/s).
    uint ib = grp;
    for (; ib + STEP < nblk; ib += 2 * STEP) {
        const uint ib0 = ib, ib1 = ib + STEP;
        device const float * y0 = x + (ulong) ib0 * 128 + il;
        device const float * y1 = x + (ulong) ib1 * 128 + il;
        float yl0[16], yl1[16];
        float sy0 = 0.0f, sy1 = 0.0f;
        for (ushort i = 0; i < SW; i++) {
            yl0[i] = y0[i]; sy0 += y0[i];
            yl1[i] = y1[i]; sy1 += y1[i];
        }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * b0 = W + row * rowBytes + (ulong) ib0 * 34;
                device const uchar * b1 = W + row * rowBytes + (ulong) ib1 * 34;
                const float d0 = (float) (*(device const half *) b0);
                const float d1 = (float) (*(device const half *) b1);
                device const uchar * q0 = b0 + 2 + il / 4;
                device const uchar * q1 = b1 + 2 + il / 4;
                float lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
                for (ushort i = 0; i < SW; i++) {
                    const uchar c0 = (q0[i >> 2] >> ((i & 3) * 2)) & 3;
                    const uchar c1 = (q1[i >> 2] >> ((i & 3) * 2)) & 3;
                    if (c0 & 1) { lo0 += yl0[i]; }
                    if (c0 & 2) { hi0 += yl0[i]; }
                    if (c1 & 1) { lo1 += yl1[i]; }
                    if (c1 & 2) { hi1 += yl1[i]; }
                }
                acc[r] += d0 * (lo0 + 2.0f * hi0 - sy0)
                        + d1 * (lo1 + 2.0f * hi1 - sy1);
            }
        }
    }
    for (; ib < nblk; ib += STEP) {           // tail (odd block count)
        device const float * y = x + (ulong) ib * 128 + il;
        float yl[16];
        float sy = 0.0f;
        for (ushort i = 0; i < SW; i++) { yl[i] = y[i]; sy += y[i]; }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + row * rowBytes + (ulong) ib * 34;
                const float d = (float) (*(device const half *) bp);
                device const uchar * qs = bp + 2 + il / 4;
                float lo = 0.0f, hi = 0.0f;
                for (ushort i = 0; i < SW; i++) {
                    const uchar code = (qs[i >> 2] >> ((i & 3) * 2)) & 3;
                    const float xv = yl[i];
                    if (code & 1) { lo += xv; }
                    if (code & 2) { hi += xv; }
                }
                acc[r] += d * (lo + 2.0f * hi - sy);
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

// ---- Q2_0 batched mat-mat (prefill): out[N,M] = X[N,K] @ W[K,M] ----------
// Token-major: X[col*K + k], out[col*M + m]. Each threadgroup owns one weight
// row m and a tile of TN=8 token-columns, so the weight row is STREAMED ONCE and
// reused across the 8 columns -- the weight-amortization that makes prefill scale
// (token-by-token re-streams the whole 7 GB per token). 32 lanes cooperate: lane
// L reads byte 2+L of each block (4 codes), coalesced; simd_sum reduces per
// column. d*(sum(code*x) - sum(x)) = d*sum((code-1)*x), matching Q2_0.matvec.
kernel void q2_0_gemm(
        device const uchar * weights [[buffer(0)]],
        device const float * X       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        constant uint      & N       [[buffer(4)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    // 2-D tile: NR0=8 weight rows x TN=4 token-columns per simdgroup. TPB=8
    // lanes cooperate per 128-weight block (coalesced qs, 4 blocks in flight);
    // each block's SW=16 codes of a row are decoded ONCE and dotted against all
    // TN columns' activation slices (weight reused across cols), while the row
    // tile reuses each column's slice (activation reused across rows). simd_sum
    // reduces the 32 lanes per (row,col).
    const ushort NR0 = 8, TN = 4, TPB = 8, SW = 16;
    const uint row0 = tgpig.y * NR0;
    const uint col0 = tgpig.x * TN;
    const uint nblk = a.K / 128;
    const ulong rowBytes = (ulong) nblk * 34;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort il  = (tiisg % TPB) * SW;
    float acc[8][4];
    for (ushort r = 0; r < NR0; r++) {
        for (ushort t = 0; t < TN; t++) { acc[r][t] = 0.0f; }
    }
    for (uint ib = grp; ib < nblk; ib += 32 / TPB) {
        float yl[4][16];
        float sy[4] = { 0, 0, 0, 0 };
        for (ushort t = 0; t < TN; t++) {
            const uint col = col0 + t;
            device const float * xc = X + (ulong) (col < N ? col : 0) * a.K
                + ib * 128 + il;
            for (ushort i = 0; i < SW; i++) { yl[t][i] = xc[i]; sy[t] += xc[i]; }
        }
        for (ushort r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + row * rowBytes + (ulong) ib * 34;
                const float d = (float) (*(device const half *) bp);
                device const uchar * qs = bp + 2 + il / 4;
                float lo[4] = { 0, 0, 0, 0 }, hi[4] = { 0, 0, 0, 0 };
                for (ushort i = 0; i < SW; i++) {
                    const uchar code = (qs[i >> 2] >> ((i & 3) * 2)) & 3;
                    if (code & 1) {
                        for (ushort t = 0; t < TN; t++) { lo[t] += yl[t][i]; }
                    }
                    if (code & 2) {
                        for (ushort t = 0; t < TN; t++) { hi[t] += yl[t][i]; }
                    }
                }
                for (ushort t = 0; t < TN; t++) {
                    acc[r][t] += d * (lo[t] + 2.0f * hi[t] - sy[t]);
                }
            }
        }
    }
    for (ushort r = 0; r < NR0; r++) {
        const uint row = row0 + r;
        for (ushort t = 0; t < TN; t++) {
            const uint col = col0 + t;
            const float s = simd_sum(acc[r][t]);
            if (tiisg == 0 && row < a.M && col < N) {
                out[(ulong) col * a.M + row] = s;
            }
        }
    }
}

// ---- Q2_0 simdgroup-matrix GEMM (prefill): out[N,M] = X[N,K] @ W[K,M] -----
// Port of llama.cpp's kernel_mul_mm (classic simdgroup_float8x8 path) for the
// ternary weight: each 128-thread threadgroup computes a 64(M) x 32(N) output
// tile by streaming 64x32 W tiles (dequantized) + 32x32 X tiles through
// threadgroup memory and multiplying on the 8x8 HARDWARE MATRIX UNITS. This is
// the compute path prefill needs -- the scalar q2_0_gemm loses ~10x here.
// Weight rows are [K,M] (row m = base + m*rowBytes, Q2_0 34-byte blocks); X is
// token-major f32 [N,K]; out is [N,M] laid out out[n*M+m] == dst[m + n*M],
// matching Q2_0.matvec's column order. All f32 (simdgroup_float8x8), no
// activation downcast, so parity holds against the SIMD oracle. Every tensor's
// K is a multiple of 128 and >=32, so no in-tile K bounds check is needed; the
// output tile is bounds-checked (M can be 48 = nVHead, N a short final chunk).
struct block_q2_0 { half d; uchar qs[32]; };

static inline void dq_q2_0(device const block_q2_0 * xb, short il,
                           thread float4x4 & reg) {
    device const uchar * qs = xb->qs;      // il-th 16-elem sub-block = 4 bytes
    const float d = (float) xb->d;
    const int bo = il * 4;
    for (int i = 0; i < 4; i++) {
        const uchar b = qs[bo + i];
        reg[i][0] = ((float) ((b >> 0) & 3) - 1.0f) * d;
        reg[i][1] = ((float) ((b >> 2) & 3) - 1.0f) * d;
        reg[i][2] = ((float) ((b >> 4) & 3) - 1.0f) * d;
        reg[i][3] = ((float) ((b >> 6) & 3) - 1.0f) * d;
    }
}

// half-tile twin: dequant a Q2_0 sub-block DIRECTLY into a half register tile
// (no float4x4 intermediate), matching the fork's f16 mul_mm weight path.
static inline void dq_q2_0_h(device const block_q2_0 * xb, short il,
                             thread half4x4 & reg) {
    device const uchar * qs = xb->qs;
    const half d = xb->d;
    const int bo = il * 4;
    for (int i = 0; i < 4; i++) {
        const uchar b = qs[bo + i];
        reg[i][0] = ((half) ((b >> 0) & 3) - 1.0h) * d;
        reg[i][1] = ((half) ((b >> 2) & 3) - 1.0h) * d;
        reg[i][2] = ((half) ((b >> 4) & 3) - 1.0h) * d;
        reg[i][3] = ((half) ((b >> 6) & 3) - 1.0h) * d;
    }
}

kernel void q2_0_gemm_mm(
        device const uchar * weights [[buffer(0)]],
        device const float * X       [[buffer(1)]],
        device       float * dst     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        constant uint      & N       [[buffer(4)]],
        threadgroup uchar  * shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int K = (int) a.K, M = (int) a.M;
    const short nl = 8;                    // 128/16 sub-blocks per Q2_0 block
    const int NR0 = 64, NR1 = 32, NK = 32, NL0 = NK / 16, NL1 = NK / 8;
    threadgroup float * sa = (threadgroup float *) (shmem);
    threadgroup float * sb = (threadgroup float *) (shmem + 8192);

    const int r0 = tgpig.y * NR0;          // first M row of this tile
    const int r1 = tgpig.x * NR1;          // first N col of this tile
    const short nr0 = (M - r0 < NR0) ? (short) (M - r0) : NR0;
    const short nr1 = ((int) N - r1 < NR1) ? (short) ((int) N - r1) : NR1;

    const short lr0 = ((short) tiitg / NL0) < nr0 ? ((short) tiitg / NL0)
                                                  : nr0 - 1;
    const short lr1 = ((short) tiitg / NL1) < nr1 ? ((short) tiitg / NL1)
                                                  : nr1 - 1;
    const short il0 = tiitg % NL0;
    short il = il0;

    const ulong rowBytes = (ulong) (K / 128) * 34;
    device const block_q2_0 * x = (device const block_q2_0 *)
        (weights + a.woff + rowBytes * (r0 + lr0));    // offset1 = il0/nl = 0
    const short iy = 8 * (tiitg % NL1);
    device const float * y = X + (ulong) (r1 + lr1) * K + iy;

    simdgroup_float8x8 ma[4], mb[2], mc[8];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        float4x4 temp_a;
        dq_q2_0(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
        }
        {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            threadgroup float * bp = sb + 64 * ib + 8 * ly;
            for (short i = 0; i < 8; i++) { bp[i] = y[i]; }
        }
        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + 1 : x;
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const float * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const float * lsmb = sb + 2 * 64 * (sgitg / 2);
        #pragma clang loop unroll(full)
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= (int) N) {
        device float * C = dst + (r0 + 32 * (sgitg & 1))
            + (ulong) (r1 + 16 * (sgitg >> 1)) * M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong) M * (i / 4),
                            M, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp = (threadgroup float *) shmem
            + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * NR0 * (i / 4),
                            NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = dst + r0 + (ulong) (r1 + j) * M;
                threadgroup float * C = temp + j * NR0;
                for (int i = 0; i < nr0; i++) { D[i] = C[i]; }
            }
        }
    }
}

// ---- Q2_0 simdgroup-matrix GEMM, HALF tiles: out[N,M] = X[N,K] @ W[K,M] --
// Identical to q2_0_gemm_mm but the A/B tiles are half (simdgroup_half8x8), the
// accumulator stays f32, and dequant writes half directly -- the fork's f16
// mul_mm config. Threadgroup memory halves to 6144 (sa 64x32 half=4096 + sb
// 32x32 half=2048), raising resident-threadgroup occupancy. Activations arrive
// f32 and downcast to half in-kernel; parity holds to fp16 tolerance (checkGemm
// maxAbsDiff ~6e-4 < the 1e-3 gate). The bounds-checked spill path reuses shmem
// as f32 temp (needs 8192), so the host allocates 8192 for a partial tile.
kernel void q2_0_gemm_mm_h(
        device const uchar * weights [[buffer(0)]],
        device const float * X       [[buffer(1)]],
        device       float * dst     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        constant uint      & N       [[buffer(4)]],
        threadgroup uchar  * shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int K = (int) a.K, M = (int) a.M;
    const short nl = 8;
    const int NR0 = 64, NR1 = 32, NK = 32, NL0 = NK / 16, NL1 = NK / 8;
    threadgroup half * sa = (threadgroup half *) (shmem);
    threadgroup half * sb = (threadgroup half *) (shmem + 4096);

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    const short nr0 = (M - r0 < NR0) ? (short) (M - r0) : NR0;
    const short nr1 = ((int) N - r1 < NR1) ? (short) ((int) N - r1) : NR1;

    const short lr0 = ((short) tiitg / NL0) < nr0 ? ((short) tiitg / NL0)
                                                  : nr0 - 1;
    const short lr1 = ((short) tiitg / NL1) < nr1 ? ((short) tiitg / NL1)
                                                  : nr1 - 1;
    const short il0 = tiitg % NL0;
    short il = il0;

    const ulong rowBytes = (ulong) (K / 128) * 34;
    device const block_q2_0 * x = (device const block_q2_0 *)
        (weights + a.woff + rowBytes * (r0 + lr0));
    const short iy = 8 * (tiitg % NL1);
    device const float * y = X + (ulong) (r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a;
        dq_q2_0_h(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
        }
        {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            threadgroup half * bp = sb + 64 * ib + 8 * ly;
            for (short i = 0; i < 8; i++) { bp[i] = (half) y[i]; }
        }
        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + 1 : x;
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);
        #pragma clang loop unroll(full)
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= (int) N) {
        device float * C = dst + (r0 + 32 * (sgitg & 1))
            + (ulong) (r1 + 16 * (sgitg >> 1)) * M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong) M * (i / 4),
                            M, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp = (threadgroup float *) shmem
            + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * NR0 * (i / 4),
                            NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = dst + r0 + (ulong) (r1 + j) * M;
                threadgroup float * C = temp + j * NR0;
                for (int i = 0; i < nr0; i++) { D[i] = C[i]; }
            }
        }
    }
}

// ---- Q2_0 row dequant (token embedding): out[k] = (code-1)*d -------------
// `woff` already points at the wanted row's first block; one thread per weight.
kernel void q2_0_dequant_row(
        device const uchar * weights [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        constant GemvArgs  & a       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.K) {
        const uint ib = gid / 128;
        const uint j  = gid % 128;
        device const uchar * bp = weights + a.woff + (ulong) ib * 34;
        const float d = (float) (*(device const half *) bp);
        const uchar code = (bp[2 + (j >> 2)] >> ((j & 3) * 2)) & 3;
        out[gid] = (float) ((int) code - 1) * d;
    }
}

// ---- RMSNorm over one contiguous n-vector: y = x/sqrt(mean(x^2)+eps)*w ----
// llama.cpp build_norm multiplies the stored weight directly (no 1+w); matches
// Kern.rmsnorm. One threadgroup; a simd + threadgroup reduction over n. `woff`
// is the weight's byte offset (f32) in the big buffer.
struct NormArgs { ulong woff; uint n; float eps; };

kernel void rmsnorm(
        device const float * x       [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device       float * y       [[buffer(2)]],
        constant NormArgs  & a       [[buffer(3)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) { ss += x[i] * x[i]; }
    ss = simd_sum(ss);
    if (tii == 0) { shmem[sgi] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
        float v = (tii < (ntg + 31) / 32) ? shmem[tii] : 0.0f;
        v = simd_sum(v);
        if (tii == 0) { shmem[0] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(shmem[0] / (float) a.n + a.eps);
    device const float * w = (device const float *) (weights + a.woff);
    for (uint i = tid; i < a.n; i += ntg) { y[i] = x[i] * scale * w[i]; }
}

// Batched whole-vector RMSNorm to a SEPARATE output (attn_norm / post_norm in
// prefill, where the input x=[N,n] must be preserved for the residual). One
// threadgroup per token-row.
kernel void rmsnorm_batch(
        device const float * x       [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device       float * y       [[buffer(2)]],
        constant NormArgs  & a       [[buffer(3)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device const float * xr = x + (ulong) row * a.n;
    device       float * yr = y + (ulong) row * a.n;
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) { ss += xr[i] * xr[i]; }
    ss = simd_sum(ss);
    if (tii == 0) { shmem[sgi] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
        float v = (tii < (ntg + 31) / 32) ? shmem[tii] : 0.0f;
        v = simd_sum(v);
        if (tii == 0) { shmem[0] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(shmem[0] / (float) a.n + a.eps);
    device const float * w = (device const float *) (weights + a.woff);
    for (uint i = tid; i < a.n; i += ntg) { yr[i] = xr[i] * scale * w[i]; }
}

// ---- Per-row RMSNorm / L2Norm of a [d, rows] buffer, in place -------------
// rows-major with d fastest. RMSNorm uses a weight (Kern.rmsnormRows); L2Norm
// has none and divides by sqrt(sum+eps) (Kern.l2normRows). One threadgroup per
// row. `xoff` lets q|k|v share one buffer via an element offset.
struct RowArgs { ulong woff; uint d; uint xoff; float eps; };

kernel void rmsnorm_rows(
        device       float * x       [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        constant RowArgs   & a       [[buffer(2)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device float * r = x + a.xoff + row * a.d;
    float ss = 0.0f;
    for (uint i = tid; i < a.d; i += ntg) { ss += r[i] * r[i]; }
    ss = simd_sum(ss);
    if (tii == 0) { shmem[sgi] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
        float v = (tii < (ntg + 31) / 32) ? shmem[tii] : 0.0f;
        v = simd_sum(v);
        if (tii == 0) { shmem[0] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(shmem[0] / (float) a.d + a.eps);
    device const float * w = (device const float *) (weights + a.woff);
    for (uint i = tid; i < a.d; i += ntg) { r[i] = r[i] * scale * w[i]; }
}

kernel void l2norm_rows(
        device       float * x     [[buffer(0)]],
        constant RowArgs   & a     [[buffer(1)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device float * r = x + a.xoff + row * a.d;
    float ss = 0.0f;
    for (uint i = tid; i < a.d; i += ntg) { ss += r[i] * r[i]; }
    ss = simd_sum(ss);
    if (tii == 0) { shmem[sgi] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
        float v = (tii < (ntg + 31) / 32) ? shmem[tii] : 0.0f;
        v = simd_sum(v);
        if (tii == 0) { shmem[0] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(shmem[0] + a.eps);
    for (uint i = tid; i < a.d; i += ntg) { r[i] *= scale; }
}

// Batched per-row L2Norm: for each of `tokens` tokens, normalize `rowsPerTok`
// rows of length d, token n's rows based at n*tokStride. One dispatch replaces
// the 2*N tiny per-token l2norm_rows calls that made GDN prefill CPU-bound: q
// and k are the contiguous first 2*keyDim of a token's convOut (= 2*nKHead rows
// of dState), so rowsPerTok=2*nKHead, tokStride=convDim covers them all. One
// threadgroup per (token, row); mirrors l2norm_rows exactly.
struct RowBatchArgs { uint d; uint rowsPerTok; uint tokStride; float eps; };

kernel void l2norm_rows_batch(
        device       float * x     [[buffer(0)]],
        constant RowBatchArgs & a  [[buffer(1)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint  gr   [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    const uint n = gr / a.rowsPerTok;
    const uint local = gr % a.rowsPerTok;
    device float * r = x + (ulong) n * a.tokStride + (ulong) local * a.d;
    float ss = 0.0f;
    for (uint i = tid; i < a.d; i += ntg) { ss += r[i] * r[i]; }
    ss = simd_sum(ss);
    if (tii == 0) { shmem[sgi] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
        float v = (tii < (ntg + 31) / 32) ? shmem[tii] : 0.0f;
        v = simd_sum(v);
        if (tii == 0) { shmem[0] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(shmem[0] + a.eps);
    for (uint i = tid; i < a.d; i += ntg) { r[i] *= scale; }
}

// ---- Partial NEOX RoPE on the first nRot dims of each head ----------------
// heads laid out [headDim, nHead]; one thread per rotated pair (Kern.ropeNeox).
struct RopeArgs { uint headDim; uint nHead; uint nRot; float base; uint pos; };

kernel void rope_neox(
        device float     * x   [[buffer(0)]],
        constant RopeArgs & a  [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hf = a.nRot / 2;
    if (gid < a.nHead * hf) {
        const uint h = gid / hf;
        const uint i = gid % hf;
        const uint hoff = h * a.headDim;
        const float freq = pow(a.base, -2.0f * (float) i / (float) a.nRot);
        const float ang = (float) a.pos * freq;
        const float c = cos(ang), s = sin(ang);
        const float p = x[hoff + i];
        const float q = x[hoff + i + hf];
        x[hoff + i]      = p * c - q * s;
        x[hoff + i + hf] = p * s + q * c;
    }
}

// ---- Elementwise glue -----------------------------------------------------
// a[i] = silu(a[i]) * b[i]  (FFN gate*up, and the GDN o*silu(z) uses siluMul
// with a,b swapped: o[i] *= silu(z[i]) -> see siluMulRev).
kernel void silu_mul(device float * a [[buffer(0)]],
                     device const float * b [[buffer(1)]],
                     constant uint & n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) { a[gid] = siluf(a[gid]) * b[gid]; }
}

// a[i] *= silu(b[i])   (o *= silu(z))
kernel void mul_silu(device float * a [[buffer(0)]],
                     device const float * b [[buffer(1)]],
                     constant uint & n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) { a[gid] = a[gid] * siluf(b[gid]); }
}

// x[i] += y[i]   (residual add)
kernel void add_inplace(device float * x [[buffer(0)]],
                        device const float * y [[buffer(1)]],
                        constant uint & n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < n) { x[gid] += y[gid]; }
}

// ---- GDN gates: beta = sigmoid(bPre); g = softplus(aPre+dt)*aNeg ----------
// dt (ssm_dt.bias) and aNeg (ssm_a = -exp(A_log)) are f32 tensors; one thread
// per value head (GDN.step gate loop).
struct GateArgs { ulong dtOff; ulong aOff; uint nV; };

kernel void gdn_gate(
        device const float * bPre    [[buffer(0)]],
        device const float * aPre    [[buffer(1)]],
        device const uchar * weights [[buffer(2)]],
        device       float * beta    [[buffer(3)]],
        device       float * g       [[buffer(4)]],
        constant GateArgs  & a       [[buffer(5)]],
        constant uint      & total   [[buffer(6)]],
        uint gid [[thread_position_in_grid]]) {
    // `total` = nV (single token) or N*nV (batched); dt/aNeg are per value head,
    // so index them by gid % nV.
    if (gid < total) {
        device const float * dt   = (device const float *) (weights + a.dtOff);
        device const float * aNeg = (device const float *) (weights + a.aOff);
        const uint h = gid % a.nV;
        beta[gid] = sigmoidf(bPre[gid]);
        g[gid] = softplusf(aPre[gid] + dt[h]) * aNeg[h];
    }
}

// ---- GDN causal depthwise conv (K=4) + silu, then shift the ring ----------
// window per channel c: [convState(3), qkvMix[c]]; out = silu(sum window*w),
// w(j,c) at c*4+j (GDN.step conv loop). One thread per channel; each reads its
// channel's ring then overwrites it (no cross-thread hazard).
struct ConvArgs { ulong cwOff; uint convDim; uint dConv; };

kernel void gdn_conv(
        device const float * qkvMix   [[buffer(0)]],
        device       float * convState[[buffer(1)]],
        device const uchar * weights  [[buffer(2)]],
        device       float * out      [[buffer(3)]],
        constant ConvArgs  & a        [[buffer(4)]],
        uint c [[thread_position_in_grid]]) {
    if (c < a.convDim) {
        device const float * cw =
            (device const float *) (weights + a.cwOff) + c * a.dConv;
        const uint kc = a.dConv;
        float acc = 0.0f;
        for (uint j = 0; j + 1 < kc; j++) {
            acc += convState[j * a.convDim + c] * cw[j];
        }
        acc += qkvMix[c] * cw[kc - 1];
        out[c] = siluf(acc);
        for (uint j = 0; j + 2 < kc; j++) {
            convState[j * a.convDim + c] =
                convState[(j + 1) * a.convDim + c];
        }
        if (kc >= 2) { convState[(kc - 2) * a.convDim + c] = qkvMix[c]; }
    }
}

// ---- GDN autoregressive delta-rule scan (one token) -----------------------
// Per value head hv: gamma=exp(g); S*=gamma; sk[j]=sum_i S[i,j]*k[i];
// d[j]=beta*(v[j]-sk[j]); S[i,j]+=k[i]*d[j]; o[j]=qScale*sum_i S[i,j]*q[i].
// One thread per (hv, column j); each owns column j of S (stride dS). Head map
// hk = hv % nK (ggml_repeat tile). Matches GDN.step exactly.
struct ScanArgs { uint nV; uint nK; uint dS; float qScale; };

kernel void gdn_scan(
        device const float * q    [[buffer(0)]],
        device const float * k    [[buffer(1)]],
        device const float * v    [[buffer(2)]],
        device const float * g    [[buffer(3)]],
        device const float * beta [[buffer(4)]],
        device       float * S    [[buffer(5)]],
        device       float * o    [[buffer(6)]],
        constant ScanArgs  & a    [[buffer(7)]],
        uint gid [[thread_position_in_grid]]) {
    const uint dS = a.dS;
    if (gid < a.nV * dS) {
        const uint hv = gid / dS;
        const uint j  = gid % dS;
        const uint hk = hv % a.nK;
        device const float * qh = q + hk * dS;
        device const float * kh = k + hk * dS;
        device       float * Sh = S + hv * dS * dS;   // S[i*dS + j]
        const float gamma = exp(g[hv]);
        const float b = beta[hv];
        float sk = 0.0f;
        for (uint i = 0; i < dS; i++) {
            const float s = Sh[i * dS + j] * gamma;
            Sh[i * dS + j] = s;
            sk += s * kh[i];
        }
        const float d = b * (v[hv * dS + j] - sk);
        float acc = 0.0f;
        for (uint i = 0; i < dS; i++) {
            const float s = Sh[i * dS + j] + kh[i] * d;
            Sh[i * dS + j] = s;
            acc += s * qh[i] * a.qScale;
        }
        o[hv * dS + j] = acc;
    }
}

// ---- Deinterleave the fused q|gate attention projection -------------------
// qFull is [q(hd) | gate(hd)] per head; split into q[hd*nH] and gate[hd*nH]
// (Attn.step split loop).
struct SplitArgs { uint hd; uint nH; };

kernel void split_qgate(
        device const float * qFull [[buffer(0)]],
        device       float * q     [[buffer(1)]],
        device       float * gate  [[buffer(2)]],
        constant SplitArgs & a     [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.hd * a.nH) {
        const uint h = gid / a.hd;
        const uint i = gid % a.hd;
        const uint src = h * a.hd * 2;
        q[gid]    = qFull[src + i];
        gate[gid] = qFull[src + a.hd + i];
    }
}

// ---- Paged full attention over the KV cache, one query token --------------
// One threadgroup per query head. K/V live in a lazy pos-major PAGE POOL: kPages
// / vPages are arrays of device pointers, one per P-position page (bindless
// gather). Position t is page kPages[t/P], slot (t%P), head-strided by kvDim =
// hd*nHeadKV. out_h[i] = sum_t softmax(scale*dot(q_h,K_t))*V_t[i], then *=
// sigmoid(gate). Matches Attn.step / the SIMD KVCache page pool.
//
// FLASH / online softmax: threadgroup memory is O(hd), NOT O(T) -- a full
// scores[T] buffer capped context at ~8K on the 32 KB threadgroup budget while
// the page pool advertises 1M. The NSG=4 simdgroups stripe the key TILES
// (lane == key within a TK=32 tile, vit_attn-style, K/V read straight from the
// device pages); each keeps a running (max, denom, dim-sliced acc) and the four
// partials are combined once at the end. hd up to 8*32 = 256 (headDim=256 on
// the 27B), lane owning dims tiisg + 32*u.
struct AttnArgs { uint hd; uint nH; uint nKV; uint T; uint kvDim; uint P; float scale; uint gated; };

// A page table = an array of device pointers, one per P-position page. MSL
// forbids a top-level buffer whose pointee is a pointer, but ALLOWS a device
// pointer as a struct member (tier-2 argument buffer); on Apple Silicon each
// `device half*` slot is just its 8-byte gpuAddress, so the host writes raw
// gpuAddresses -- no MTLArgumentEncoder. KV_MAXP*P bounds the context (2048*512
// = 1M positions).
//
// K/V are stored HALF and accumulated f32. Decode cost is linear in cached
// positions, so at any real context the KV read dominates what a token moves:
// on the dense 1.7B at 4K it is ~0.9 GB against a 0.46 GB weight stream, and
// the pages are ~112 MB per 512 positions, which is what actually bounds
// context on a 3 GB phone. llama.cpp has shipped f16 KV as its default for
// years and the CoreML PagePool on the ANE side already stores fp16, so this
// makes the two backends agree rather than breaking new ground.
#define KV_MAXP 2048
struct KVTable { device const half * pages[KV_MAXP]; };

kernel void attn_head(
        device const float   * q     [[buffer(0)]],
        device const KVTable & kT    [[buffer(1)]],
        device const KVTable & vT    [[buffer(2)]],
        device const float   * gate  [[buffer(3)]],
        device       float   * out   [[buffer(4)]],
        constant AttnArgs    & a     [[buffer(5)]],
        threadgroup float    * shmem [[threadgroup(0)]],
        uint   h     [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint hd = a.hd;
    const ushort NSG = 4, TK = 32;
    const ushort SL = (ushort) ((hd + 31) / 32);   // dim slices per lane (<=8)
    const uint group = a.nH / a.nKV;
    const uint kvh = h / group;
    // shmem = qs[hd] | redM[NSG] | redL[NSG] | redAcc[NSG*hd]  (O(hd), not O(T))
    threadgroup float * qs     = shmem;
    threadgroup float * redM   = qs + hd;
    threadgroup float * redL   = redM + NSG;
    threadgroup float * redAcc = redL + NSG;

    device const float * qh = q + h * hd;
    for (uint i = sgitg * 32 + tiisg; i < hd; i += NSG * 32) { qs[i] = qh[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // This simdgroup's running online-softmax state over its stripe of tiles.
    float m = -INFINITY, l = 0.0f;
    // The V accumulator is sliced FOUR-WIDE: lane L owns dims 4L..4L+3 of each
    // 128-dim slice, so one half4 load per lane per key covers what four
    // scalar loads used to. hd <= 256 on every geometry this kernel accepts
    // (SL <= 8 below assumes it), so two slices are always enough.
    const ushort VS = (ushort) ((hd + 127) / 128);
    float4 acc4[2];
    for (ushort u = 0; u < VS; u++) { acc4[u] = 0.0f; }
    for (uint t0 = sgitg * TK; t0 < a.T; t0 += NSG * TK) {
        const uint tk = min((uint) TK, a.T - t0);
        // lane tiisg owns key t0+tiisg: its whole score dot (K read from device)
        float sc = -INFINITY;
        if (tiisg < tk) {
            const uint t = t0 + tiisg;
            device const half * kt =
                kT.pages[t / a.P] + (t % a.P) * a.kvDim + kvh * hd;
            // half4: each LANE walks a whole head vector while the 32 lanes
            // sit kvDim apart, so a scalar loop issues 32 SCATTERED requests
            // per step and never fills a cache line. Vectorizing quarters the
            // request count and half storage halves the bytes each moves.
            // Vector only when hd % 4 == 0, which is also what makes the
            // kvh * hd row base 8B-aligned; the tail covers the rest.
            const uint hd4 = (hd % 4 == 0) ? hd : 0;
            threadgroup const float4 * q4 = (threadgroup const float4 *) qs;
            device const half4 * k4 = (device const half4 *) kt;
            float4 p4 = 0.0f;
            for (uint i = 0; i < hd4 / 4; i++) {
                p4 += q4[i] * float4(k4[i]);
            }
            float p = p4.x + p4.y + p4.z + p4.w;
            for (uint i = hd4; i < hd; i++) { p += qs[i] * (float) kt[i]; }
            sc = p * a.scale;
        }
        const float mNew = max(m, simd_max(sc));
        const float corr = exp(m - mNew);
        const float w = sc > -INFINITY ? exp(sc - mNew) : 0.0f;
        l = l * corr + simd_sum(w);
        for (ushort u = 0; u < VS; u++) { acc4[u] *= corr; }
        // One half4 per lane per key. The old per-dim form issued tk * SL
        // requests here against the K dot's hd/4 -- four times as many -- and
        // once the K dot was vectorized this became the request count that
        // sets the pace. Widening the slice keeps it perfectly coalesced
        // (lane L takes bytes 8L..8L+7, so a warp sweeps 256 contiguous
        // bytes) while cutting the requests fourfold.
        for (ushort t = 0; t < tk; t++) {
            const float wt = simd_broadcast(w, t);
            const uint tt = t0 + t;
            device const half4 * v4 = (device const half4 *)
                (vT.pages[tt / a.P] + (tt % a.P) * a.kvDim + kvh * hd);
            for (ushort u = 0; u < VS; u++) {
                const uint j4 = tiisg + 32 * u;
                if (4 * j4 < hd) { acc4[u] += wt * float4(v4[j4]); }
            }
        }
        m = mNew;
    }
    if (tiisg == 0) { redM[sgitg] = m; redL[sgitg] = l; }
    // Scatter the four-wide slices back per dim: redAcc stays a plain [NSG,hd]
    // float array, so the combine below is unchanged.
    for (ushort u = 0; u < VS; u++) {
        const uint j = 4 * (tiisg + 32 * u);
        if (j < hd) {
            redAcc[sgitg * hd + j + 0] = acc4[u].x;
            redAcc[sgitg * hd + j + 1] = acc4[u].y;
            redAcc[sgitg * hd + j + 2] = acc4[u].z;
            redAcc[sgitg * hd + j + 3] = acc4[u].w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Combine the NSG partials: rescale each to the global max, sum the denoms,
    // sum the accs. An empty stripe has m=-INF, so exp(m-M)=0 -> no contribution.
    if (sgitg == 0) {
        float M = -INFINITY;
        for (ushort g = 0; g < NSG; g++) { M = max(M, redM[g]); }
        float L = 0.0f;
        for (ushort g = 0; g < NSG; g++) { L += redL[g] * exp(redM[g] - M); }
        const float inv = 1.0f / L;
        for (ushort u = 0; u < SL; u++) {
            const uint j = tiisg + 32 * u;
            if (j < hd) {
                float o = 0.0f;
                for (ushort g = 0; g < NSG; g++) {
                    o += redAcc[g * hd + j] * exp(redM[g] - M);
                }
                o *= inv;
                // Dense qwen3 has no output gate; the hybrid gates by
                // sigmoid(gate). a.gated selects (gate unread when 0).
                out[h * hd + j] =
                    a.gated ? o * sigmoidf(gate[h * hd + j]) : o;
            }
        }
    }
}

// ---- Append this token's K,V rows to the cache at position pos ------------
// K/V caches are [cap, kvDim] HALF; kCur/vCur are [kvDim] f32 activations
// (Attn.step kv.append). The narrowing happens here, once per position, so
// every later read of that position moves half the bytes.
struct KVArgs { uint kvDim; uint pos; };

kernel void kv_append(
        device const float * kCur [[buffer(0)]],
        device const float * vCur [[buffer(1)]],
        device       half  * K    [[buffer(2)]],
        device       half  * V    [[buffer(3)]],
        constant KVArgs    & a    [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.kvDim) {
        K[a.pos * a.kvDim + gid] = (half) kCur[gid];
        V[a.pos * a.kvDim + gid] = (half) vCur[gid];
    }
}

// ==== BATCHED (prefill) kernels: N tokens processed per dispatch ==========
// Activations are token-major [N, dim] (token n at n*dim). The GEMM projections
// stream weights ONCE for the whole batch; these batched cheap/recurrent ops
// mirror their per-token counterparts exactly (validated against the SIMD
// engine), looping N internally for the recurrences (conv, scan) and gridding N
// for the parallel ones.

// embed_batch: dequant N token-embedding rows. woff = token_embd base, K =
// nEmbd, M = N. grid = N*nEmbd.
kernel void embed_batch(
        device const uchar * weights [[buffer(0)]],
        device const int   * ids     [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.M * a.K) {
        const uint n = gid / a.K;
        const uint k = gid % a.K;
        const uint rowBytes = a.K / 128 * 34;
        device const uchar * bp = weights + a.woff
            + (ulong) ids[n] * rowBytes + (ulong) (k / 128) * 34;
        const float d = (float) (*(device const half *) bp);
        const uchar code = (bp[2 + (k % 128) / 4] >> (((k % 128) & 3) * 2)) & 3;
        out[gid] = (float) ((int) code - 1) * d;
    }
}

// gdn_conv_batch: N tokens sequentially per channel; the ring lives in registers
// across the batch and is written back to convState. Mirrors gdn_conv looped.
struct ConvBatchArgs { ulong cwOff; uint convDim; uint dConv; uint N; };
kernel void gdn_conv_batch(
        device const float * qkvMixN [[buffer(0)]],
        device       float * convState [[buffer(1)]],
        device const uchar * weights [[buffer(2)]],
        device       float * outN    [[buffer(3)]],
        constant ConvBatchArgs & a   [[buffer(4)]],
        uint c [[thread_position_in_grid]]) {
    if (c < a.convDim) {
        device const float * cw =
            (device const float *) (weights + a.cwOff) + c * a.dConv;
        const uint kc = a.dConv;
        float ring[3];
        for (uint j = 0; j + 1 < kc; j++) { ring[j] = convState[j * a.convDim + c]; }
        for (uint n = 0; n < a.N; n++) {
            const float cur = qkvMixN[n * a.convDim + c];
            float acc = 0.0f;
            for (uint j = 0; j + 1 < kc; j++) { acc += ring[j] * cw[j]; }
            acc += cur * cw[kc - 1];
            outN[n * a.convDim + c] = siluf(acc);
            for (uint j = 0; j + 2 < kc; j++) { ring[j] = ring[j + 1]; }
            if (kc >= 2) { ring[kc - 2] = cur; }
        }
        for (uint j = 0; j + 1 < kc; j++) { convState[j * a.convDim + c] = ring[j]; }
    }
}

// gdn_scan_batch: N tokens sequentially per (value head hv, column j). q|k|v are
// read from convOutN (q/k already l2-normed in place per token); o written per
// token. Mirrors gdn_scan looped over N.
struct ScanBatchArgs {
    uint nV; uint nK; uint dS; float qScale;
    uint N; uint convDim; uint keyDim; uint valueDim;
};
kernel void gdn_scan_batch(
        device const float * convOutN [[buffer(0)]],
        device const float * gN       [[buffer(1)]],
        device const float * betaN    [[buffer(2)]],
        device       float * S        [[buffer(3)]],
        device       float * oN       [[buffer(4)]],
        constant ScanBatchArgs & a    [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    const uint dS = a.dS;
    if (gid < a.nV * dS) {
        const uint hv = gid / dS;
        const uint j  = gid % dS;
        const uint hk = hv % a.nK;
        device float * Sh = S + hv * dS * dS;
        for (uint n = 0; n < a.N; n++) {
            device const float * q = convOutN + n * a.convDim + hk * dS;
            device const float * k = convOutN + n * a.convDim + a.keyDim + hk * dS;
            device const float * v = convOutN + n * a.convDim + 2 * a.keyDim + hv * dS;
            const float gamma = exp(gN[n * a.nV + hv]);
            const float b = betaN[n * a.nV + hv];
            float sk = 0.0f;
            for (uint i = 0; i < dS; i++) {
                const float s = Sh[i * dS + j] * gamma;
                Sh[i * dS + j] = s;
                sk += s * k[i];
            }
            const float d = b * (v[j] - sk);
            float acc = 0.0f;
            for (uint i = 0; i < dS; i++) {
                const float s = Sh[i * dS + j] + k[i] * d;
                Sh[i * dS + j] = s;
                acc += s * q[i] * a.qScale;
            }
            oN[n * a.valueDim + hv * dS + j] = acc;
        }
    }
}

// gdn_scan_batch2: the delta-rule scan with the recurrent state held RESIDENT
// in registers across the whole N-token sequence, instead of round-tripping the
// dS x dS state matrix through device memory every token (the bandwidth wall of
// gdn_scan_batch: ~4 device passes over S per token). One simdgroup per (value
// head hv, output column j); its 32 lanes split the key dimension i into KS =
// dS/32 slices (ls[KS] in registers). The sk[j]=sum_i S[i,j]*k[i] and
// o[j]=sum_i S[i,j]*q[i] reductions become simd_sum over the key dim. S is
// loaded from device ONCE at entry and stored ONCE at exit. Numerics match
// GDN.step / gdn_scan_batch (the only change is the key-dim reduction order;
// fp non-associativity stays well under the parity gate). Requires dS a
// multiple of 32 and <= 256 (KS <= 8); the host routes other dS to
// gdn_scan_batch. COLS (simdgroups per threadgroup = output columns owned) is
// fixed at 4, independent of dS.
kernel void gdn_scan_batch2(
        device const float * convOutN [[buffer(0)]],
        device const float * gN       [[buffer(1)]],
        device const float * betaN    [[buffer(2)]],
        device       float * S        [[buffer(3)]],
        device       float * oN       [[buffer(4)]],
        constant ScanBatchArgs & a    [[buffer(5)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint dS = a.dS;
    const ushort COLS = 4;                  // output columns owned per threadgroup
    const ushort KS = (ushort) (dS / 32);   // key-dim slices per lane (<=8)
    const uint hv = tgpig.y;
    const uint j  = tgpig.x * COLS + sgitg;
    const uint hk = hv % a.nK;
    device float * Sh = S + hv * dS * dS;   // S[i*dS + j]
    float ls[8];                            // lane owns i = tiisg + 32*m, m<KS
    for (ushort m = 0; m < KS; m++) {
        ls[m] = Sh[(tiisg + 32 * m) * dS + j];
    }
    for (uint n = 0; n < a.N; n++) {
        device const float * q = convOutN + n * a.convDim + hk * dS;
        device const float * k = convOutN + n * a.convDim + a.keyDim + hk * dS;
        device const float * v =
            convOutN + n * a.convDim + 2 * a.keyDim + hv * dS;
        const float gamma = exp(gN[n * a.nV + hv]);
        const float b = betaN[n * a.nV + hv];
        float kk[8], qq[8];
        for (ushort m = 0; m < KS; m++) {
            kk[m] = k[tiisg + 32 * m];
            qq[m] = q[tiisg + 32 * m];
        }
        float ksum = 0.0f;
        for (ushort m = 0; m < KS; m++) { ls[m] *= gamma; ksum += ls[m] * kk[m]; }
        const float sk = simd_sum(ksum);
        const float d = b * (v[j] - sk);
        float osum = 0.0f;
        for (ushort m = 0; m < KS; m++) { ls[m] += kk[m] * d; osum += ls[m] * qq[m]; }
        const float o = simd_sum(osum) * a.qScale;
        if (tiisg == 0) { oN[n * a.valueDim + hv * dS + j] = o; }
    }
    for (ushort m = 0; m < KS; m++) {
        Sh[(tiisg + 32 * m) * dS + j] = ls[m];
    }
}

// split_qgate_batch: deinterleave [q|gate] for N tokens. grid = N*hd*nH.
kernel void split_qgate_batch(
        device const float * qFullN [[buffer(0)]],
        device       float * qN     [[buffer(1)]],
        device       float * gateN  [[buffer(2)]],
        constant SplitArgs & a      [[buffer(3)]],
        constant uint      & N      [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const uint per = a.hd * a.nH;
    if (gid < N * per) {
        const uint n = gid / per;
        const uint r = gid % per;
        const uint h = r / a.hd;
        const uint i = r % a.hd;
        const uint src = n * a.hd * 2 * a.nH + h * a.hd * 2;
        qN[n * per + r]    = qFullN[src + i];
        gateN[n * per + r] = qFullN[src + a.hd + i];
    }
}

// rope_batch: partial NEOX rope for N tokens; token n is at absolute position
// basePos + n. grid = N * nHead * (nRot/2).
struct RopeBatchArgs {
    uint headDim; uint nHead; uint nRot; float base; uint basePos; uint N;
};
kernel void rope_batch(
        device float          * x [[buffer(0)]],
        constant RopeBatchArgs & a [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hf = a.nRot / 2;
    const uint per = a.nHead * hf;
    if (gid < a.N * per) {
        const uint n = gid / per;
        const uint r = gid % per;
        const uint h = r / hf;
        const uint i = r % hf;
        const uint hoff = n * a.nHead * a.headDim + h * a.headDim;
        const float freq = pow(a.base, -2.0f * (float) i / (float) a.nRot);
        const float ang = (float) (a.basePos + n) * freq;
        const float c = cos(ang), s = sin(ang);
        const float p = x[hoff + i];
        const float q = x[hoff + i + hf];
        x[hoff + i]      = p * c - q * s;
        x[hoff + i + hf] = p * s + q * c;
    }
}

// rope_mrope_batch: interleaved M-RoPE for N tokens with per-token 3D
// positions (t,h,w -- an image span's grid coordinates; text has t==h==w,
// where this equals rope_batch exactly). Frequency i takes component i % 3,
// reproducing HF's apply_interleaved_mrope [11,11,10] split for nRot 64.
// grid = N * nHead * (nRot/2).
struct RopeMBatchArgs { uint headDim; uint nHead; uint nRot; float base; uint N; };
kernel void rope_mrope_batch(
        device float           * x    [[buffer(0)]],
        device const int       * pos3 [[buffer(1)]],   // [N][3] t,h,w
        constant RopeMBatchArgs & a   [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hf = a.nRot / 2;
    const uint per = a.nHead * hf;
    if (gid < a.N * per) {
        const uint n = gid / per;
        const uint r = gid % per;
        const uint h = r / hf;
        const uint i = r % hf;
        const uint hoff = n * a.nHead * a.headDim + h * a.headDim;
        const float freq = pow(a.base, -2.0f * (float) i / (float) a.nRot);
        const float ang = (float) pos3[n * 3 + (i % 3)] * freq;
        const float c = cos(ang), s = sin(ang);
        const float p = x[hoff + i];
        const float q = x[hoff + i + hf];
        x[hoff + i]      = p * c - q * s;
        x[hoff + i + hf] = p * s + q * c;
    }
}

// kv_append_batch: write N tokens' K/V into the paged pool via the page table;
// token n -> position basePos+n -> page (basePos+n)/P slot (basePos+n)%P.
// grid = N * kvDim.
struct KVBatchArgs { uint kvDim; uint basePos; uint P; uint N; };
kernel void kv_append_batch(
        device const float   * kCurN [[buffer(0)]],
        device const float   * vCurN [[buffer(1)]],
        device const KVTable & kT    [[buffer(2)]],
        device const KVTable & vT    [[buffer(3)]],
        constant KVBatchArgs & a     [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.N * a.kvDim) {
        const uint n = gid / a.kvDim;
        const uint c = gid % a.kvDim;
        const uint pos = a.basePos + n;
        device half * K = (device half *) kT.pages[pos / a.P];
        device half * V = (device half *) vT.pages[pos / a.P];
        const uint slot = pos % a.P;
        K[slot * a.kvDim + c] = (half) kCurN[n * a.kvDim + c];
        V[slot * a.kvDim + c] = (half) vCurN[n * a.kvDim + c];
    }
}

// attn_batch: causal attention for N query tokens over the paged KV. Query token
// n (absolute position basePos+n) attends to keys 0..basePos+n. One threadgroup
// per (token, head). Same FLASH / online-softmax structure as attn_head (O(hd)
// threadgroup memory, NSG=4 simdgroups striping key tiles), with the causal
// length T set per token.
struct AttnBatchArgs {
    uint hd; uint nH; uint nKV; uint kvDim; uint P; float scale;
    uint basePos; uint N; uint gated;
};
kernel void attn_batch(
        device const float   * qN    [[buffer(0)]],
        device const KVTable & kT    [[buffer(1)]],
        device const KVTable & vT    [[buffer(2)]],
        device const float   * gateN [[buffer(3)]],
        device       float   * outN  [[buffer(4)]],
        constant AttnBatchArgs & a   [[buffer(5)]],
        threadgroup float    * shmem [[threadgroup(0)]],
        uint   tgid  [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint hd = a.hd;
    const ushort NSG = 4, TK = 32;
    const ushort SL = (ushort) ((hd + 31) / 32);
    const uint n  = tgid / a.nH;          // query token in the batch
    const uint h  = tgid % a.nH;
    const uint T  = a.basePos + n + 1;    // causal length for this token
    const uint group = a.nH / a.nKV;
    const uint kvh = h / group;
    threadgroup float * qs     = shmem;
    threadgroup float * redM   = qs + hd;
    threadgroup float * redL   = redM + NSG;
    threadgroup float * redAcc = redL + NSG;

    device const float * qh = qN + n * a.nH * hd + h * hd;
    for (uint i = sgitg * 32 + tiisg; i < hd; i += NSG * 32) { qs[i] = qh[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY, l = 0.0f;
    // The V accumulator is sliced FOUR-WIDE: lane L owns dims 4L..4L+3 of each
    // 128-dim slice, so one half4 load per lane per key covers what four
    // scalar loads used to. hd <= 256 on every geometry this kernel accepts
    // (SL <= 8 below assumes it), so two slices are always enough.
    const ushort VS = (ushort) ((hd + 127) / 128);
    float4 acc4[2];
    for (ushort u = 0; u < VS; u++) { acc4[u] = 0.0f; }
    for (uint t0 = sgitg * TK; t0 < T; t0 += NSG * TK) {
        const uint tk = min((uint) TK, T - t0);
        float sc = -INFINITY;
        if (tiisg < tk) {
            const uint t = t0 + tiisg;
            device const half * kt =
                kT.pages[t / a.P] + (t % a.P) * a.kvDim + kvh * hd;
            // half4: each LANE walks a whole head vector while the 32 lanes
            // sit kvDim apart, so a scalar loop issues 32 SCATTERED requests
            // per step and never fills a cache line. Vectorizing quarters the
            // request count and half storage halves the bytes each moves.
            // Vector only when hd % 4 == 0, which is also what makes the
            // kvh * hd row base 8B-aligned; the tail covers the rest.
            const uint hd4 = (hd % 4 == 0) ? hd : 0;
            threadgroup const float4 * q4 = (threadgroup const float4 *) qs;
            device const half4 * k4 = (device const half4 *) kt;
            float4 p4 = 0.0f;
            for (uint i = 0; i < hd4 / 4; i++) {
                p4 += q4[i] * float4(k4[i]);
            }
            float p = p4.x + p4.y + p4.z + p4.w;
            for (uint i = hd4; i < hd; i++) { p += qs[i] * (float) kt[i]; }
            sc = p * a.scale;
        }
        const float mNew = max(m, simd_max(sc));
        const float corr = exp(m - mNew);
        const float w = sc > -INFINITY ? exp(sc - mNew) : 0.0f;
        l = l * corr + simd_sum(w);
        for (ushort u = 0; u < VS; u++) { acc4[u] *= corr; }
        // One half4 per lane per key. The old per-dim form issued tk * SL
        // requests here against the K dot's hd/4 -- four times as many -- and
        // once the K dot was vectorized this became the request count that
        // sets the pace. Widening the slice keeps it perfectly coalesced
        // (lane L takes bytes 8L..8L+7, so a warp sweeps 256 contiguous
        // bytes) while cutting the requests fourfold.
        for (ushort t = 0; t < tk; t++) {
            const float wt = simd_broadcast(w, t);
            const uint tt = t0 + t;
            device const half4 * v4 = (device const half4 *)
                (vT.pages[tt / a.P] + (tt % a.P) * a.kvDim + kvh * hd);
            for (ushort u = 0; u < VS; u++) {
                const uint j4 = tiisg + 32 * u;
                if (4 * j4 < hd) { acc4[u] += wt * float4(v4[j4]); }
            }
        }
        m = mNew;
    }
    if (tiisg == 0) { redM[sgitg] = m; redL[sgitg] = l; }
    // Scatter the four-wide slices back per dim: redAcc stays a plain [NSG,hd]
    // float array, so the combine below is unchanged.
    for (ushort u = 0; u < VS; u++) {
        const uint j = 4 * (tiisg + 32 * u);
        if (j < hd) {
            redAcc[sgitg * hd + j + 0] = acc4[u].x;
            redAcc[sgitg * hd + j + 1] = acc4[u].y;
            redAcc[sgitg * hd + j + 2] = acc4[u].z;
            redAcc[sgitg * hd + j + 3] = acc4[u].w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        float M = -INFINITY;
        for (ushort g = 0; g < NSG; g++) { M = max(M, redM[g]); }
        float L = 0.0f;
        for (ushort g = 0; g < NSG; g++) { L += redL[g] * exp(redM[g] - M); }
        const float inv = 1.0f / L;
        for (ushort u = 0; u < SL; u++) {
            const uint j = tiisg + 32 * u;
            if (j < hd) {
                float o = 0.0f;
                for (ushort g = 0; g < NSG; g++) {
                    o += redAcc[g * hd + j] * exp(redM[g] - M);
                }
                o *= inv;
                const uint oi = n * a.nH * hd + h * hd + j;
                outN[oi] = a.gated ? o * sigmoidf(gateN[oi]) : o;
            }
        }
    }
}

// ==== ViT (Qwen3-VL vision tower) kernels ==================================
// The mmproj tower on the GPU: f16 weights (dequanted once at load by
// MetalViT into plain half buffers) x f32 activations, f32 accumulation.
// Each kernel mirrors the CPU ViT (SIMD/ViT.swift) op for op -- that engine
// is the oracle (`gadeon-cli --vit` cross-gates the two forwards).

// f16-weight simdgroup GEMM: dst[N,M] = X[N,K] @ W[M,K]^T. W is a plain
// row-major [M][K] half buffer (ggml's native [out][in] order, so the CPU
// path's load-time transpose does not exist here). Structure mirrors
// q2_0_gemm_mm_h with the Q2_0 block walk replaced by direct half loads and
// a K-tail guard (ViT K values are not all multiples of 32; out-of-range
// lanes load 0, which contributes nothing). Same 64(M) x 32(N) tile, half
// smem tiles (6144 B), f32 accumulate, and the bounds-checked f32 spill
// path for partial tiles (host allocates 8192 B smem then).
struct F16WArgs { uint K; uint M; };

kernel void f16w_gemm_mm(
        device const half  * A       [[buffer(0)]],
        device const float * X       [[buffer(1)]],
        device       float * dst     [[buffer(2)]],
        constant F16WArgs  & a       [[buffer(3)]],
        constant uint      & N       [[buffer(4)]],
        threadgroup uchar  * shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int K = (int) a.K, M = (int) a.M;
    const int NR0 = 64, NR1 = 32, NK = 32, NL0 = NK / 16, NL1 = NK / 8;
    threadgroup half * sa = (threadgroup half *) (shmem);
    threadgroup half * sb = (threadgroup half *) (shmem + 4096);

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    const short nr0 = (M - r0 < NR0) ? (short) (M - r0) : NR0;
    const short nr1 = ((int) N - r1 < NR1) ? (short) ((int) N - r1) : NR1;
    const short lr0 = ((short) (tiitg / NL0) < nr0) ? (short) (tiitg / NL0)
                                                    : nr0 - 1;
    const short lr1 = ((short) (tiitg / NL1) < nr1) ? (short) (tiitg / NL1)
                                                    : nr1 - 1;
    const short il0 = tiitg % NL0;
    device const half  * arow = A + (ulong) (r0 + lr0) * K;
    device const float * yrow = X + (ulong) (r1 + lr1) * K;
    const short iy = 8 * (tiitg % NL1);

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const int k = loop_k + il0 * 16 + i;
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            sa[64 * (8 * sx + sy) + 8 * ly + lx] =
                k < K ? arow[k] : (half) 0.0h;
        }
        {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            threadgroup half * bp = sb + 64 * (4 * sx + sy) + 8 * ly;
            for (short i = 0; i < 8; i++) {
                const int k = loop_k + iy + i;
                bp[i] = k < K ? (half) yrow[k] : (half) 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);
        #pragma clang loop unroll(full)
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= (int) N) {
        device float * C = dst + (r0 + 32 * (sgitg & 1))
            + (ulong) (r1 + 16 * (sgitg >> 1)) * M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong) M * (i / 4),
                            M, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp = (threadgroup float *) shmem
            + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * NR0 * (i / 4),
                            NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = dst + r0 + (ulong) (r1 + j) * M;
                threadgroup float * C = temp + j * NR0;
                for (int i = 0; i < nr0; i++) { D[i] = C[i]; }
            }
        }
    }
}

// LayerNorm with bias over rows of length n (the ViT is pre-LN --
// mean-centered with a learned bias, unlike the LM's rmsnorm). x -> y so x
// survives for the residual. One threadgroup per row; the sum and
// sum-of-squares reduce together (shmem: 32 floats each).
struct LNArgs { uint n; float eps; };

kernel void vit_layernorm(
        device const float * x     [[buffer(0)]],
        device const float * w     [[buffer(1)]],
        device const float * b     [[buffer(2)]],
        device       float * y     [[buffer(3)]],
        constant LNArgs    & a     [[buffer(4)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device const float * xr = x + (ulong) row * a.n;
    device       float * yr = y + (ulong) row * a.n;
    float s1 = 0.0f, s2 = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) {
        const float v = xr[i];
        s1 += v;
        s2 += v * v;
    }
    s1 = simd_sum(s1);
    s2 = simd_sum(s2);
    if (tii == 0) { shmem[sgi] = s1; shmem[32 + sgi] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
        float v1 = (tii < (ntg + 31) / 32) ? shmem[tii] : 0.0f;
        float v2 = (tii < (ntg + 31) / 32) ? shmem[32 + tii] : 0.0f;
        v1 = simd_sum(v1);
        v2 = simd_sum(v2);
        if (tii == 0) { shmem[0] = v1; shmem[32] = v2; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = shmem[0] / (float) a.n;
    const float inv =
        1.0f / sqrt(shmem[32] / (float) a.n - mean * mean + a.eps);
    for (uint i = tid; i < a.n; i += ntg) {
        yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
    }
}

// x[r*m + c] += bias[c] for every row (the ViT projections carry biases).
kernel void add_bias_rows(device float * x [[buffer(0)]],
                          device const float * bias [[buffer(1)]],
                          constant uint & m [[buffer(2)]],
                          constant uint & total [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid < total) { x[gid] += bias[gid % m]; }
}

// ggml's tanh-approximation GELU (matches ViT.gelu / vvtanhf numerics).
kernel void gelu_tanh(device float * x [[buffer(0)]],
                      constant uint & n [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        const float v = x[gid];
        const float t =
            precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v));
        x[gid] = 0.5f * v * (1.0f + t);
    }
}

// Vision M-RoPE over the q or k span of the fused qkv rows: cos/sin come
// precomputed per (sequence slot, pair) from the host (ViT.ropeTables --
// row-keyed first half, column-keyed second), pairs (j, j+hf) within each
// head. `off` selects q (0) or k (e) inside a 3e row.
struct VRopeArgs { uint rowStride; uint off; uint headDim; uint nHead; uint N; };

kernel void vit_rope(
        device float       * x    [[buffer(0)]],
        device const float * cosT [[buffer(1)]],
        device const float * sinT [[buffer(2)]],
        constant VRopeArgs & a    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hf = a.headDim / 2;
    const uint per = a.nHead * hf;
    if (gid < a.N * per) {
        const uint s = gid / per;
        const uint r = gid % per;
        const uint h = r / hf;
        const uint j = r % hf;
        const uint base = s * a.rowStride + a.off + h * a.headDim;
        const float c = cosT[s * hf + j];
        const float sn = sinT[s * hf + j];
        const float x0 = x[base + j];
        const float x1 = x[base + j + hf];
        x[base + j]      = x0 * c - x1 * sn;
        x[base + j + hf] = x0 * sn + x1 * c;
    }
}

// Bidirectional attention over the fused qkv rows ([N][3e]: q | k | v
// spans, head-major within each), flash-style: one threadgroup per (head,
// tile of ntg/32 queries), one simdgroup per query, K/V streamed in TK-key
// tiles through threadgroup memory shared by all the simdgroups. Within a
// tile the softmax is LANE-PER-KEY -- each lane computes its key's whole
// dot (float4-vectorized) and one exp, and the online-softmax state (m, l,
// lane-sliced acc) updates once per TILE, not per key (a per-key simd_sum +
// exp chain is latency-bound, measured ~2x slower; half-staged tiles and
// TK=16 both also measured SLOWER -- 2-byte scalar loads, and per-tile
// overhead outweighing the occupancy gain). No causal mask and no
// GQA; the context lands at the head's slot of out [N][e]. The 4 x 32 lane
// slice caps hd at 128 (the host asserts).
struct VAttnArgs { uint n; uint e; uint hd; uint nHead; float scale; };

kernel void vit_attn(
        device const float * qkv   [[buffer(0)]],
        device       float * out   [[buffer(1)]],
        constant VAttnArgs & a     [[buffer(2)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  tpitg [[thread_position_in_threadgroup]],
        uint3  ntg3  [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint tid = tpitg.x, ntg = ntg3.x;
    const uint TK = 32;                       // keys per streamed tile
    const uint h = tgpig.y;
    const uint s = tgpig.x * (ntg / 32) + sgitg;
    const uint row = 3 * a.e;
    threadgroup float * kt = shmem;           // [TK][hd]
    threadgroup float * vt = shmem + TK * a.hd;
    threadgroup float * qs = shmem + 2 * TK * a.hd;   // [ntg/32][hd]
    float acc[4] = { 0, 0, 0, 0 };            // lane owns dims tiisg + 32u
    if (s < a.n) {
        device const float * qh = qkv + (ulong) s * row + h * a.hd;
        for (uint j = tiisg; j < a.hd; j += 32) { qs[sgitg * a.hd + j] = qh[j]; }
    }
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = 0; t0 < a.n; t0 += TK) {
        const uint tk = min(TK, a.n - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < tk * a.hd; i += ntg) {
            const uint t = i / a.hd, j = i % a.hd;
            kt[i] = qkv[(ulong) (t0 + t) * row + a.e + h * a.hd + j];
            vt[i] = qkv[(ulong) (t0 + t) * row + 2 * a.e + h * a.hd + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sc = -INFINITY;
        if (s < a.n && tiisg < tk) {
            // hd rows start 16B-aligned whenever hd % 4 == 0 (every Qwen
            // tower; the scalar tail covers any other geometry).
            const uint hd4 = a.hd & ~3u;
            threadgroup const float4 * q4 =
                (threadgroup const float4 *) (qs + sgitg * a.hd);
            threadgroup const float4 * k4 =
                (threadgroup const float4 *) (kt + tiisg * a.hd);
            float4 p4 = 0.0f;
            for (uint i = 0; i < hd4 / 4; i++) { p4 += q4[i] * k4[i]; }
            float p = p4.x + p4.y + p4.z + p4.w;
            for (uint i = hd4; i < a.hd; i++) {
                p += qs[sgitg * a.hd + i] * kt[tiisg * a.hd + i];
            }
            sc = p * a.scale;
        }
        const float mNew = max(m, simd_max(sc));
        const float corr = exp(m - mNew);     // 0 on the first tile
        const float w = sc > -INFINITY ? exp(sc - mNew) : 0.0f;
        l = l * corr + simd_sum(w);
        for (ushort u = 0; u < 4; u++) { acc[u] *= corr; }
        for (uint t = 0; t < tk; t++) {
            const float wt = simd_broadcast(w, (ushort) t);
            // Fixed 4-slice guarded loop, NOT a hoisted per-lane bound: the
            // compile-time trip count unrolls; a runtime bound measured 25%
            // slower.
            for (ushort u = 0; u < 4; u++) {
                const uint j = tiisg + 32 * u;
                if (j < a.hd) { acc[u] += wt * vt[t * a.hd + j]; }
            }
        }
        m = mNew;
    }
    if (s < a.n) {
        device float * o = out + (ulong) s * a.e + h * a.hd;
        const float inv = 1.0f / l;
        for (ushort u = 0; u < 4; u++) {
            const uint j = tiisg + 32 * u;
            if (j < a.hd) { o[j] = acc[u] * inv; }
        }
    }
}

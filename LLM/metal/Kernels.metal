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
// Weights live in a handful of no-copy buffers over the mmap'd GGUF; every
// kernel that reads a weight takes a byte offset into the one it was handed.
// [block-layout]

#include <metal_stdlib>
using namespace metal;

inline float siluf(float x)    { return x / (1.0f + exp(-x)); }
inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float softplusf(float x) {
    return max(x, 0.0f) + log(1.0f + exp(-fabs(x)));
}

// Q2_0 ternary mat-vec: out[m] = sum_k W[m,k] * x[k]
// W ne0=K (input, fastest), ne1=M (rows), at byte offset woff -- 64-bit,
// because the weight buffer is the whole GGUF (>4 GB). The two-blocks-in-
// flight shape below is TUNED, not incidental; do not simplify it without
// re-measuring. [gemv-unroll]
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
    // Two blocks (ib0, ib1) per iteration, held in DISTINCT scalar arrays
    // (not a 2D array -- a loop-indexed yl[u][i] spills to local memory and
    // regresses):
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
                device const uchar * b0 =
                    W + row * rowBytes + (ulong) ib0 * 34;
                device const uchar * b1 =
                    W + row * rowBytes + (ulong) ib1 * 34;
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
        for (ushort i = 0; i < SW; i++) {
            yl[i] = y[i];
            sy += y[i];
        }
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

kernel void q2_x_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 128;
    const ushort TPB = 8, SW = 16, STEP = 32 / TPB;
    const ulong rowBytes = (ulong) nblk * 34;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort il  = (tiisg % TPB) * SW;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
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
                device const uchar * b0 =
                    W + row * rowBytes + (ulong) ib0 * 34;
                device const uchar * b1 =
                    W + row * rowBytes + (ulong) ib1 * 34;
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
                acc[r] += d0 * (2.0f * lo0 + 4.0f * hi0 - 3.0f * sy0)
                        + d1 * (2.0f * lo1 + 4.0f * hi1 - 3.0f * sy1);
            }
        }
    }
    for (; ib < nblk; ib += STEP) {
        device const float * y = x + (ulong) ib * 128 + il;
        float yl[16];
        float sy = 0.0f;
        for (ushort i = 0; i < SW; i++) {
            yl[i] = y[i];
            sy += y[i];
        }
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
                acc[r] += d * (2.0f * lo + 4.0f * hi - 3.0f * sy);
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

kernel void q2_x_gemv_un4(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 4;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 128;
    const ushort TPB = 8, SW = 16, STEP = 32 / TPB;
    const ulong rowBytes = (ulong) nblk * 34;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort il  = (tiisg % TPB) * SW;
    float acc[4] = { 0, 0, 0, 0 };
    uint ib = grp;
    for (; ib + 3 * STEP < nblk; ib += 4 * STEP) {
        float yl0[16], yl1[16], yl2[16], yl3[16];
        float sy0 = 0, sy1 = 0, sy2 = 0, sy3 = 0;
        device const float * y0 = x + (ulong) ib * 128 + il;
        device const float * y1 = y0 + (ulong) STEP * 128;
        device const float * y2 = y1 + (ulong) STEP * 128;
        device const float * y3 = y2 + (ulong) STEP * 128;
        for (ushort i = 0; i < SW; i++) {
            yl0[i] = y0[i]; sy0 += y0[i];
            yl1[i] = y1[i]; sy1 += y1[i];
            yl2[i] = y2[i]; sy2 += y2[i];
            yl3[i] = y3[i]; sy3 += y3[i];
        }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * rp = W + row * rowBytes;
                float sum = 0.0f;
                for (ushort u = 0; u < 4; u++) {
                    device const uchar * bp = rp + (ulong) (ib + u * STEP) * 34;
                    const float d = (float) (*(device const half *) bp);
                    device const uchar * qs = bp + 2 + il / 4;
                    thread float * yl = u == 0 ? yl0 : (u == 1 ? yl1
                                      : (u == 2 ? yl2 : yl3));
                    const float sy = u == 0 ? sy0 : (u == 1 ? sy1
                                   : (u == 2 ? sy2 : sy3));
                    float lo = 0.0f, hi = 0.0f;
                    for (ushort i = 0; i < SW; i++) {
                        const uchar c = (qs[i >> 2] >> ((i & 3) * 2)) & 3;
                        if (c & 1) { lo += yl[i]; }
                        if (c & 2) { hi += yl[i]; }
                    }
                    sum += d * (2.0f * lo + 4.0f * hi - 3.0f * sy);
                }
                acc[r] += sum;
            }
        }
    }
    for (; ib < nblk; ib += STEP) {
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
                    const uchar c = (qs[i >> 2] >> ((i & 3) * 2)) & 3;
                    if (c & 1) { lo += yl[i]; }
                    if (c & 2) { hi += yl[i]; }
                }
                acc[r] += d * (2.0f * lo + 4.0f * hi - 3.0f * sy);
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

kernel void q2_x_gemv_r16(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 16;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 128;
    const ushort TPB = 8, SW = 16, STEP = 32 / TPB;
    const ulong rowBytes = (ulong) nblk * 34;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort il  = (tiisg % TPB) * SW;
    float acc[16];
    for (ushort r = 0; r < NR0; r++) { acc[r] = 0.0f; }
    for (uint ib = grp; ib < nblk; ib += STEP) {
        device const float * y = x + (ulong) ib * 128 + il;
        float yl[16];
        float sy = 0.0f;
        for (ushort i = 0; i < SW; i++) {
            yl[i] = y[i];
            sy += y[i];
        }
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
                acc[r] += d * (2.0f * lo + 4.0f * hi - 3.0f * sy);
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

// Q2_0 batched mat-mat (prefill): out[N,M] = X[N,K] @ W[K,M]
// Token-major: X[col*K + k], out[col*M + m]. Each threadgroup owns one weight
// row m and a tile of TN=8 token-columns, so the weight row is STREAMED
// ONCE and reused across the 8 columns -- the weight-amortization that makes
// prefill scale (token-by-token re-streams the whole 7 GB per token). 32
// lanes cooperate: lane
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
    // each block's SW=16 codes of a row are decoded ONCE and dotted against
    // all
    // TN columns' activation slices (weight reused across cols), while the row
    // tile reuses each column's slice (activation reused across rows).
    // simd_sum
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
            for (ushort i = 0; i < SW; i++) {
                yl[t][i] = xc[i];
                sy[t] += xc[i];
            }
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

// Quantized simdgroup-matrix GEMM (prefill): out[N,M] = X[N,K] @ W[K,M].
// One body for all three block types at both tile precisions. Weight rows are
// [K,M]; X is token-major f32; out[n*M+m] matches GQ.matvec's column order.
// [gemm-tiles]
struct block_q2_0 { half d; uchar qs[32]; };
struct block_q4_0 { half d; uchar qs[16]; };
struct block_q8_0 { half d; char qs[32]; };

// One 16-element sub-block into a 4x4 register tile. The `_h` twins write
// half DIRECTLY rather than through a float intermediate. Q2_0 codes are
// 2-bit, 4 per byte, w = (code-1)*d; Q8_0 is one signed byte each, w = q*d;
// Q4_0 splits a byte into element j (LOW nibble) and j+16 (HIGH), w =
// (q-8)*d -- so its two sub-blocks are every low nibble and every high one
// rather than two contiguous spans. [gemm-tiles]
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

static inline void dq_q2_x(device const block_q2_0 * xb, short il,
                           thread float4x4 & reg) {
    device const uchar * qs = xb->qs;
    const float d = (float) xb->d;
    const int bo = il * 4;
    for (int i = 0; i < 4; i++) {
        const uchar b = qs[bo + i];
        reg[i][0] = ((float) (2 * ((b >> 0) & 3)) - 3.0f) * d;
        reg[i][1] = ((float) (2 * ((b >> 2) & 3)) - 3.0f) * d;
        reg[i][2] = ((float) (2 * ((b >> 4) & 3)) - 3.0f) * d;
        reg[i][3] = ((float) (2 * ((b >> 6) & 3)) - 3.0f) * d;
    }
}

static inline void dq_q2_x_h(device const block_q2_0 * xb, short il,
                             thread half4x4 & reg) {
    device const uchar * qs = xb->qs;
    const half d = xb->d;
    const int bo = il * 4;
    for (int i = 0; i < 4; i++) {
        const uchar b = qs[bo + i];
        reg[i][0] = ((half) (2 * ((b >> 0) & 3)) - 3.0h) * d;
        reg[i][1] = ((half) (2 * ((b >> 2) & 3)) - 3.0h) * d;
        reg[i][2] = ((half) (2 * ((b >> 4) & 3)) - 3.0h) * d;
        reg[i][3] = ((half) (2 * ((b >> 6) & 3)) - 3.0h) * d;
    }
}

static inline void dq_q4_0(device const block_q4_0 * xb, short il,
                           thread float4x4 & reg) {
    device const uchar * qs = xb->qs;
    const float d = (float) xb->d;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            const uchar byte = qs[i * 4 + j];
            const uchar code = il == 0 ? (byte & 0x0F) : (byte >> 4);
            reg[i][j] = ((float) code - 8.0f) * d;
        }
    }
}

static inline void dq_q4_0_h(device const block_q4_0 * xb, short il,
                             thread half4x4 & reg) {
    device const uchar * qs = xb->qs;
    const half d = xb->d;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            const uchar byte = qs[i * 4 + j];
            const uchar code = il == 0 ? (byte & 0x0F) : (byte >> 4);
            reg[i][j] = ((half) code - 8.0h) * d;
        }
    }
}

static inline void dq_q8_0(device const block_q8_0 * xb, short il,
                           thread float4x4 & reg) {
    device const char * qs = xb->qs;
    const float d = (float) xb->d;
    const int bo = il * 16;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            reg[i][j] = (float) qs[bo + i * 4 + j] * d;
        }
    }
}

static inline void dq_q8_0_h(device const block_q8_0 * xb, short il,
                             thread half4x4 & reg) {
    device const char * qs = xb->qs;
    const half d = xb->d;
    const int bo = il * 16;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            reg[i][j] = (half) qs[bo + i * 4 + j] * d;
        }
    }
}

// The 8x8 matrix-unit product over one staged NK slice: four A tiles against
// two B tiles, into the eight output tiles this simdgroup owns. Shared by
// every GEMM here, quantized or not. [gemm-tiles]
template <typename Reg>
inline void simd_mm_slice(threadgroup const Reg * sa,
                          threadgroup const Reg * sb,
                          thread simdgroup_float8x8 (&mc)[8],
                          ushort sgitg) {
    simdgroup_matrix<Reg, 8, 8> ma[4], mb[2];
    threadgroup const Reg * lsma = sa + 4 * 64 * (sgitg % 2);
    threadgroup const Reg * lsmb = sb + 2 * 64 * (sgitg / 2);
    #pragma clang loop unroll(full)
    for (short ik = 0; ik < 4; ik++) {        // NK / 8
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

// Write the accumulated tiles to dst, or stage them through threadgroup
// memory when this tile runs off the end of M or N. The spill stages as F32
// in the SAME shmem, which is why a partial tile needs 8192 B even at half
// precision. [gemm-tiles]
inline void store_mm_tile(thread simdgroup_float8x8 (&mc)[8],
                          device float * dst, threadgroup uchar * shmem,
                          int r0, int r1, int M, int N,
                          short nr0, short nr1,
                          ushort tiitg, ushort sgitg) {
    const int NR0 = 64, NR1 = 32;
    if (r0 + NR0 <= M && r1 + NR1 <= N) {
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

// `Reg` is the TILE precision (f32 or half) and drives the threadgroup
// layout: half tiles halve the staging memory, which raises resident-
// threadgroup occupancy and is the shipping default. `NSUB` is 16-element
// sub-blocks per quantization block and `QK` its weight count -- together
// they are the block walk, which is the one place the three types genuinely
// differ. DQ is a TEMPLATE parameter, not a function pointer argument, so
// the call is resolved at compile time by the language rather than by hoping
// the optimizer devirtualizes it. [gemm-tiles]
template <typename Block, typename Reg, short NSUB, int QK,
          void (*DQ)(device const Block *, short, thread matrix<Reg, 4, 4> &)>
inline void gemm_mm_impl(
        device const uchar * weights,
        device const float * X,
        device       float * dst,
        constant GemvArgs  & a,
        constant uint      & N,
        threadgroup uchar  * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort sgitg) {
    const int K = (int) a.K, M = (int) a.M;
    const int NR0 = 64, NR1 = 32, NK = 32, NL0 = NK / 16, NL1 = NK / 8;
    // The f32 tiles need 8192 for sa where half needs 4096; sb follows it.
    const ulong sbOff = sizeof(Reg) == 2 ? 4096 : 8192;
    threadgroup Reg * sa = (threadgroup Reg *) (shmem);
    threadgroup Reg * sb = (threadgroup Reg *) (shmem + sbOff);
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
    const ulong rowBytes = (ulong) (K / QK) * sizeof(Block);
    device const Block * x = (device const Block *)
        (weights + a.woff + rowBytes * (r0 + lr0));
    const short iy = 8 * (tiitg % NL1);
    device const float * y = X + (ulong) (r1 + lr1) * K + iy;
    simdgroup_float8x8 mc[8];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        matrix<Reg, 4, 4> temp_a;
        DQ(x, il, temp_a);
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
            threadgroup Reg * bp = sb + 64 * ib + 8 * ly;
            for (short i = 0; i < 8; i++) { bp[i] = (Reg) y[i]; }
        }
        // NSUB == 2 (Q4_0, Q8_0) leaves il at il0 and advances every step,
        // which is what those types' unconditional x += 1 was; NSUB == 8
        // (Q2_0) walks four steps inside one 128-weight block first.
        il = (il + 2 < NSUB) ? il + 2 : il % 2;
        x  = (il < 2) ? x + 1 : x;
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simd_mm_slice(sa, sb, mc, sgitg);
    }
    store_mm_tile(mc, dst, shmem, r0, r1, M, (int) N, nr0, nr1,
                  tiitg, sgitg);
}

// The six instantiations. A kernel cannot be a template in MSL, so each is a
// named entry point over the shared body; MetalEnc.gemm picks one by the
// tensor's type and LLM_F16_TILES.
#define GEMM_MM_KERNEL(NAME, BLOCK, REG, NSUB, QK, DQ)                      \
kernel void NAME(                                                           \
        device const uchar * weights [[buffer(0)]],                         \
        device const float * X       [[buffer(1)]],                         \
        device       float * dst     [[buffer(2)]],                         \
        constant GemvArgs  & a       [[buffer(3)]],                         \
        constant uint      & N       [[buffer(4)]],                         \
        threadgroup uchar  * shmem   [[threadgroup(0)]],                    \
        uint3  tgpig [[threadgroup_position_in_grid]],                      \
        ushort tiitg [[thread_index_in_threadgroup]],                       \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {                  \
    gemm_mm_impl<BLOCK, REG, NSUB, QK, DQ>(                                 \
        weights, X, dst, a, N, shmem, tgpig, tiitg, sgitg);                 \
}

GEMM_MM_KERNEL(q2_0_gemm_mm,   block_q2_0, float, 8, 128, dq_q2_0)
GEMM_MM_KERNEL(q2_0_gemm_mm_h, block_q2_0, half,  8, 128, dq_q2_0_h)
GEMM_MM_KERNEL(q2_x_gemm_mm,   block_q2_0, float, 8, 128, dq_q2_x)
GEMM_MM_KERNEL(q2_x_gemm_mm_h, block_q2_0, half,  8, 128, dq_q2_x_h)
GEMM_MM_KERNEL(q4_0_gemm_mm,   block_q4_0, float, 2,  32, dq_q4_0)
GEMM_MM_KERNEL(q4_0_gemm_mm_h, block_q4_0, half,  2,  32, dq_q4_0_h)
GEMM_MM_KERNEL(q8_0_gemm_mm,   block_q8_0, float, 2,  32, dq_q8_0)
GEMM_MM_KERNEL(q8_0_gemm_mm_h, block_q8_0, half,  2,  32, dq_q8_0_h)


// Q2_0 row dequant (token embedding): out[k] = (code-1)*d
// `woff` already points at the wanted row's first block; one thread per
// weight.
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

kernel void q2_x_dequant_row(
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
        out[gid] = (float) (2 * (int) code - 3) * d;
    }
}

// Q4_0 mat-vec: out[m] = sum_k W[m,k] * x[k]
// Q4_0 block: 18 bytes, 32 weights, { half d; uchar qs[16] }. Byte j carries
// element j in its LOW nibble and element j+16 in its HIGH one, w = (q-8)*d.
// That is the same codebook and offset gemma's INT4 QAT uses, which is what
// makes the repack a code-for-code transfer rather than a re-quantization.
// One simdgroup per NR0 output rows, lanes striping whole blocks; simd_sum
// reduces. Mirrors GQ.matvec's q4_0 arm, which is the oracle.
kernel void q4_0_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 32;
    const ulong rowBytes = (ulong) nblk * 18;
    device const uchar * W = weights + a.woff;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint ib = tiisg; ib < nblk; ib += 32) {
        device const float * y = x + (ulong) ib * 32;
        float yl[16], yh[16];
        for (ushort i = 0; i < 16; i++) {
            yl[i] = y[i];
            yh[i] = y[i + 16];
        }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + row * rowBytes + (ulong) ib * 18;
                const float d = (float) (*(device const half *) bp);
                device const uchar * qs = bp + 2;
                float s = 0.0f;
                for (ushort i = 0; i < 16; i++) {
                    const uchar b = qs[i];
                    s += ((float) (b & 0x0F) - 8.0f) * yl[i];
                    s += ((float) (b >> 4) - 8.0f) * yh[i];
                }
                acc[r] += s * d;
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

// Q4_0 span dequant: `woff` points at the first block of the span
// The gemma embedding gathers run through this: token_embd one whole row, the
// per-layer table one 256-wide slice of its 8960-wide row. Both land on block
// boundaries, so the span always starts at a block.
kernel void q4_0_dequant_row(
        device const uchar * weights [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        constant GemvArgs  & a       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.K) {
        const uint ib = gid / 32;
        const uint j  = gid % 32;
        device const uchar * bp = weights + a.woff + (ulong) ib * 18;
        const float d = (float) (*(device const half *) bp);
        const uchar b = bp[2 + (j % 16)];
        const uchar code = (j < 16) ? (b & 0x0F) : (b >> 4);
        out[gid] = ((float) code - 8.0f) * d;
    }
}

// BF16 is the top 16 bits of the f32 bit pattern. Every gemma norm is stored
// that way (bit-exact and half the size), so reading one through the f32 path
// would fuse two weights into one garbage float -- the kernel and the
// repacker are each correct alone and only their contract is wrong.
inline float bf16_at(device const uchar * p, uint i) {
    const ushort bits = *(device const ushort *) (p + 2 * i);
    return as_type<float>((uint) bits << 16);
}

// Q8_0 mat-vec
// Q8_0 block: 34 bytes, 32 weights, { half d; char qs[32] }, w = q * d. The
// QAT spends 8 bits on the per-layer gate and projection -- 0.5% of the model
// whose bit width moves text KL 3x, far more than the 45%-of-the-model
// per-layer table does -- and on the vision tower.
kernel void q8_0_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 32;
    const ulong rowBytes = (ulong) nblk * 34;
    device const uchar * W = weights + a.woff;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint ib = tiisg; ib < nblk; ib += 32) {
        device const float * y = x + (ulong) ib * 32;
        float yl[32];
        for (ushort i = 0; i < 32; i++) { yl[i] = y[i]; }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + row * rowBytes + (ulong) ib * 34;
                const float d = (float) (*(device const half *) bp);
                device const char * qs = (device const char *) (bp + 2);
                float s = 0.0f;
                for (ushort i = 0; i < 32; i++) { s += (float) qs[i] * yl[i]; }
                acc[r] += s * d;
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

kernel void q8_0_dequant_row(
        device const uchar * weights [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        constant GemvArgs  & a       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.K) {
        const uint ib = gid / 32;
        const uint j  = gid % 32;
        device const uchar * bp = weights + a.woff + (ulong) ib * 34;
        const float d = (float) (*(device const half *) bp);
        const char q = ((device const char *) (bp + 2))[j];
        out[gid] = (float) q * d;
    }
}

// dense mat-vec for the weights the QAT left unquantized
// modules_to_not_convert keeps a handful of matrices at full width, and the
// per-layer model projection is the big one (13.7 M weights). One simdgroup
// per NR0 rows, the 32 lanes striding K; simd_sum reduces. Mirrors
// GQ.denseMatvec.
kernel void bf16_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const uint row0 = tgpig.x * NR0;
    const ulong rowBytes = (ulong) a.K * 2;
    device const uchar * W = weights + a.woff;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint r = 0; r < NR0; r++) {
        const uint row = row0 + r;
        if (row < a.M) {
            device const uchar * rp = W + (ulong) row * rowBytes;
            float s = 0.0f;
            for (uint k = tiisg; k < a.K; k += 32) {
                s += bf16_at(rp, k) * x[k];
            }
            acc[r] = s;
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

kernel void f32_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const uint row0 = tgpig.x * NR0;
    device const float * W = (device const float *) (weights + a.woff);
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint r = 0; r < NR0; r++) {
        const uint row = row0 + r;
        if (row < a.M) {
            device const float * rp = W + (ulong) row * a.K;
            float s = 0.0f;
            for (uint k = tiisg; k < a.K; k += 32) { s += rp[k] * x[k]; }
            acc[r] = s;
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

// One threadgroup's sum of a per-thread value: simd_sum within each
// simdgroup, then simd_sum over those partials in simdgroup 0. `shmem` needs
// one float per simdgroup (<= 32) and is CLOBBERED, so two reductions in one
// kernel must pass disjoint regions -- see vit_layernorm. [tg-reduce]
inline float tg_reduce_sum(float v, threadgroup float * shmem,
                           uint ntg, uint sgi, uint tii) {
    float s = simd_sum(v);
    if (tii == 0) { shmem[sgi] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
        float t = (tii < (ntg + 31) / 32) ? shmem[tii] : 0.0f;
        t = simd_sum(t);
        if (tii == 0) { shmem[0] = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shmem[0];
}

// RMSNorm over one contiguous n-vector: y = x/sqrt(mean(x^2)+eps)*w
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
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.n + a.eps);
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
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.n + a.eps);
    device const float * w = (device const float *) (weights + a.woff);
    for (uint i = tid; i < a.n; i += ntg) { yr[i] = xr[i] * scale * w[i]; }
}

// Per-row RMSNorm / L2Norm of a [d, rows] buffer, in place, d fastest.
// RMSNorm takes a weight; L2Norm has none and divides by sqrt(sum+eps) --
// note the mean, which is the difference. `xoff` lets q|k|v share a buffer.
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
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.d + a.eps);
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
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq + a.eps);
    for (uint i = tid; i < a.d; i += ntg) { r[i] *= scale; }
}

// Batched per-row L2Norm: `rowsPerTok` rows of length d per token, token n
// based at n*tokStride. One dispatch replaces the 2*N tiny per-token calls
// that made GDN prefill CPU-bound. Mirrors l2norm_rows exactly.
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
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq + a.eps);
    for (uint i = tid; i < a.d; i += ntg) { r[i] *= scale; }
}

// BF16-weight norms (gemma)
// Same reductions as their f32 twins; only the weight read differs. They are
// separate kernels rather than a type flag on NormArgs/RowArgs so the ternary
// path's kernels stay byte-identical.
kernel void rmsnorm_bf16(
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
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.n + a.eps);
    device const uchar * w = weights + a.woff;
    for (uint i = tid; i < a.n; i += ntg) {
        y[i] = x[i] * scale * bf16_at(w, i);
    }
}

kernel void rmsnorm_rows_bf16(
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
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.d + a.eps);
    device const uchar * w = weights + a.woff;
    for (uint i = tid; i < a.d; i += ntg) {
        r[i] = r[i] * scale * bf16_at(w, i);
    }
}

// The scale-free variant: gemma's v_norm is Gemma4RMSNorm(with_scale=False),
// a real op that leaves no tensor in the checkpoint, so a tensor scan cannot
// see it. Divides by the RMS and stops -- note the mean, unlike l2norm_rows.
kernel void rmsnorm_rows_noweight(
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
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.d + a.eps);
    for (uint i = tid; i < a.d; i += ntg) { r[i] *= scale; }
}

// Half-rotation RoPE with an explicit rotated-pair count (gemma). Mirrors
// GK.rope. TRAP: pairs (j, j + headDim/2), NOT rope_neox's (i, i + nRot/2).
// [rope-pairing]
struct RopeGemmaArgs {
    uint headDim; uint nHead; uint rotated; float base; uint pos;
};

// One RoPE butterfly: rotate the pair (i1, i2) by the given cos/sin.
// The PAIRING stays the CALLER's, deliberately -- see [rope-pairing].
inline void rope_pair(device float * x, uint i1, uint i2,
                      float c, float s) {
    const float p = x[i1];
    const float q = x[i2];
    x[i1] = p * c - q * s;
    x[i2] = p * s + q * c;
}

kernel void rope_gemma(
        device float           * x [[buffer(0)]],
        constant RopeGemmaArgs & a [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.nHead * a.rotated) {
        const uint hf = a.headDim / 2;
        const uint h = gid / a.rotated;
        const uint j = gid % a.rotated;
        const uint hoff = h * a.headDim;
        const float freq = pow(a.base, -2.0f * (float) j / (float) a.headDim);
        const float ang = (float) a.pos * freq;
        rope_pair(x, hoff + j, hoff + j + hf, cos(ang), sin(ang));
    }
}

// Batched BF16-weight RMSNorm to a separate output: the gemma vision tower
// runs four of these per block over every patch row.
kernel void rmsnorm_batch_bf16(
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
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.n + a.eps);
    device const uchar * w = weights + a.woff;
    for (uint i = tid; i < a.n; i += ntg) {
        yr[i] = xr[i] * scale * bf16_at(w, i);
    }
}

// gemma's vision rope is TWO-dimensional: the head's first half rotates by
// the patch's x and the second by its y. TRAP: pairs WITHIN each axis half,
// not across the head like vit_rope. [rope-pairing]
struct GVRopeArgs {
    uint rowStride; uint off; uint headDim; uint nHead; uint N;
};

kernel void gemma_vit_rope(
        device float        * x    [[buffer(0)]],
        device const float  * cosT [[buffer(1)]],
        device const float  * sinT [[buffer(2)]],
        constant GVRopeArgs & a    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hd = a.headDim;
    const uint per = hd / 2;        // channels driven by one axis
    const uint hf = per / 2;        // rotated pairs per axis
    const uint pairs = 2 * hf;
    const uint perRow = a.nHead * pairs;
    if (gid < a.N * perRow) {
        const uint p = gid / perRow;
        const uint r = gid % perRow;
        const uint h = r / pairs;
        const uint pi = r % pairs;
        const uint axis = pi / hf;
        const uint j = pi % hf;
        const uint base = p * a.rowStride + a.off + h * hd + axis * per;
        const uint t = p * hd + axis * per + j;
        rope_pair(x, base + j, base + j + hf, cosT[t], sinT[t]);
    }
}

// gemma vision attention: bidirectional over separate q/k/v, scaling 1.0.
// TRAP: padding patches must be MASKED, not merely zeroed -- a padded key
// still has a finite score, so leaving it unmasked puts real weight on it.
struct GVAttnArgs { uint n; uint hd; uint nHead; };

kernel void gemma_vit_attn(
        device const float * q     [[buffer(0)]],
        device const float * k     [[buffer(1)]],
        device const float * v     [[buffer(2)]],
        device const float * mask  [[buffer(3)]],
        device       float * out   [[buffer(4)]],
        constant GVAttnArgs & a    [[buffer(5)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint   tgid  [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint hd = a.hd;
    const uint row = tgid / a.nHead;
    const uint head = tgid % a.nHead;
    const uint stride = a.nHead * hd;
    threadgroup float * qs = shmem;
    device const float * qh = q + (ulong) row * stride + head * hd;
    for (uint i = tiisg; i < hd; i += 32) { qs[i] = qh[i]; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY, l = 0.0f;
    float acc[8];
    const ushort SL = (ushort) ((hd + 31) / 32);
    for (ushort u = 0; u < SL; u++) { acc[u] = 0.0f; }
    for (uint t0 = 0; t0 < a.n; t0 += 32) {
        const uint t = t0 + tiisg;
        float sc = -INFINITY;
        if (t < a.n && mask[t] == 0.0f) {
            device const float * kt = k + (ulong) t * stride + head * hd;
            float p = 0.0f;
            for (uint i = 0; i < hd; i++) { p += qs[i] * kt[i]; }
            sc = p;
        }
        const float mNew = max(m, simd_max(sc));
        const float corr = exp(m - mNew);
        const float w = sc > -INFINITY ? exp(sc - mNew) : 0.0f;
        l = l * corr + simd_sum(w);
        for (ushort u = 0; u < SL; u++) { acc[u] *= corr; }
        const uint tk = min(32u, a.n - t0);
        for (ushort j = 0; j < tk; j++) {
            const float wt = simd_broadcast(w, j);
            device const float * vt =
                v + (ulong) (t0 + j) * stride + head * hd;
            for (ushort u = 0; u < SL; u++) {
                const uint i = tiisg + 32 * u;
                if (i < hd) { acc[u] += wt * vt[i]; }
            }
        }
        m = mNew;
    }
    const float inv = 1.0f / l;
    device float * o = out + (ulong) row * stride + head * hd;
    for (ushort u = 0; u < SL; u++) {
        const uint i = tiisg + 32 * u;
        if (i < hd) { o[i] = acc[u] * inv; }
    }
}

// gemma elementwise glue
// x[i] *= s. Carries embed_scale, the per-layer embed scale, the 1/sqrt(dim)
// on the per-layer projection, and layer_scalar (which multiplies the WHOLE
// layer output, last).
kernel void scale_inplace(device float * x [[buffer(0)]],
                          constant uint & n [[buffer(1)]],
                          constant float & s [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid < n) { x[gid] *= s; }
}

// a[i] = (a[i] + b[i]) * s -- the per-layer input is the mean of the gathered
// table row and the projected hidden, weighted 1/sqrt(2).
kernel void add_scaled(device float * a [[buffer(0)]],
                       device const float * b [[buffer(1)]],
                       constant uint & n [[buffer(2)]],
                       constant float & s [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < n) { a[gid] = (a[gid] + b[gid]) * s; }
}

// a[i] = gelu(a[i]) * b[boff + i]. Both the MLP (boff 0) and the per-layer
// gate (boff = layer's slice of the gathered table) take this shape, and
// gemma activates with gelu where the ternary path uses silu.
struct GeluMulArgs { uint n; uint boff; };

kernel void gelu_mul(device float * a [[buffer(0)]],
                     device const float * b [[buffer(1)]],
                     constant GeluMulArgs & g [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < g.n) {
        const float v = a[gid];
        const float t =
            precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v));
        a[gid] = 0.5f * v * (1.0f + t) * b[g.boff + gid];
    }
}

// gelu_mul over N rows where A and B have DIFFERENT row strides: each token's
// [perLayerDim] gate against its own [nLayer*perLayerDim] table row at a fixed
// layer offset. One dispatch instead of ~9000 a chunk.
struct GeluMulRowsArgs {
    uint n; uint rows; uint aStride; uint bStride; uint boff;
};

kernel void gelu_mul_rows(
        device       float          * a [[buffer(0)]],
        device const float          * b [[buffer(1)]],
        constant GeluMulRowsArgs    & g [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < g.rows * g.n) {
        const uint r = gid / g.n;
        const uint i = gid % g.n;
        const uint ai = r * g.aStride + i;
        const float v = a[ai];
        const float t =
            precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v));
        a[ai] = 0.5f * v * (1.0f + t) * b[r * g.bStride + g.boff + i];
    }
}

// tanh(x/cap)*cap on the logits: nothing in gemma's reference dump exceeds
// +/-30 because the model caps them here, not because the weights are small.
kernel void softcap(device float * x [[buffer(0)]],
                    constant uint & n [[buffer(1)]],
                    constant float & cap [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) { x[gid] = precise::tanh(x[gid] / cap) * cap; }
}

// Partial NEOX RoPE on the first nRot dims of each head
// heads laid out [headDim, nHead]; one thread per rotated pair
// (Kern.ropeNeox).
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
        rope_pair(x, hoff + i, hoff + i + hf, cos(ang), sin(ang));
    }
}

// Elementwise glue
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

kernel void accum_outer(device float * h [[buffer(0)]],
                        device const float * x [[buffer(1)]],
                        constant uint & n [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x < n && gid.y < n) {
        h[(ulong) gid.y * n + gid.x] += x[gid.y] * x[gid.x];
    }
}

kernel void accum_sq(device float * dst [[buffer(0)]],
                     device const float * src [[buffer(1)]],
                     constant uint & n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) { dst[gid] += src[gid] * src[gid]; }
}

// x[i] += y[i]   (residual add)
kernel void add_inplace(device float * x [[buffer(0)]],
                        device const float * y [[buffer(1)]],
                        constant uint & n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < n) { x[gid] += y[gid]; }
}

// GDN gates: beta = sigmoid(bPre); g = softplus(aPre+dt)*aNeg
// dt (ssm_dt.bias) and aNeg (ssm_a = -exp(A_log)) are f32 tensors; one thread
// per value head (GDN.step gate loop). They are SEPARATE tensors and take a
// buffer each, since nothing puts two tensors in one weight window.
// [block-layout]
struct GateArgs { ulong dtOff; ulong aOff; uint nV; };

kernel void gdn_gate(
        device const float * bPre    [[buffer(0)]],
        device const float * aPre    [[buffer(1)]],
        device const uchar * dtW     [[buffer(2)]],
        device const uchar * aW      [[buffer(3)]],
        device       float * beta    [[buffer(4)]],
        device       float * g       [[buffer(5)]],
        constant GateArgs  & a       [[buffer(6)]],
        constant uint      & total   [[buffer(7)]],
        uint gid [[thread_position_in_grid]]) {
    // `total` = nV (single token) or N*nV (batched); dt/aNeg are per value
    // head,
    // so index them by gid % nV.
    if (gid < total) {
        device const float * dt   = (device const float *) (dtW + a.dtOff);
        device const float * aNeg = (device const float *) (aW + a.aOff);
        const uint h = gid % a.nV;
        beta[gid] = sigmoidf(bPre[gid]);
        g[gid] = softplusf(aPre[gid] + dt[h]) * aNeg[h];
    }
}

// GDN causal depthwise conv (K=4) + silu, then shift the ring
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

// GDN autoregressive delta-rule scan (one token)
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

// Deinterleave the fused q|gate attention projection
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

// kargs for attn_head. [kv-pages] [flash-attn]
struct AttnArgs {
    uint hd; uint nH; uint nKV; uint T; uint kvDim; uint P;
    float scale; uint gated; uint lo;
};

// A page table = an array of device pointers, one per P-position page, held
// as raw gpuAddresses. KV_MAXP * P bounds the context (2048 * 512 = 1M
// positions). K/V are stored HALF and accumulated f32. [kv-pages]
#define KV_MAXP 2048
struct KVTable { device const half * pages[KV_MAXP]; };

// One position's row in a paged K or V table. The same arithmetic serves
// both, and it was written twice. [kv-pages]
inline device const half * kv_row(device const KVTable & tab, uint t,
                                  uint P, uint kvDim, uint kvh, uint hd) {
    return tab.pages[t / P] + (t % P) * kvDim + kvh * hd;
}

// One query-key score: this lane's dot against one K row.
// TRAP: vectorize only when hd % 4 == 0 -- that is also what makes the
// kvh * hd row base 8B-aligned. The scalar tail covers the rest.
inline float attn_score(threadgroup const float * qs,
                        device const half * kt, uint hd, float scale) {
    const uint hd4 = (hd % 4 == 0) ? hd : 0;
    threadgroup const float4 * q4 = (threadgroup const float4 *) qs;
    device const half4 * k4 = (device const half4 *) kt;
    float4 p4 = 0.0f;
    for (uint i = 0; i < hd4 / 4; i++) {
        p4 += q4[i] * float4(k4[i]);
    }
    float p = p4.x + p4.y + p4.z + p4.w;
    for (uint i = hd4; i < hd; i++) {
        p += qs[i] * (float) kt[i];
    }
    return p * scale;
}

// Combine the NSG per-simdgroup partials into the output row, then gate and
// store. An empty stripe has m = -INF, so exp(m - M) = 0 and adds nothing.
inline void attn_combine(threadgroup const float * redM,
                         threadgroup const float * redL,
                         threadgroup const float * redAcc,
                         device const float * gate,
                         device       float * out,
                         uint hd, uint gated, ushort NSG, ushort tiisg) {
    const ushort SL = (ushort) ((hd + 31) / 32);   // dim slices per lane (<=8)
    float M = -INFINITY;
    for (ushort g = 0; g < NSG; g++) {
        M = max(M, redM[g]);
    }
    float L = 0.0f;
    for (ushort g = 0; g < NSG; g++) {
        L += redL[g] * exp(redM[g] - M);
    }
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
            // sigmoid(gate). `gated` selects (gate unread when 0).
            out[j] = gated ? o * sigmoidf(gate[j]) : o;
        }
    }
}

// The flash / online-softmax body, shared by the one-token and the batched
// attention kernels. q, gate and out arrive ALREADY OFFSET to this
// (row, head). [flash-attn]
inline void attn_flash(
        device const float   * qh,
        device const KVTable & kT,
        device const KVTable & vT,
        device const float   * gate,
        device       float   * out,
        uint hd, uint kvh, uint kvDim, uint P, float scale, uint gated,
        uint lo, uint T,
        threadgroup float    * shmem,
        ushort sgitg,
        ushort tiisg) {
    const ushort NSG = 4, TK = 32;
    // shmem = qs[hd] | redM[NSG] | redL[NSG] | redAcc[NSG*hd]
    // (O(hd), not O(T))
    threadgroup float * qs     = shmem;
    threadgroup float * redM   = qs + hd;
    threadgroup float * redL   = redM + NSG;
    threadgroup float * redAcc = redL + NSG;
    for (uint i = sgitg * 32 + tiisg; i < hd; i += NSG * 32) {
        qs[i] = qh[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // This simdgroup's running online-softmax state over its stripe of tiles.
    // It stays inline: every step mutates m, l and acc4 together, so lifting
    // it out would trade six lines for three by-reference accumulators.
    float m = -INFINITY, l = 0.0f;
    // The V accumulator is sliced FOUR-WIDE: lane L owns dims 4L..4L+3 of each
    // 128-dim slice, so one half4 load per lane per key covers what four
    // scalar loads used to. [flash-attn]
    const ushort VS = (ushort) ((hd + 127) / 128);
    float4 acc4[4];
    for (ushort u = 0; u < VS; u++) {
        acc4[u] = 0.0f;
    }
    for (uint t0 = lo + sgitg * TK; t0 < T; t0 += NSG * TK) {
        const uint tk = min((uint) TK, T - t0);
        // lane tiisg owns key t0+tiisg
        float sc = -INFINITY;
        if (tiisg < tk) {
            device const half * kt =
                kv_row(kT, t0 + tiisg, P, kvDim, kvh, hd);
            sc = attn_score(qs, kt, hd, scale);
        }
        const float mNew = max(m, simd_max(sc));
        const float corr = exp(m - mNew);
        const float w = sc > -INFINITY ? exp(sc - mNew) : 0.0f;
        l = l * corr + simd_sum(w);
        for (ushort u = 0; u < VS; u++) {
            acc4[u] *= corr;
        }
        for (ushort t = 0; t < tk; t++) {
            const float wt = simd_broadcast(w, t);
            device const half4 * v4 = (device const half4 *)
                kv_row(vT, t0 + t, P, kvDim, kvh, hd);
            for (ushort u = 0; u < VS; u++) {
                const uint j4 = tiisg + 32 * u;
                if (4 * j4 < hd) {
                    acc4[u] += wt * float4(v4[j4]);
                }
            }
        }
        m = mNew;
    }
    if (tiisg == 0) {
        redM[sgitg] = m;
        redL[sgitg] = l;
    }
    // Scatter the four-wide slices back per dim: redAcc stays a plain [NSG,hd]
    // float array, so attn_combine sees a flat layout.
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
        attn_combine(redM, redL, redAcc, gate, out, hd, gated, NSG, tiisg);
    }
}

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
    const uint group = a.nH / a.nKV;
    const uint off = h * a.hd;
    attn_flash(q + off, kT, vT, gate + off, out + off,
               a.hd, h / group, a.kvDim, a.P, a.scale, a.gated, a.lo, a.T,
               shmem, sgitg, tiisg);
}

// Append this token's K,V rows at position pos. The f32 -> half narrowing
// happens HERE, once per position, so every later read moves half the bytes.
// [kv-pages]
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

// BATCHED (prefill) kernels: N tokens processed per dispatch.
// Activations are token-major [N, dim] (token n at n*dim). The GEMM
// projections stream weights ONCE for the whole batch; these batched
// cheap/recurrent ops mirror their per-token counterparts exactly (validated
// against the SIMD engine), looping N internally for the recurrences (conv,
// scan) and gridding N for the parallel ones.

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
        const uint j = k % 128;
        const uchar code = (bp[2 + j / 4] >> ((j & 3) * 2)) & 3;
        out[gid] = (float) ((int) code - 1) * d;
    }
}

kernel void q2_x_embed_batch(
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
        const uint j = k % 128;
        const uchar code = (bp[2 + j / 4] >> ((j & 3) * 2)) & 3;
        out[gid] = (float) (2 * (int) code - 3) * d;
    }
}

// gdn_conv_batch: N tokens sequentially per channel; the ring lives in
// registers
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
        for (uint j = 0; j + 1 < kc; j++) {
            ring[j] = convState[j * a.convDim + c];
        }
        for (uint n = 0; n < a.N; n++) {
            const float cur = qkvMixN[n * a.convDim + c];
            float acc = 0.0f;
            for (uint j = 0; j + 1 < kc; j++) { acc += ring[j] * cw[j]; }
            acc += cur * cw[kc - 1];
            outN[n * a.convDim + c] = siluf(acc);
            for (uint j = 0; j + 2 < kc; j++) { ring[j] = ring[j + 1]; }
            if (kc >= 2) { ring[kc - 2] = cur; }
        }
        for (uint j = 0; j + 1 < kc; j++) {
            convState[j * a.convDim + c] = ring[j];
        }
    }
}

// gdn_scan_batch: N tokens sequentially per (value head hv, column j).
// q|k|v are read from convOutN (q/k already l2-normed in place per token);
// o written per token. Mirrors gdn_scan looped over N.
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
            device const float * row = convOutN + n * a.convDim;
            const uint kOff = hk * dS, vOff = hv * dS;
            device const float * q = row + kOff;
            device const float * k = row + a.keyDim + kOff;
            device const float * v = row + 2 * a.keyDim + vOff;
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
// in registers across the whole N-token sequence. Requires dS a multiple of
// 32 and <= 256 (KS <= 8); the host routes any other dS to gdn_scan_batch.
// [gdn-scan2]
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
    const ushort COLS = 4; // columns owned per threadgroup
    const ushort KS = (ushort) (dS / 32); // key-dim slices per lane (<=8)
    const uint hv = tgpig.y;
    const uint j  = tgpig.x * COLS + sgitg;
    const uint hk = hv % a.nK;
    device float * Sh = S + hv * dS * dS; // S[i*dS + j]
    float ls[8]; // lane owns i = tiisg + 32*m, m<KS
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
        for (ushort m = 0; m < KS; m++) {
            ls[m] *= gamma;
            ksum += ls[m] * kk[m];
        }
        const float sk = simd_sum(ksum);
        const float d = b * (v[j] - sk);
        float osum = 0.0f;
        for (ushort m = 0; m < KS; m++) {
            ls[m] += kk[m] * d;
            osum += ls[m] * qq[m];
        }
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
        rope_pair(x, hoff + i, hoff + i + hf, cos(ang), sin(ang));
    }
}

// gemma's rope for N tokens at basePos..basePos+N-1. TRAP: same pairing as
// rope_gemma, NOT rope_batch's. [rope-pairing]
struct RopeGemmaBatchArgs {
    uint headDim; uint nHead; uint rotated; float base; uint basePos; uint N;
};

kernel void rope_gemma_batch(
        device float                 * x [[buffer(0)]],
        constant RopeGemmaBatchArgs  & a [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const uint per = a.nHead * a.rotated;
    if (gid < a.N * per) {
        const uint n = gid / per;
        const uint r = gid % per;
        const uint hf = a.headDim / 2;
        const uint h = r / a.rotated;
        const uint j = r % a.rotated;
        const uint hoff = n * a.nHead * a.headDim + h * a.headDim;
        const float freq = pow(a.base, -2.0f * (float) j / (float) a.headDim);
        const float ang = (float) (a.basePos + n) * freq;
        rope_pair(x, hoff + j, hoff + j + hf, cos(ang), sin(ang));
    }
}

// Interleaved M-RoPE for N tokens with per-token 3D positions (t,h,w). Text
// has t==h==w and degenerates to rope_batch. Frequency i takes component
// i % 3, reproducing HF's apply_interleaved_mrope. [rope-pairing]
struct RopeMBatchArgs {
    uint headDim; uint nHead; uint nRot; float base; uint N;
};

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
        rope_pair(x, hoff + i, hoff + i + hf, cos(ang), sin(ang));
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

// Causal attention for N query tokens over the paged KV: token n (absolute
// position basePos+n) attends to keys 0..basePos+n, one threadgroup per
// (token, head). [flash-attn]
//
// The `blk` buffer carries the vision block each chunk ROW belongs to, two
// absolute uints (lo, hi) per row and 0/0 for none, so one chunk may hold
// several blocks. A query inside one reads the WHOLE block, which is the only
// place attention passes its own position. It is read up to FA_Q rows past N
// (attn_batch_mm masks a partial tail tile rather than branching around it),
// so the host sizes it past the batch. [vision-block]
struct AttnBatchArgs {
    uint hd; uint nH; uint nKV; uint kvDim; uint P; float scale;
    uint basePos; uint N; uint gated; uint window;
};

// The last key a query at absolute position q may read. Causal length T
// normally; a query inside a vision block reads to the end of that block
// instead, and the union stays CONTIGUOUS because the block contains q.
inline uint attn_end(uint q, uint T, uint blockLo, uint blockHi) {
    return (q >= blockLo && q < blockHi) ? max(T, blockHi) : T;
}

kernel void attn_batch(
        device const float   * qN    [[buffer(0)]],
        device const KVTable & kT    [[buffer(1)]],
        device const KVTable & vT    [[buffer(2)]],
        device const float   * gateN [[buffer(3)]],
        device       float   * outN  [[buffer(4)]],
        constant AttnBatchArgs & a   [[buffer(5)]],
        device const uint    * blk   [[buffer(6)]],
        threadgroup float    * shmem [[threadgroup(0)]],
        uint   tgid  [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint n = tgid / a.nH;           // query token in the batch
    const uint h = tgid % a.nH;
    const uint T = a.basePos + n + 1;     // causal length for this token
    // Sliding layers read a WINDOW rather than the whole history, and each
    // query row in the batch has its own start. window == 0 is unwindowed.
    const uint lo = a.window == 0 || T <= a.window ? 0 : T - a.window;
    // attn_flash reads [lo, hi) with no further mask -- the RANGE is the mask
    // -- so a vision block needs nothing but a longer bound. [vision-block]
    const uint hi = attn_end(a.basePos + n, T, blk[2 * n], blk[2 * n + 1]);
    const uint group = a.nH / a.nKV;
    const uint off = n * a.nH * a.hd + h * a.hd;
    attn_flash(qN + off, kT, vT, gateN + off, outN + off,
               a.hd, h / group, a.kvDim, a.P, a.scale, a.gated, lo, hi,
               shmem, sgitg, tiisg);
}

// The SAME causal attention on the 8x8 matrix units. [flash-attn-mm]
//
// FA_Q query rows and FA_K keys per iteration; the four simdgroups split the
// KEY range for the scores and the HEAD DIM for the context, so neither
// product is computed twice. Everything that is not a matmul -- the paged row
// addressing, the online-softmax algebra, the causal and window masks -- is
// the same arithmetic the scalar kernel performs, in the same order per query.
#define FA_Q 8
#define FA_K 32

// One 8x8 block of scores: S[q][k] = sum_d Q[q][d] * K[k][d], accumulated over
// the whole head dim. K is loaded TRANSPOSED, which is what turns a row-major
// [key][dim] tile into the [dim][key] operand the product wants.
inline simdgroup_float8x8 fa_scores(threadgroup const half * qs, uint hd,
                                    device const half * kBase, uint kvDim) {
    simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint d = 0; d < hd; d += 8) {
        simdgroup_half8x8 qa, kb;
        simdgroup_load(qa, qs + d, hd, ulong2(0, 0), false);
        simdgroup_load(kb, kBase + d, kvDim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s, qa, kb, s);
    }
    return s;
}

// Rescale the running context by the online-softmax correction. A simdgroup
// matrix carries no scalar multiply, so the correction rides a DIAGONAL matrix
// -- one extra product per dim tile, against the sixteen the context itself
// costs.
inline void fa_rescale(thread simdgroup_float8x8 * o, ushort tiles,
                       threadgroup const half * diag) {
    simdgroup_half8x8 d;
    simdgroup_load(d, diag, 8, ulong2(0, 0), false);
    for (ushort i = 0; i < tiles; i++) {
        simdgroup_float8x8 t = make_filled_simdgroup_matrix<float, 8>(0.f);
        simdgroup_multiply_accumulate(t, d, o[i], t);
        o[i] = t;
    }
}

// Stage the tile's queries as half, zero-filling a short tail so the product
// reads a whole 8-row block whatever the batch ends on.
inline void fa_stage_q(threadgroup half * qs, device const float * qN,
                       uint base, uint hd, uint row, ushort nq, ushort tiitg) {
    for (uint i = tiitg; i < FA_Q * hd; i += 128) {
        const uint q = i / hd;
        qs[i] = q < nq ? (half) qN[base + q * row + i % hd] : 0.0h;
    }
}

// Scale one tile of scores and MASK it per ROW: a key can be past one row's
// causal end and below another's window start, so the bound is not the tile's
// -- and with a block per row, two rows of one tile can belong to DIFFERENT
// vision blocks. Rows past nq are masked wholesale, but their `blk` pair is
// still read, which is why the host sizes that buffer FA_Q rows past N.
inline void fa_mask(threadgroup float * s, uint t0, uint T0, uint window,
                    device const uint * blk, uint n0,
                    float scale, ushort nq, ushort tiitg) {
    for (uint i = tiitg; i < FA_Q * FA_K; i += 128) {
        const uint q = i / FA_K;
        const uint t = t0 + i % FA_K;
        const uint Tq = T0 + q;
        const uint loQ = (window == 0 || Tq <= window) ? 0 : Tq - window;
        const uint r = 2 * (n0 + q);
        const uint hiQ = attn_end(Tq - 1, Tq, blk[r], blk[r + 1]);
        s[i] = (q < nq && t < hiQ && t >= loQ) ? s[i] * scale : -INFINITY;
    }
}

// Fold one masked tile into the running softmax: P out as half for the next
// product, the max and denominator advanced, the correction left on the
// diagonal for fa_rescale.
inline void fa_softmax(threadgroup float * s, threadgroup half * p,
                       threadgroup float * mRow, threadgroup float * lRow,
                       threadgroup half * cDiag, ushort tiitg) {
    for (uint q = tiitg; q < FA_Q; q += 128) {
        float mx = mRow[q];
        for (ushort k = 0; k < FA_K; k++) {
            mx = max(mx, s[q * FA_K + k]);
        }
        // A wholly masked tile leaves the row untouched; -INF minus -INF is a
        // NaN that would poison the context for the rest of the sweep.
        const float corr = mx > -INFINITY ? exp(mRow[q] - mx) : 1.0f;
        float sum = 0.0f;
        for (ushort k = 0; k < FA_K; k++) {
            const float e =
                s[q * FA_K + k] > -INFINITY ? exp(s[q * FA_K + k] - mx) : 0.0f;
            p[q * FA_K + k] = (half) e;
            sum += e;
        }
        mRow[q] = mx;
        lRow[q] = lRow[q] * corr + sum;
        cDiag[q * 8 + q] = (half) corr;
    }
}

// O[q][d] += sum_k P[q][k] * V[k][d] over one tile, this simdgroup's dim slice.
inline void fa_context(thread simdgroup_float8x8 * o, ushort oTiles,
                       threadgroup const half * p,
                       device const KVTable & vT, uint t0, uint tEnd,
                       uint P, uint kvDim, uint kvh, uint hd, uint slice) {
    for (ushort kb = 0; kb < FA_K / 8; kb++) {
        const uint t = t0 + kb * 8;
        if (t < tEnd) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, p + kb * 8, FA_K, ulong2(0, 0), false);
            device const half * vb = kv_row(vT, t, P, kvDim, kvh, hd) + slice;
            for (ushort i = 0; i < oTiles; i++) {
                simdgroup_half8x8 vv;
                simdgroup_load(vv, vb + i * 8, kvDim, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(o[i], pm, vv, o[i]);
            }
        }
    }
}

// Normalize this simdgroup's slice and write it out. The accumulators go back
// through threadgroup memory because only there is the [row][dim] layout the
// output wants; each simdgroup lands in its OWN 64-float window, since all
// four store at the same moment.
inline void fa_store(thread simdgroup_float8x8 * o, ushort oTiles,
                     threadgroup float * mine, threadgroup const float * lRow,
                     device float * outN, device const float * gateN,
                     uint base, uint row, uint hd, uint slice, uint gated,
                     ushort nq, ushort tiisg) {
    for (ushort i = 0; i < oTiles; i++) {
        simdgroup_store(o[i], mine, 8, ulong2(0, 0), false);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort j = tiisg; j < FA_Q * 8; j += 32) {
            const uint q = j / 8;
            if (q < nq) {
                const uint at = base + q * row + slice + i * 8 + j % 8;
                const float v = mine[j] / lRow[q];
                outN[at] = gated ? v * sigmoidf(gateN[at]) : v;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void attn_batch_mm(
        device const float   * qN    [[buffer(0)]],
        device const KVTable & kT    [[buffer(1)]],
        device const KVTable & vT    [[buffer(2)]],
        device const float   * gateN [[buffer(3)]],
        device       float   * outN  [[buffer(4)]],
        constant AttnBatchArgs & a   [[buffer(5)]],
        device const uint    * blk   [[buffer(6)]],
        threadgroup uchar    * shmem [[threadgroup(0)]],
        uint   tgid  [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint hd = a.hd;
    const uint slice = sgitg * (hd / 4);       // this simdgroup's dim slice
    const ushort oTiles = (ushort) (hd / 4 / 8);
    const uint n0 = (tgid / a.nH) * FA_Q;
    const uint h = tgid % a.nH;
    const ushort nq = (ushort) min((uint) FA_Q, a.N - n0);
    const uint kvh = h / (a.nH / a.nKV);
    const uint row = a.nH * hd;                // one token's q / out stride
    const uint base = n0 * row + h * hd;
    // shmem = qs[FA_Q*hd half] | s[FA_Q*FA_K float] | p[FA_Q*FA_K half]
    //       | m[FA_Q] | l[FA_Q] | corr[64 half, a diagonal]
    threadgroup half  * qs    = (threadgroup half *) shmem;
    threadgroup float * sBuf  = (threadgroup float *) (qs + FA_Q * hd);
    threadgroup half  * pBuf  = (threadgroup half *) (sBuf + FA_Q * FA_K);
    threadgroup float * mRow  = (threadgroup float *) (pBuf + FA_Q * FA_K);
    threadgroup float * lRow  = mRow + FA_Q;
    threadgroup half  * cDiag = (threadgroup half *) (lRow + FA_Q);
    fa_stage_q(qs, qN, base, hd, row, nq, tiitg);
    for (uint i = tiitg; i < FA_Q; i += 128) {
        mRow[i] = -INFINITY;
        lRow[i] = 0.0f;
    }
    // The correction matrix is DIAGONAL, so its off-diagonal must be zero
    // before the first tile reads it -- and stay zero, which is why each pass
    // clears only the cell it wrote.
    for (uint i = tiitg; i < FA_Q * 8; i += 128) { cDiag[i] = 0.0h; }
    simdgroup_float8x8 o[16];
    for (ushort i = 0; i < oTiles; i++) {
        o[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    // The union of the LIVE rows' ranges, which is what the key sweep below
    // has to cover; fa_mask still bounds each row on its own, so a key past
    // one row's end contributes nothing to it. Rows past nq are excluded
    // deliberately -- their `blk` pair is padding, and folding it in would
    // sweep keys no live row can reach. [vision-block]
    const uint tLast = a.basePos + n0 + nq;
    uint tEnd = tLast;
    for (ushort q = 0; q < nq; q++) {
        const uint r = 2 * (n0 + q);
        const uint qp = a.basePos + n0 + q;
        tEnd = max(tEnd, attn_end(qp, qp + 1, blk[r], blk[r + 1]));
    }
    const uint T0 = a.basePos + n0 + 1;
    const uint loF = (a.window == 0 || T0 <= a.window) ? 0 : T0 - a.window;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t0 = loF / FA_K * FA_K; t0 < tEnd; t0 += FA_K) {
        // Scores: simdgroup s owns keys t0 + 8s .. +8.
        const uint kt = t0 + sgitg * 8;
        const simdgroup_float8x8 sBlk = kt < tEnd
            ? fa_scores(qs, hd, kv_row(kT, kt, a.P, a.kvDim, kvh, hd), a.kvDim)
            : make_filled_simdgroup_matrix<float, 8>(0.f);
        simdgroup_store(sBlk, sBuf + sgitg * 8, FA_K, ulong2(0, 0), false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fa_mask(sBuf, t0, T0, a.window, blk, n0, a.scale, nq, tiitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fa_softmax(sBuf, pBuf, mRow, lRow, cDiag, tiitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fa_rescale(o, oTiles, cDiag);
        fa_context(o, oTiles, pBuf, vT, t0, tEnd, a.P, a.kvDim, kvh, hd,
                   slice);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tiitg; i < FA_Q * 8; i += 128) { cDiag[i] = 0.0h; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    fa_store(o, oTiles, sBuf + sgitg * 64, lRow, outN, gateN, base, row, hd,
             slice, a.gated, nq, tiisg);
}

// ViT (Qwen3-VL vision tower) kernels. Each mirrors the CPU ViT
// (SIMD/ViT.swift) op for op; that engine is the oracle. [vit-tower]

// f16-weight simdgroup GEMM: dst[N,M] = X[N,K] @ W[M,K]^T, W a plain
// row-major [M][K] half buffer. Unlike the quantized GEMMs it needs a K-tail
// guard: ViT K values are NOT all multiples of 32. [vit-tower]
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
            const short iy = 8 * sx;
            threadgroup half * bp = sb + 64 * (4 * sx + sy) + 8 * ly;
            for (short i = 0; i < 8; i++) {
                const int k = loop_k + iy + i;
                bp[i] = k < K ? (half) yrow[k] : (half) 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simd_mm_slice(sa, sb, mc, sgitg);
    }
    store_mm_tile(mc, dst, shmem, r0, r1, M, (int) N, nr0, nr1,
                  tiitg, sgitg);
}

// LayerNorm with bias: the ViT is pre-LN, mean-centered with a learned bias,
// unlike the LM's rmsnorm. x -> y so x survives for the residual. Needs TWO
// reductions, hence the disjoint shmem halves. [tg-reduce]
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
    // Disjoint scratch halves, because tg_reduce_sum clobbers what it is
    // handed: the second reduction would otherwise overwrite the first's
    // result before it is read. Two calls cost two extra barriers against the
    // hand-fused form; the arithmetic is untouched. [tg-reduce]
    const float sum = tg_reduce_sum(s1, shmem, ntg, sgi, tii);
    const float sumsq = tg_reduce_sum(s2, shmem + 32, ntg, sgi, tii);
    const float mean = sum / (float) a.n;
    const float inv =
        1.0f / sqrt(sumsq / (float) a.n - mean * mean + a.eps);
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

// Vision M-RoPE over the q or k span of the fused qkv rows; cos/sin come
// precomputed per (slot, pair) from ViT.ropeTables. `off` selects q (0) or
// k (e) inside a 3e row. [rope-pairing]
struct VRopeArgs {
    uint rowStride; uint off; uint headDim; uint nHead; uint N;
};

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
        rope_pair(x, base + j, base + j + hf, cosT[s * hf + j],
                  sinT[s * hf + j]);
    }
}

// Bidirectional attention over the fused qkv rows ([N][3e]), no causal mask
// and no GQA. The 4 x 32 lane slice caps hd at 128; the host asserts it.
// [vit-attn]
struct VAttnArgs { uint n; uint e; uint hd; uint nHead; float scale; };

// Stage one TK-key tile of K and V into threadgroup memory. Reads the fused
// qkv rows, writes the two tiles, touches nothing else -- so it lifts out of
// the softmax cleanly. [vit-attn]
inline void vit_stage_kv(device const float * qkv,
                         threadgroup float * kt, threadgroup float * vt,
                         uint t0, uint tk, uint h, uint row,
                         uint e, uint hd, uint tid, uint ntg) {
    for (uint i = tid; i < tk * hd; i += ntg) {
        const uint t = i / hd, j = i % hd;
        kt[i] = qkv[(ulong) (t0 + t) * row + e + h * hd + j];
        vt[i] = qkv[(ulong) (t0 + t) * row + 2 * e + h * hd + j];
    }
}

// One query-key dot, both operands already staged in threadgroup memory.
// TRAP: vectorize only on the hd % 4 == 0 prefix -- that is what keeps the
// float4 loads 16B-aligned. The scalar tail covers any other geometry.
inline float vit_score(threadgroup const float * qrow,
                       threadgroup const float * krow, uint hd) {
    const uint hd4 = hd & ~3u;
    threadgroup const float4 * q4 = (threadgroup const float4 *) qrow;
    threadgroup const float4 * k4 = (threadgroup const float4 *) krow;
    float4 p4 = 0.0f;
    for (uint i = 0; i < hd4 / 4; i++) {
        p4 += q4[i] * k4[i];
    }
    float p = p4.x + p4.y + p4.z + p4.w;
    for (uint i = hd4; i < hd; i++) {
        p += qrow[i] * krow[i];
    }
    return p;
}

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
        for (uint j = tiisg; j < a.hd; j += 32) {
            qs[sgitg * a.hd + j] = qh[j];
        }
    }
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = 0; t0 < a.n; t0 += TK) {
        const uint tk = min(TK, a.n - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        vit_stage_kv(qkv, kt, vt, t0, tk, h, row, a.e, a.hd, tid, ntg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sc = -INFINITY;
        if (s < a.n && tiisg < tk) {
            // hd rows start 16B-aligned whenever hd % 4 == 0 (every Qwen
            // tower; the scalar tail covers any other geometry).
            const float p = vit_score(qs + sgitg * a.hd,
                                     kt + tiisg * a.hd, a.hd);
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

// gemma audio: elementwise pieces
kernel void silu_inplace(device float * x [[buffer(0)]],
                         constant uint & n [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < n) { x[gid] = siluf(x[gid]); }
}

// x[i] += y[i] * s -- the macaron halves add at HALF weight, not full.
struct FmaArgs { uint n; float s; };

kernel void fma_inplace(device float * x [[buffer(0)]],
                        device const float * y [[buffer(1)]],
                        constant FmaArgs & a [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < a.n) { x[gid] += y[gid] * a.s; }
}

// GLU over a 2e-wide row: out[c] = lo[c] * sigmoid(hi[c]).
struct GluArgs { uint n; uint e; };

kernel void glu_rows(device const float * src [[buffer(0)]],
                     device       float * dst [[buffer(1)]],
                     constant GluArgs   & a   [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < a.n * a.e) {
        const uint t = gid / a.e;
        const uint c = gid % a.e;
        const uint row = t * 2 * a.e;
        dst[gid] = src[row + c] * sigmoidf(src[row + a.e + c]);
    }
}

// Causal depthwise conv: each channel convolves only its own history.
struct DwArgs { uint n; uint e; uint k; };

kernel void depthwise_causal(device const float * x   [[buffer(0)]],
                             device const float * w   [[buffer(1)]],
                             device       float * out [[buffer(2)]],
                             constant DwArgs    & a   [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid < a.n * a.e) {
        const uint t = gid / a.e;
        const uint c = gid % a.e;
        float acc = 0.0f;
        for (uint j = 0; j < a.k; j++) {
            const int src = (int) t - (int) (a.k - 1) + (int) j;
            if (src >= 0) { acc += x[(uint) src * a.e + c] * w[c * a.k + j]; }
        }
        out[gid] = acc;
    }
}

// The audio query is scaled per head_dim by a learned, softplus'd factor.
struct HeadScaleArgs { uint n; uint e; uint hd; };

kernel void scale_head_dims(device       float * x [[buffer(0)]],
                            device const float * s [[buffer(1)]],
                            constant HeadScaleArgs & a [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid < a.n * a.e) { x[gid] *= s[(gid % a.e) % a.hd]; }
}

// gemma audio attention: Transformer-XL relative attention over blocked
// local windows. TRAP: the window holds `chunk` positions INCLUDING self, so
// `rel` runs 1..past and allowing 0 hands every query one extra key.
// [audio-attn]
struct AAttnArgs {
    uint n; uint e; uint heads; uint hd; uint chunk; uint ctx; uint past;
    float cap;
};

// The absolute key position a window slot maps to. Both loops below need it,
// and it was computed in each -- the mask rule written twice. [audio-attn]
inline int audio_key_at(uint b, uint o, uint chunk, uint past) {
    return (int) (b * chunk + o) - (int) past;
}

// Whether a window slot addresses a real position at all.
inline bool audio_key_live(int kAbs, uint n) {
    return kAbs >= 0 && kAbs < (int) n;
}

// One query's window scores, and their max for the softmax shift.
// TRAP: `rel` runs 1..past, NOT 0..past -- the window holds `chunk` positions
// INCLUDING self, so allowing 0 hands every query one extra key. [audio-attn]
inline float audio_scores(device const float * q, device const float * k,
                          device const float * relk,
                          thread float (&sc)[32],
                          uint qo, uint qi, uint b, uint h,
                          constant AAttnArgs & a) {
    float mx = -INFINITY;
    for (uint o = 0; o < a.ctx; o++) {
        const int kAbs = audio_key_at(b, o, a.chunk, a.past);
        const int rel = (int) o - (int) qi;
        const bool ok = audio_key_live(kAbs, a.n)
            && rel >= 1 && rel <= (int) a.past;
        float s = -INFINITY;
        if (ok) {
            const uint ko = (uint) kAbs * a.e + h * a.hd;
            const uint ro = (uint) rel * a.e + h * a.hd;
            float ac = 0.0f, bd = 0.0f;
            for (uint c = 0; c < a.hd; c++) {
                ac += q[qo + c] * k[ko + c];
                bd += q[qo + c] * relk[ro + c];
            }
            s = precise::tanh((ac + bd) / a.cap) * a.cap;
        }
        sc[o] = s;
        mx = max(mx, s);
    }
    return mx;
}

// The weighted sum of V over the live slots, into this query's output row.
inline void audio_context(device const float * v, device float * out,
                          thread const float (&sc)[32], float inv,
                          uint qo, uint b, uint h,
                          constant AAttnArgs & a) {
    for (uint c = 0; c < a.hd; c++) {
        out[qo + c] = 0.0f;
    }
    for (uint o = 0; o < a.ctx; o++) {
        const int kAbs = audio_key_at(b, o, a.chunk, a.past);
        if (sc[o] > 0.0f && audio_key_live(kAbs, a.n)) {
            const uint vo = (uint) kAbs * a.e + h * a.hd;
            const float wgt = sc[o] * inv;
            for (uint c = 0; c < a.hd; c++) {
                out[qo + c] += wgt * v[vo + c];
            }
        }
    }
}

kernel void gemma_audio_attn(
        device const float * q    [[buffer(0)]],
        device const float * k    [[buffer(1)]],
        device const float * v    [[buffer(2)]],
        device const float * relk [[buffer(3)]],
        device       float * out  [[buffer(4)]],
        constant AAttnArgs  & a   [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.n * a.heads) {
        const uint t = gid / a.heads;
        const uint h = gid % a.heads;
        const uint b = t / a.chunk;          // which blocked window
        const uint qi = t % a.chunk;         // this query's slot in it
        const uint qo = t * a.e + h * a.hd;
        float sc[32];
        const float mx = audio_scores(q, k, relk, sc, qo, qi, b, h, a);
        float sum = 0.0f;
        for (uint o = 0; o < a.ctx; o++) {
            sc[o] = sc[o] > -INFINITY ? exp(sc[o] - mx) : 0.0f;
            sum += sc[o];
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        audio_context(v, out, sc, inv, qo, b, h, a);
    }
}

// static range quantization: part of the ARITHMETIC the QAT trained with,
// not a compression step. `rint` rounds half to EVEN to match torch.round;
// anything else drifts on the exact .5 the trained scales hit. Two entry
// points because an input buffer is usually shared and an output is not.
// [srq-clamp]
struct SrqArgs { uint n; float s; };

kernel void srq_inplace(device float * x [[buffer(0)]],
                        constant SrqArgs & a [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < a.n) {
        x[gid] = clamp(rint(x[gid] / a.s), -128.0f, 127.0f) * a.s;
    }
}

kernel void srq_to(device const float * src [[buffer(0)]],
                   device       float * dst [[buffer(1)]],
                   constant SrqArgs   & a   [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < a.n) {
        dst[gid] = clamp(rint(src[gid] / a.s), -128.0f, 127.0f) * a.s;
    }
}

/* FOOTNOTES

Rationale and measurements live here rather than beside the code, so a
kernel body reads as its op sequence and one explanation can serve several
call sites instead of being copied to each. Referenced by a [tag] at the
site. Tags are NAMES and are never renumbered, so inserting one below
disturbs nothing and a stale reference is findable with grep.

A TRAP stays inline. Anything that is silently wrong if you change it --
a pairing rule, a stride, a type width -- keeps a one-line warning at the
code, because someone editing that line must not have to follow a link to
learn it. Only the reasoning and the numbers move here.

### [block-layout]  how a quantized weight is addressed

Weights are read straight out of the mapped GGUF, so every kernel that touches
one takes a 64-bit byte offset rather than a pointer. 64-bit because a window
can exceed 4 GB.

The mapping is covered by SEVERAL no-copy buffers, not one, because
maxBufferLength is a hard per-device ceiling (2048 MB on an A14) that a 2.5 GB
set cannot fit however much memory is free. A kernel is handed the one buffer
holding its tensor and an offset INSIDE that buffer; it never sees a
file-relative offset. Windows are cut at tensor boundaries, so no tensor
straddles two of them and a row offset can be added to a tensor's base without
leaving its buffer. gdn_gate is the only kernel reading two tensors at once
and therefore the only one taking two weight buffers.

The three block formats, byte for byte as the repackers write them:

    Q2_0  128 weights / 34 bytes  { half d; uchar qs[32] }  w = (code-1)*d
          codes are 2-bit, 4 per byte, LSB-first
    Q4_0   32 weights / 18 bytes  { half d; uchar qs[16] }  w = (q-8)*d
          byte j holds element j in the LOW nibble and j+16 in the HIGH one
    Q8_0   32 weights / 34 bytes  { half d; char  qs[32] }  w = q*d

Q2_0 is a PrismML fork type, not upstream ggml, so there is no llama.cpp
cross-check for it -- which is what makes the SIMD oracle load-bearing.

The GEMV path addresses blocks by manual byte arithmetic (stride, d at +0, qs
at +2) rather than through a struct, to avoid depending on any MSL packing
assumption. The GEMM path does use structs, which is safe because it only ever
reads whole sub-blocks.

Activation and state buffers are shared-storage f32 (unified memory), and all
dispatches for one token ride ONE command encoder -- so Metal's hazard
tracking serializes the dependent steps and scratch is safely reused across
layers.

### [gemv-unroll]  why `q2_0_gemv` looks hand-rolled

Decode is LATENCY-bound here, not bandwidth-bound: measured 8.1 t/s against
a 12.4 t/s memory wall, and only 12% of compute peak. So each lane keeps
TWO blocks in flight per iteration, in DISTINCT scalar arrays -- a
loop-indexed yl[u][i] spills to local memory and regresses. UN=2 is the
sweet spot; UN=4 blows the register budget to 5.7 t/s. Net +8%, 8.1 -> 8.7.

The scalar lo/hi select-add beats every alternative tried: float4-dot,
scalar-fma and NSG=4 all regressed. TPB=8 lanes cooperate on one 128-weight
block so their 32 qs bytes are read 4 consecutive bytes each, which is what
makes the weight stream coalesce. Per-block dot is d*(lo + 2*hi - sumy) =
d*sum((code-1)*x), matching `Q2_0.matvec`.

This is the most tuned kernel in the file and the one least worth
"simplifying". It is also why gemv did not get folded into a shared
template: the shape that makes it fast is not the shape q4_0/q8_0 have.

### [kv-pages]  the bindless page table, and why K/V are half

MSL forbids a top-level buffer whose pointee is a pointer, but ALLOWS a
device pointer as a struct member (a tier-2 argument buffer). On Apple
Silicon each `device half*` slot is just its 8-byte gpuAddress, so the host
writes raw addresses and no `MTLArgumentEncoder` is involved.

K/V are stored half because decode cost is LINEAR in cached positions, so
at any real context the KV read dominates what a token moves: on the dense
1.7B at 4K it is ~0.9 GB against a 0.46 GB weight stream. The pages are
~112 MB per 512 positions, which is what actually bounds context on a 3 GB
phone. llama.cpp has shipped f16 KV by default for years and the CoreML
PagePool already stores fp16, so this makes the two backends agree rather
than breaking new ground.

### [gdn-scan2]  the register-resident delta-rule scan

`gdn_scan_batch` round-trips the dS x dS state matrix through device memory
every token -- about 4 device passes over S per token, which is its
bandwidth wall. Here one simdgroup owns (value head hv, output column j)
and its 32 lanes split the key dimension into KS = dS/32 register slices,
so S is loaded ONCE at entry and stored ONCE at exit. The two reductions
(`sk = sum_i S[i,j]*k[i]` and `o = sum_i S[i,j]*q[i]`) become `simd_sum`
over the key dim.

Numerics match `GDN.step`: the only change is the key-dim reduction ORDER,
and fp non-associativity stays well under the parity gate. COLS
(simdgroups per threadgroup, = output columns owned) is fixed at 4,
independent of dS.

### [srq-clamp]  why the clamp is arithmetic

The QAT trained with each quantized linear's input and output rounded to a
per-linear scale, so removing it does not merely lose precision -- it
changes the model. Measured: the same weights unclamped answer "the content
is unclear" where the clamped ones transcribe.

Two entry points because an INPUT is usually a shared buffer (one norm
feeds q, k and v), so clamping in place would corrupt the next reader,
while an OUTPUT belongs to its own linear alone and is safe in place.

This is also the mechanism behind the batched-prefill gap in
[gemm-tiles]: `rint()` is a step function, so ANY reordering of a reduction
moves some element a whole quantization step.

### [vit-attn]  the Qwen3-VL vision attention, and what was tried

One threadgroup per (head, tile of ntg/32 queries), one simdgroup per
query, K/V streamed in TK=32-key tiles through threadgroup memory shared by
all the simdgroups. Within a tile the softmax is LANE-PER-KEY: each lane
computes its key's whole dot (float4-vectorized) and one exp, and the
online-softmax state (m, l, lane-sliced acc) updates once per TILE rather
than per key.

Measured alternatives, all SLOWER: a per-key `simd_sum` + exp chain is
latency-bound at ~2x; half-staged tiles lose to 2-byte scalar loads; TK=16
pays more per-tile overhead than the occupancy gain returns.

The context lands at the head's slot of out [N][e].

### [vit-tower]  the mmproj tower on the GPU

f16 weights (dequanted once at load by `MetalViT` into plain half buffers) x
f32 activations, f32 accumulation. Each kernel mirrors the CPU ViT
(SIMD/ViT.swift) op for op and that engine is the oracle --
`gadeon-cli --vit` cross-gates the two forwards.

`f16w_gemm_mm`'s W is row-major [M][K] half in ggml's native [out][in] order,
so the CPU path's load-time transpose does not exist here. It shares the
64(M) x 32(N) tile, the half staging tiles (6144 B), the f32 accumulate and
the bounds-checked f32 spill path with the quantized GEMMs; what it does
NOT share is the K-tail guard, because ViT K values are not all multiples
of 32 and out-of-range lanes must load 0.

### [audio-attn]  the window that holds `chunk` positions including self

The bug this footnote exists to prevent, found once by bisection: the local
window holds `chunk` positions INCLUDING self, so the lag runs 0..past-1
and the encoding row `rel = o - qi` runs 1..past. Allowing rel = 0 hands
every query ONE EXTRA key. It is invisible in the first block, where that
key falls before position zero and is masked anyway, so the whole tower
looks structurally right and only the numbers are wrong.

What found it was dumping the reference's attn_weights and COUNTING the
nonzeros per query. No amount of staring at the formula did.

One thread per (position, head) -- the whole tower is a few hundred of
them, so there is nothing to gain from a fancier decomposition.

### [tg-reduce]  the two-stage threadgroup sum

`simd_sum` reduces 32 lanes in one instruction, and a threadgroup is at most
1024 threads = 32 simdgroups, so a SECOND `simd_sum` over the per-simdgroup
partials finishes any legal threadgroup in one more step. That is why there
is no loop here and why the scratch is 32 floats regardless of size.

Both barriers are load-bearing. The first orders the per-simdgroup writes
before simdgroup 0 reads them; the second orders that write of the total
before every thread reads it back. The `tii < (ntg + 31) / 32` guard zeroes
the lanes past the live simdgroup count, whose scratch slots were never
written.

The helper returns shmem[0] rather than the register it just computed
because only lane 0 of simdgroup 0 holds that value; every other thread has
to read it from memory, which is what the second barrier makes safe.

### [vision-block]  why a bidirectional block needs no mask term

A gemma-4 checkpoint with `use_bidirectional_attention: vision` lets the
tokens of ONE image attend to each other in both directions, on EVERY layer.
HF builds the sliding one as `AND(sliding_window, OR(causal, blockwise))`.

Do not trust `create_masks_for_vision_model`'s docstring here, which says the
global layers stay causal: that function is not the one the forward runs. The
forward puts `block_sequence_ids` into mask_kwargs and lets
`create_masks_for_generate` build BOTH masks from it. Measured against the
reference, sliding-only leaves layer 5 at 0.95 where both reach six nines.

Written literally that is a per-key predicate. It does not have to be. The
block always CONTAINS the query, so `[lo, T)` and `[blockLo, blockHi)` always
overlap, and their union is the single contiguous range
`[lo, max(T, blockHi))`. The window still bounds it from below, so an image
longer than the window is cut by `lo` exactly as HF's AND cuts it.

That is why `attn_end` returns a bound rather than a mask, and why
`attn_batch` needed one changed argument instead of a third kernel. Only
`fa_mask` tests it per row, because a matrix-unit tile carries eight queries
whose ends differ.

THE CONSTRAINT this puts on the host: the block's forward keys must already
be in the pool when the query runs, so a whole vision block must ride ONE
chunk. A per-token prefill cannot express this mask at all -- at query q only
0..q have been appended.

### [rope-pairing]  why `rope_pair` takes indices and not a pairing rule

Seven kernels rotate a pair, and they do NOT agree on which two elements
are a pair. Three regimes are live at once:

  rope_neox / rope_batch / rope_mrope_batch   (i, i + nRot/2)
  rope_gemma / rope_gemma_batch / vit_rope    (j, j + headDim/2)
  gemma_vit_rope                              (j, j + per/2) WITHIN an axis
                                              half, the halves driven by
                                              the patch's x and y

These are different permutations of the same head, not different constants.
A helper that derived the partner index would have to be told which regime
it was in, which is the same decision moved somewhere the caller cannot
see -- and getting it wrong is silent: every variant still fills the buffer
with finite numbers and still produces fluent text. Reusing vit_rope's
pairing for gemma's vision tower scored 0.69 and read as a weight bug.

So the caller computes both indices and the helper only rotates. It takes
cos/sin rather than an angle for the same reason: the two vision ropes read
theirs from host-built tables, and an angle-taking helper would have shut
them out of the sharing.

### [gemm-tiles]  the simdgroup-matrix GEMM, its tile precision, and its cost

SHAPE. A 128-thread threadgroup owns a 64(M) x 32(N) output tile. Weight
rows are dequantized into a 64x32 staging tile and activations into a
32x32 one, both in threadgroup memory, then multiplied on the 8x8 hardware
matrix units. The scalar q2_0_gemm next door does the same product with
per-lane accumulation and loses about 10x, which is why prefill routes
here and only decode uses the GEMV.

TILE PRECISION. Half tiles are the shipping default. They halve the
staging memory (6144 B against 12288), and since prefill is compute-bound
the win is more resident threadgroups in flight rather than fewer bytes
moved. MEASURED on an M3, interleaved A/B with a cooldown before every run
-- uncooled runs throttle and can invert the result: 27B 42.7 -> 47.6 t/s
(+11.5%, four reps 1.104-1.127), 1.7B 602 -> 638 t/s (+6.0%). Parity holds
at 1.9e-4 RELATIVE error, inside fp16's own 4.9e-4 epsilon.

The f32 twins exist as the A/B instrument, not as a fallback. They were
written to test whether fp16 tiles were what kept batched gemma prefill
from reproducing the per-token path. They are NOT: the batched-vs-per-token
cosine moved 0.9885 -> 0.9895, i.e. nowhere, and both land equally close to
HF's own logits (0.99875 half, 0.99890 f32, 0.99901 per-token). The real
mechanism is SRQ's `rint()`, which is discontinuous, so ANY reordering of a
reduction flips a whole quantization step. f32 tiles cost 6% of prefill for
that non-difference. `LLM_F16_TILES`=0 selects them, and its reach is wider
than the name suggests: every gemma projection goes through
MetalEnc.linear(X:N:), so it swaps the vision and audio towers too.

THE SPILL PATH. A tile that runs off the end of M or N cannot
`simdgroup_store` straight to dst, so it stages through threadgroup memory as
F32 and copies out row by row. That is why a partial tile needs 8192 B even
at half precision -- the host allocates 6144 only when M % 64 == 0 and
N % 32 == 0. Getting that wrong is a threadgroup-memory overrun, not a
wrong answer.

WHY THE WEIGHTS STAY QUANTIZED. Blocks are read straight out of the mapped
GGUF and expanded in registers. The gemma vision tower alone is 0.167 B
parameters, which would be 336 MB dequantized to resident f16 and is
nothing at all this way.

### [flash-attn]  the shared attention body, and why it is shared

`attn_head` (one query) and `attn_batch` (N queries) were 120-line twins whose
ONLY differences were where T and lo come from and the base offset into
q / gate / out. Everything subtle -- the tile striping, the online softmax,
the four-wide V accumulator, the NSG combine -- was written twice. That is
the dangerous kind of duplication here: it is precisely the seam where the
batched path can drift from the per-token one, and the SRQ noise floor
(see [gemm-tiles]) is wide enough to hide a small drift from cosine gates.
Passing q, gate and out ALREADY OFFSET removes the difference entirely, so
the loop exists once and the two cannot disagree.

WHY O(hd) THREADGROUP MEMORY. A scores[T] buffer capped context near 8K on
the 32 KB threadgroup budget while the page pool advertises 1M positions.
The online softmax keeps only (max, denom, dim-sliced acc) per simdgroup,
so the request is 5*hd+8 floats and is CONSTANT in T -- which is what lets
a long prefill chunk run at all.

WHY half4 LOADS. Each LANE walks a whole head vector while the 32 lanes sit
kvDim apart, so a scalar loop issues 32 SCATTERED requests per step and
never fills a cache line. Vectorizing quarters the request count, and half
storage halves the bytes each one moves. Once the K dot was vectorized, the
V gather became the request count that set the pace, so its slice was
widened to four-wide too: lane L takes bytes 8L..8L+7, so a warp sweeps 256
contiguous bytes. Vector only when hd % 4 == 0, which is also what makes
the kvh * hd row base 8B-aligned; the scalar tail covers any other
geometry.

THE WINDOW IS A READ BOUND, not an eviction. `lo` starts the key sweep late
rather than dropping pages, so a shared layer windows its SOURCE's full
history exactly as HF does and the pool stays append-only -- which is what
makes index-based rollback possible. In the batched kernel every query row
computes its own lo, since each sits at its own absolute position.

### [flash-attn-mm]  the same attention on the matrix units, and why

WHY A SECOND BODY AT ALL, when [flash-attn] exists to stop exactly that. The
two are not variants of one algorithm: the scalar body is lane-per-key with a
per-lane online softmax, the matrix body is tile-per-block with the softmax in
threadgroup memory between two products. Merging them would mean a kernel
carrying both decompositions behind a flag, which is worse than two honest
bodies. What IS shared is everything that is not the product itself -- the
paged row addressing (`kv_row`), the KV table, `AttnBatchArgs`, the causal and
window masks, and the same per-row algebra in the same order. And the scalar
body is not vestigial: it is the ONLY path on a GPU without matrix units.

WHY IT IS FASTER, in one number. Per 32-key tile the scalar kernel issues 256
float4 FMAs per lane and performs exactly the useful MACs -- it is already at
~100% ALU efficiency, so there is no waste to reclaim. A
`simdgroup_multiply_accumulate` does 8x8x8 = 512 MACs per issue against a
float4 FMA's 32x4 = 128. Four times the MACs per instruction is the entire
available win, and it lands on the term that grows with context.

MEASURED, M3, gemma-4-12B, ctx 4096, one binary with LLM_ATTN_MM as the knob,
ABBA with a cooldown before every run, scoring the warm-up prefill pass's GPU
milliseconds:

    rep   scalar            matrix            speedup
      1   67830 ms  60.4    45155 ms  90.7    1.502
      2   76666 ms  53.4    45577 ms  89.9    1.682
      3   95725 ms  42.8    42861 ms  95.6    2.233

Take 1.5x, not 2.2x: the scalar arm DEGRADES across the run (67830 -> 95725, a
41% thermal loss over ~15 minutes) while the matrix arm is flat (45155 /
45577 / 42861). The rising ratio is the baseline collapsing. Rep 1 is the
coldest machine AND puts the matrix arm in the warmer slot, so 1.5x is a lower
bound. That the matrix arm barely moves under sustained load is its own
finding -- long sessions are where this model actually lives.

Per chunk at pos 3968: 2028 -> 1518 ms. Against the measured attention-free
base (~1040 ms, flat in context) that is attention 988 -> 478 ms, and the
growth slope above the window 0.24 -> 0.106 ms/position.

THE SHAPE. FA_Q=8 query rows and FA_K=32 keys per iteration; the four
simdgroups split the KEY range for the scores and the HEAD DIM for the
context, so neither product is computed twice. The correction rides a DIAGONAL
matrix because a simdgroup matrix has no scalar multiply -- one extra product
per dim tile against the sixteen the context costs.

THREE THINGS THAT WOULD SHIP BROKEN. Every simdgroup storing to one scratch
window in the final normalize (they need their own; FA_Q*FA_K is exactly
NSG*64). The correction matrix's off-diagonal left uninitialized on the first
tile. And the PAGE ALIGNMENT: an 8-key block is read as ONE strided
`simdgroup_load`, which is contiguous only inside a single page, so the host
gates on `P % 8 == 0`. The self-test's P=4 pool scored 0/12 before that term
existed -- shipping pools are 512, and depending on that silently is how a
kernel acquires an assumption nobody wrote down.

PARITY. The ternary self-test passes whole, including `batched prefill 8/8`
and `long-context 6/6` on exact token ids through this kernel; gemma is 15/15
against the SIMD oracle. Both `attn_head` byte-goldens are UNCHANGED, so
decode is untouched. The four `attn_batch` goldens move by 1.4e-4, which is
the fp16-operand magnitude (Q is staged half, P is stored half) and sits
inside fp16's own 4.9e-4 epsilon, as [gemm-tiles] already records for the
shipping half-tile GEMM.

WHAT `f16w_gemm_mm` SHARES, and what it does not. It was first judged
unshareable because its A tile comes from a plain dequantized half array
rather than block-quantized bytes in the mmap, and it needs a K-tail guard
the quantized six never do (their K is always a multiple of the block size).
That was the wrong conclusion drawn from a true premise: it differs on ONE
axis, and the units it agrees on extract cleanly. It now shares
`simd_mm_slice` and `store_mm_tile` with `gemm_mm_impl` and keeps only its
own A/B tile staging. The lesson generalizes -- when two bodies differ on
one axis, extract the axes they AGREE on rather than giving up on sharing.
*/

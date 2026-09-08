// ggml IQ and K quant decode on the GPU, ported from LLM/src/Quantize (which
// is gated bit-for-bit against ggml). dq_sub decodes ONE 32-weight sub-block;
// gemv, dequant and the embedding gather all build on it, so eleven types
// cost three kernels. [iq-subblock-decode]

#define TY_Q2K     10
#define TY_Q3K     11
#define TY_Q4K     12
#define TY_Q5K     13
#define TY_Q6K     14
#define TY_IQ2XXS  16
#define TY_IQ2XS   17
#define TY_IQ3XXS  18
#define TY_IQ1S    19
#define TY_IQ3S    21
#define TY_IQ2S    22
#define TY_IQ4XS   23
#define TY_IQ1M    29

// A super-block is 50..144 bytes, so nothing inside one is reliably aligned
// for a ushort/uint cast. Every multi-byte field is assembled from bytes.
inline ushort iq_u16(device const uchar *p) {
    return (ushort) p[0] | ((ushort) p[1] << 8);
}

inline uint iq_u32(device const uchar *p) {
    return (uint) p[0] | ((uint) p[1] << 8)
         | ((uint) p[2] << 16) | ((uint) p[3] << 24);
}

inline float iq_f16(device const uchar *p) {
    return (float) as_type<half>(iq_u16(p));
}

inline float iq_sgn(uchar signs, uint j, float v) {
    return (signs & kmask_iq2xs[j]) ? -v : v;
}

inline void dq_sub_iq1s(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const ushort h = iq_u16(b + 34 + ib * 2);
    const float dl = d * (float) (2 * ((h >> 12) & 7) + 1);
    const float delta = (h & 0x8000) ? -0.125f : 0.125f;
    for (uint l = 0; l < 4; ++l) {
        const uint e = iq1s_grid_gpu[b[2 + ib * 4 + l]
                                     | (((h >> (3 * l)) & 7) << 8)];
        for (uint j = 0; j < 8; ++j) {
            const float n = (float) ((e >> (8 * (j % 4) + 4 * (j / 4))) & 0xF);
            w[l * 8 + j] = dl * (n - 1.0f + delta);
        }
    }
}

inline void dq_sub_iq1m(device const uchar *b, uint ib, thread float *w) {
    ushort sc[4];
    for (uint i = 0; i < 4; ++i) { sc[i] = iq_u16(b + 48 + i * 2); }
    const ushort bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0)
                      | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = (float) as_type<half>(bits);
    const ushort s = sc[ib / 2];
    const uint sh = 6 * (ib % 2);
    const float dl1 = d * (float) (2 * ((s >> sh) & 7) + 1);
    const float dl2 = d * (float) (2 * ((s >> (sh + 3)) & 7) + 1);
    const uchar h0 = b[32 + ib * 2], h1 = b[33 + ib * 2];
    for (uint l = 0; l < 4; ++l) {
        const uchar h = (l < 2) ? h0 : h1;
        const uint up = (l % 2 == 0) ? ((uint) h << 8) : ((uint) h << 4);
        const uint e = iq1s_grid_gpu[b[ib * 4 + l] | (up & 0x700)];
        const float delta = (h & ((l % 2 == 0) ? 0x08 : 0x80))
                          ? -0.125f : 0.125f;
        const float dl = (l < 2) ? dl1 : dl2;
        for (uint j = 0; j < 8; ++j) {
            const float n = (float) ((e >> (8 * (j % 4) + 4 * (j / 4))) & 0xF);
            w[l * 8 + j] = dl * (n - 1.0f + delta);
        }
    }
}

inline void dq_sub_iq2xxs(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const uint a0 = iq_u32(b + 2 + ib * 8);
    const uint a1 = iq_u32(b + 6 + ib * 8);
    const float db = d * (0.5f + (float) (a1 >> 28)) * 0.25f;
    for (uint l = 0; l < 4; ++l) {
        const uint gi = ((a0 >> (8 * l)) & 0xFF) * 2;
        const uint lo = iq2xxs_grid_u32[gi], hi = iq2xxs_grid_u32[gi + 1];
        const uchar signs = ksigns_iq2xs[(a1 >> (7 * l)) & 127];
        for (uint j = 0; j < 8; ++j) {
            const uint e = (j < 4) ? lo : hi;
            const float g = (float) ((e >> (8 * (j % 4))) & 0xFF);
            w[l * 8 + j] = iq_sgn(signs, j, db * g);
        }
    }
}

inline void dq_sub_iq2xs(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const uchar sc = b[66 + ib];
    const float db0 = d * (0.5f + (float) (sc & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
    for (uint l = 0; l < 4; ++l) {
        const ushort q = iq_u16(b + 2 + (ib * 4 + l) * 2);
        const uint gi = (q & 511) * 2;
        const uint lo = iq2xs_grid_u32[gi], hi = iq2xs_grid_u32[gi + 1];
        const uchar signs = ksigns_iq2xs[q >> 9];
        const float dl = (l < 2) ? db0 : db1;
        for (uint j = 0; j < 8; ++j) {
            const uint e = (j < 4) ? lo : hi;
            const float g = (float) ((e >> (8 * (j % 4))) & 0xFF);
            w[l * 8 + j] = iq_sgn(signs, j, dl * g);
        }
    }
}

inline void dq_sub_iq2s(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const uchar sc = b[74 + ib];
    const uchar qh = b[66 + ib];
    const float db0 = d * (0.5f + (float) (sc & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
    for (uint l = 0; l < 4; ++l) {
        const uint gi = (b[2 + ib * 4 + l]
                         | ((((uint) qh) << (8 - 2 * l)) & 0x300)) * 2;
        const uint lo = iq2s_grid_u32[gi], hi = iq2s_grid_u32[gi + 1];
        const uchar signs = b[34 + ib * 4 + l];
        const float dl = (l < 2) ? db0 : db1;
        for (uint j = 0; j < 8; ++j) {
            const uint e = (j < 4) ? lo : hi;
            const float g = (float) ((e >> (8 * (j % 4))) & 0xFF);
            w[l * 8 + j] = iq_sgn(signs, j, dl * g);
        }
    }
}

inline void dq_sub_iq3xxs(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const uint aux = iq_u32(b + 66 + ib * 4);
    const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
    for (uint l = 0; l < 4; ++l) {
        const uchar signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
        const uint g1 = iq3xxs_grid[b[2 + ib * 8 + 2 * l]];
        const uint g2 = iq3xxs_grid[b[2 + ib * 8 + 2 * l + 1]];
        for (uint j = 0; j < 4; ++j) {
            w[l * 8 + j] = iq_sgn(signs, j,
                db * (float) ((g1 >> (8 * j)) & 0xFF));
            w[l * 8 + j + 4] = iq_sgn(signs, j + 4,
                db * (float) ((g2 >> (8 * j)) & 0xFF));
        }
    }
}

inline void dq_sub_iq3s(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const uchar sc = b[106 + ib / 2];
    const float db = (ib % 2 == 0) ? d * (float) (1 + 2 * (sc & 0xF))
                                   : d * (float) (1 + 2 * (sc >> 4));
    const uchar qh = b[66 + ib];
    for (uint l = 0; l < 4; ++l) {
        const uint base = 2 + ib * 8 + 2 * l;
        const uint g1 = iq3s_grid[b[base]
            | ((((uint) qh) << (8 - 2 * l)) & 256)];
        const uint g2 = iq3s_grid[b[base + 1]
            | ((((uint) qh) << (7 - 2 * l)) & 256)];
        const uchar signs = b[74 + ib * 4 + l];
        for (uint j = 0; j < 4; ++j) {
            w[l * 8 + j] = iq_sgn(signs, j,
                db * (float) ((g1 >> (8 * j)) & 0xFF));
            w[l * 8 + j + 4] = iq_sgn(signs, j + 4,
                db * (float) ((g2 >> (8 * j)) & 0xFF));
        }
    }
}

// Four nibbles of one 32-bit word as floats, byte 0 first (little endian),
// so element 4i of a sub-block is the low byte of word i.
__attribute__((always_inline))
inline float4 nib4(uint word) {
    return float4(as_type<uchar4>(word));
}

inline void put4(thread float *w, uint at, float4 v) {
    w[at] = v.x;
    w[at + 1] = v.y;
    w[at + 2] = v.z;
    w[at + 3] = v.w;
}

// A 136-byte block is 8-byte aligned, so the 16 codes of a sub-block are two
// uint2 loads; the low nibble of byte j is element j and the high one j+16.
inline void dq_sub_iq4xs(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const ushort sh = iq_u16(b + 2);
    const uchar sl = b[4 + ib / 2];
    const int ls = (int) ((sl >> (4 * (ib % 2))) & 0xF)
                 | (int) (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * (float) (ls - 32);
    device const uint2 * q2 = (device const uint2 *) (b + 8 + ib * 16);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint2 v = q2[k];
        #pragma clang loop unroll(full)
        for (uint m = 0; m < 2; ++m) {
            const uint word = m == 0 ? v.x : v.y;
            const uchar4 lo = as_type<uchar4>(word & 0x0F0F0F0Fu);
            const uchar4 hi = as_type<uchar4>((word >> 4) & 0x0F0F0F0Fu);
            const uint at = (k * 2 + m) * 4;
            put4(w, at, dl * float4(kvalues_iq4nl[lo.x], kvalues_iq4nl[lo.y],
                                    kvalues_iq4nl[lo.z], kvalues_iq4nl[lo.w]));
            put4(w, at + 16, dl * float4(kvalues_iq4nl[hi.x],
                                         kvalues_iq4nl[hi.y],
                                         kvalues_iq4nl[hi.z],
                                         kvalues_iq4nl[hi.w]));
        }
    }
}

inline void dq_sub_q2k(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b + 80), dmin = iq_f16(b + 82);
    const uint shift = 2 * (ib % 4);
    const uint qOff = 16 + (ib / 4) * 32;
    for (uint h = 0; h < 2; ++h) {
        const uchar sc = b[ib * 2 + h];
        const float dl = d * (float) (sc & 0xF);
        const float ml = dmin * (float) (sc >> 4);
        for (uint l = 0; l < 16; ++l) {
            const uchar q = b[qOff + h * 16 + l];
            w[h * 16 + l] = dl * (float) ((q >> shift) & 3) - ml;
        }
    }
}

inline void dq_sub_q3k(device const uchar *b, uint ib, thread float *w) {
    const float dAll = iq_f16(b + 108);
    uint aux[4];
    aux[0] = iq_u32(b + 96);
    aux[1] = iq_u32(b + 100);
    aux[2] = iq_u32(b + 104);
    const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    const uint tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const uint shift = 2 * (ib % 4);
    const uint qOff = 32 + (ib / 4) * 32;
    const uchar m = (uchar) (1u << ib);
    for (uint h = 0; h < 2; ++h) {
        const uint isc = ib * 2 + h;
        const int s = (int) (char) ((aux[isc / 4] >> (8 * (isc % 4))) & 0xFF);
        const float dl = dAll * (float) (s - 32);
        for (uint l = 0; l < 16; ++l) {
            const uint at = h * 16 + l;
            const float bump = (b[at] & m) ? 0.0f : 4.0f;
            w[at] = dl * ((float) ((b[qOff + at] >> shift) & 3) - bump);
        }
    }
}

// ggml's get_scale_min_k4 over the twelve packed pairs at byte 4, shared by
// q4_K and q5_K.
inline float2 dq_scale_min(device const uchar *b, uint j) {
    float2 out;
    if (j < 4) {
        out.x = (float) (b[4 + j] & 63);
        out.y = (float) (b[8 + j] & 63);
    } else {
        out.x = (float) ((b[8 + j] & 0xF) | ((b[j] >> 6) << 4));
        out.y = (float) ((b[8 + j] >> 4) | ((b[4 + j] >> 6) << 4));
    }
    return out;
}

// A 144-byte block is 16-byte aligned, so a sub-block's 32 codes are two
// uint4 loads; even and odd sub-blocks share the bytes, low and high nibble.
inline void dq_sub_q4k(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b), dmin = iq_f16(b + 2);
    const float2 sm = dq_scale_min(b, ib);
    const float dv = d * sm.x, ov = dmin * sm.y;
    device const uint4 * q4 = (device const uint4 *) (b + 16 + (ib / 2) * 32);
    const uint sh = 4 * (ib & 1);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint4 v = q4[k];
        put4(w, k * 16, dv * nib4((v.x >> sh) & 0x0F0F0F0Fu) - ov);
        put4(w, k * 16 + 4, dv * nib4((v.y >> sh) & 0x0F0F0F0Fu) - ov);
        put4(w, k * 16 + 8, dv * nib4((v.z >> sh) & 0x0F0F0F0Fu) - ov);
        put4(w, k * 16 + 12, dv * nib4((v.w >> sh) & 0x0F0F0F0Fu) - ov);
    }
}

// Q4_K's shape plus the fifth bit: bit ib of qh[l] lifts element l by 16.
inline void dq_sub_q5k(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b), dmin = iq_f16(b + 2);
    const float2 sm = dq_scale_min(b, ib);
    const float dv = d * sm.x, ov = dmin * sm.y;
    device const uint4 * q4 = (device const uint4 *) (b + 48 + (ib / 2) * 32);
    device const uint4 * h4 = (device const uint4 *) (b + 16);
    const uint sh = 4 * (ib & 1);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint4 v = q4[k];
        const uint4 h = h4[k];
        const uint4 n = ((v >> sh) & 0x0F0F0F0Fu)
                      | (((h >> ib) & 0x01010101u) << 4);
        put4(w, k * 16, dv * nib4(n.x) - ov);
        put4(w, k * 16 + 4, dv * nib4(n.y) - ov);
        put4(w, k * 16 + 8, dv * nib4(n.z) - ov);
        put4(w, k * 16 + 12, dv * nib4(n.w) - ov);
    }
}

// A 210-byte block is only 2-byte aligned, so the codes come in as ushort
// pairs assembled into words: low nibble or high by r, the two high bits
// from qh, one int8 scale per 16 elements.
inline void dq_sub_q6k(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b + 208);
    const uint n = ib / 4, r = ib % 4;
    device const ushort * ql = (device const ushort *)
        (b + n * 64 + (r % 2) * 32);
    device const ushort * qh = (device const ushort *) (b + 128 + n * 32);
    const uint scOff = 192 + n * 8 + 2 * r;
    const float s0 = d * (float) (int) (char) b[scOff];
    const float s1 = d * (float) (int) (char) b[scOff + 1];
    const uint shq = (r < 2) ? 0 : 4;
    const uint shh = 2 * r;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const uint lw = (uint) ql[2 * i] | ((uint) ql[2 * i + 1] << 16);
        const uint hw = (uint) qh[2 * i] | ((uint) qh[2 * i + 1] << 16);
        const uint q = ((lw >> shq) & 0x0F0F0F0Fu)
                     | (((hw >> shh) & 0x03030303u) << 4);
        const float s = i < 4 ? s0 : s1;
        put4(w, i * 4, s * (nib4(q) - 32.0f));
    }
}

inline void dq_sub(uint ty, device const uchar *b, uint ib, thread float *w) {
    switch (ty) {
        case TY_IQ1S:   dq_sub_iq1s(b, ib, w);   break;
        case TY_IQ1M:   dq_sub_iq1m(b, ib, w);   break;
        case TY_IQ2XXS: dq_sub_iq2xxs(b, ib, w); break;
        case TY_IQ2XS:  dq_sub_iq2xs(b, ib, w);  break;
        case TY_IQ2S:   dq_sub_iq2s(b, ib, w);   break;
        case TY_IQ3XXS: dq_sub_iq3xxs(b, ib, w); break;
        case TY_IQ3S:   dq_sub_iq3s(b, ib, w);   break;
        case TY_IQ4XS:  dq_sub_iq4xs(b, ib, w);  break;
        case TY_Q2K:    dq_sub_q2k(b, ib, w);    break;
        case TY_Q3K:    dq_sub_q3k(b, ib, w);    break;
        case TY_Q5K:    dq_sub_q5k(b, ib, w);    break;
        case TY_Q6K:    dq_sub_q6k(b, ib, w);    break;
        case TY_Q4K:    dq_sub_q4k(b, ib, w);    break;
        default:
            for (uint j = 0; j < 32; ++j) { w[j] = NAN; }
            break;
    }
}

// The fused form the mat-vec kernels take: a lane's SPAN consecutive
// sub-blocks (Q4_K and Q5_K pairs share their code bytes, so a lane takes
// both nibbles of one 32-byte load) dotted against R1 staged activation
// columns of SPAN*8 float4 each, the scale and minimum applied ONCE per
// sub-block: Sum(dv*q - ov)*y is dv*Sum(q*y) - ov*Sum(y), and Sum(y) per
// 16 elements (`sy16`, SPAN*2 per column) comes from the caller, computed
// once per block rather than once per row. [kquant-gemv]
__attribute__((always_inline))
inline float hsum(float4 v) { return v.x + v.y + v.z + v.w; }

inline uint kq_byte(uint u, uint i) { return (u >> (8u * i)) & 0xFFu; }

// get_scale_min_k4 over the 16-byte header {d, dmin, scales[12]} held in
// one register, both sides three ops so the select is free of a branch.
__attribute__((always_inline))
inline float2 kq_sm(uint4 h, uint j) {
    const uint k = j & 3u;
    const uint u = kq_byte(h.y, k), v = kq_byte(h.z, k), t = kq_byte(h.w, k);
    const uint sc = (j < 4u) ? (u & 63u) : ((t & 15u) | ((u >> 6) << 4));
    const uint mn = (j < 4u) ? (v & 63u) : ((t >> 4) | ((v >> 6) << 4));
    return float2((float) sc, (float) mn);
}

inline float kq_f16lo(uint u) {
    return (float) as_type<half>((ushort) (u & 0xFFFFu));
}

inline float kq_f16hi(uint u) {
    return (float) as_type<half>((ushort) (u >> 16));
}

// One decoded 32-weight sub-block: the codes as eight float4 with the
// per-16 scale folded out, so a column costs eight FMAs and four scalars:
// a0*Sum(f*y)[0..15] + a1*Sum(f*y)[16..31] + b0*Sum(y)[0..15]
// + b1*Sum(y)[16..31].
struct KqSub {
    float4 f[8];
    float a0, a1, b0, b1;
};

template <ushort R1, ushort CS, typename Y>
__attribute__((always_inline))
inline void kq_dot(const thread KqSub &w, Y yl, thread const float *sy16,
                   thread float *s) {
    #pragma clang loop unroll(full)
    for (ushort c = 0; c < R1; ++c) {
        Y y = yl + c * CS;
        const float4 t0 = w.f[0] * y[0] + w.f[1] * y[1];
        const float4 u0 = w.f[2] * y[2] + w.f[3] * y[3];
        const float4 t1 = w.f[4] * y[4] + w.f[5] * y[5];
        const float4 u1 = w.f[6] * y[6] + w.f[7] * y[7];
        s[c] = w.a0 * hsum(t0 + u0) + w.a1 * hsum(t1 + u1)
             + w.b0 * sy16[c * 2] + w.b1 * sy16[c * 2 + 1];
    }
}

// Lanes 2j and 2j+1 own sub-blocks 2j and 2j+1, which share one 32-byte
// code span, so each loads half and the pair exchanges over the simdgroup.
__attribute__((always_inline))
inline void kq_dq_q4k(device const uchar *b, uint ib, thread KqSub &w) {
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    device const uint4 * q4 = (device const uint4 *) (b + 16 + (ib / 2) * 32);
    const uint4 mine = q4[ib & 1];
    const uint4 other = simd_shuffle_xor(mine, 1);
    const uint sh = 4 * (ib & 1);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint4 v = (k == (ib & 1)) ? mine : other;
        const uint4 lo = (v >> sh) & 0x0F0F0F0Fu;
        w.f[k * 4] = nib4(lo.x);
        w.f[k * 4 + 1] = nib4(lo.y);
        w.f[k * 4 + 2] = nib4(lo.z);
        w.f[k * 4 + 3] = nib4(lo.w);
    }
    w.a0 = d * sm.x;
    w.a1 = w.a0;
    w.b0 = -dmin * sm.y;
    w.b1 = w.b0;
}

__attribute__((always_inline))
inline void kq_dq_q5k(device const uchar *b, uint ib, thread KqSub &w) {
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    device const uint4 * q4 = (device const uint4 *) (b + 48 + (ib / 2) * 32);
    device const uint4 * h4 = (device const uint4 *) (b + 16);
    const uint4 mine = q4[ib & 1];
    const uint4 other = simd_shuffle_xor(mine, 1);
    const uint sh = 4 * (ib & 1);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint4 v = (k == (ib & 1)) ? mine : other;
        const uint4 h = h4[k];
        const uint4 lo = ((v >> sh) & 0x0F0F0F0Fu)
                       | (((h >> ib) & 0x01010101u) << 4);
        w.f[k * 4] = nib4(lo.x);
        w.f[k * 4 + 1] = nib4(lo.y);
        w.f[k * 4 + 2] = nib4(lo.z);
        w.f[k * 4 + 3] = nib4(lo.w);
    }
    w.a0 = d * sm.x;
    w.a1 = w.a0;
    w.b0 = -dmin * sm.y;
    w.b1 = w.b0;
}

__attribute__((always_inline))
inline void kq_dq_q6k(device const uchar *b, uint ib, thread KqSub &w) {
    const float d = iq_f16(b + 208);
    const uint n = ib / 4, r = ib % 4;
    device const packed_ushort4 * ql = (device const packed_ushort4 *)
        (b + n * 64 + (r % 2) * 32);
    device const packed_ushort4 * qh = (device const packed_ushort4 *)
        (b + 128 + n * 32);
    const ushort ss = *(device const ushort *) (b + 192 + n * 8 + 2 * r);
    const float s0 = d * (float) (char) (ss & 0xFFu);
    const float s1 = d * (float) (char) (ss >> 8);
    const uint shq = (r < 2) ? 0 : 4;
    const uint shh = 2 * r;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        const uint2 lw = as_type<uint2>(ushort4(ql[i]));
        const uint2 hw = as_type<uint2>(ushort4(qh[i]));
        w.f[2 * i] = nib4(((lw.x >> shq) & 0x0F0F0F0Fu)
                          | (((hw.x >> shh) & 0x03030303u) << 4));
        w.f[2 * i + 1] = nib4(((lw.y >> shq) & 0x0F0F0F0Fu)
                              | (((hw.y >> shh) & 0x03030303u) << 4));
    }
    w.a0 = s0;
    w.a1 = s1;
    w.b0 = -32.0f * s0;
    w.b1 = -32.0f * s1;
}

__attribute__((always_inline))
inline void kq_dq_iq4xs(device const uchar *b, uint ib, thread KqSub &w) {
    const float d = iq_f16(b);
    const ushort sh = iq_u16(b + 2);
    const uchar sl = b[4 + ib / 2];
    const int ls = (int) ((sl >> (4 * (ib % 2))) & 0xF)
                 | (int) (((sh >> (2 * ib)) & 3) << 4);
    device const uint2 * q2 = (device const uint2 *) (b + 8 + ib * 16);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint2 v = q2[k];
        #pragma clang loop unroll(full)
        for (uint m = 0; m < 2; ++m) {
            const uint word = m == 0 ? v.x : v.y;
            const uchar4 lo = as_type<uchar4>(word & 0x0F0F0F0Fu);
            const uchar4 hi = as_type<uchar4>((word >> 4) & 0x0F0F0F0Fu);
            const uint at = k * 2 + m;
            w.f[at] = float4(kvalues_iq4nl[lo.x], kvalues_iq4nl[lo.y],
                             kvalues_iq4nl[lo.z], kvalues_iq4nl[lo.w]);
            w.f[at + 4] = float4(kvalues_iq4nl[hi.x], kvalues_iq4nl[hi.y],
                                 kvalues_iq4nl[hi.z], kvalues_iq4nl[hi.w]);
        }
    }
    w.a0 = d * (float) (ls - 32);
    w.a1 = w.a0;
    w.b0 = 0.0f;
    w.b1 = 0.0f;
}

template <ushort F4>
struct KqSlice;

template <typename D, ushort F4>
__attribute__((always_inline))
inline void kq_dq_generic_slice(device const uchar *b, uint ib, uint p,
                                thread KqSlice<F4> &w) {
    float v[32];
    D::dq(b, ib, v);
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) {
        const uint at = (p * F4 + j) * 4;
        w.f[j] = float4(v[at], v[at + 1], v[at + 2], v[at + 3]);
    }
    w.a = 1.0f;
    w.b = 0.0f;
}

// A decoder with no fused form: materialize, then dot.
template <typename D>
__attribute__((always_inline))
inline void kq_dq_generic(device const uchar *b, uint ib, thread KqSub &w) {
    float v[32];
    D::dq(b, ib, v);
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < 8; ++j) {
        w.f[j] = float4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
    }
    w.a0 = 1.0f;
    w.a1 = 1.0f;
    w.b0 = 0.0f;
    w.b1 = 0.0f;
}

// A slice of a sub-block, F4 float4 of its eight, with one scale pair, for
// a kernel that cannot afford a column's 32 activations in registers. The
// minimum term is linear, so it is applied per slice against that slice's
// activation sum.
template <ushort F4>
struct KqSlice {
    float4 f[F4];
    float a, b;
};

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_q4k_slice(device const uchar *b, uint ib, uint p,
                            thread KqSlice<F4> &w) {
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    device const uint4 * q4 = (device const uint4 *) (b + 16 + (ib / 2) * 32);
    const uint4 mine = q4[ib & 1];
    const uint4 other = simd_shuffle_xor(mine, 1);
    const uint sh = 4 * (ib & 1);
    const uint h = (p * F4) / 4;
    const uint4 v = (h == (ib & 1)) ? mine : other;
    const uint4 lo = (v >> sh) & 0x0F0F0F0Fu;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) { w.f[j] = nib4(lo[(p * F4) % 4 + j]); }
    w.a = d * sm.x;
    w.b = -dmin * sm.y;
}

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_q5k_slice(device const uchar *b, uint ib, uint p,
                            thread KqSlice<F4> &w) {
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    device const uint4 * q4 = (device const uint4 *) (b + 48 + (ib / 2) * 32);
    device const uint4 * h4 = (device const uint4 *) (b + 16);
    const uint4 mine = q4[ib & 1];
    const uint4 other = simd_shuffle_xor(mine, 1);
    const uint sh = 4 * (ib & 1);
    const uint h = (p * F4) / 4;
    const uint4 v = (h == (ib & 1)) ? mine : other;
    const uint4 hb = h4[h];
    const uint4 lo = ((v >> sh) & 0x0F0F0F0Fu)
                   | (((hb >> ib) & 0x01010101u) << 4);
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) { w.f[j] = nib4(lo[(p * F4) % 4 + j]); }
    w.a = d * sm.x;
    w.b = -dmin * sm.y;
}

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_q6k_slice(device const uchar *b, uint ib, uint p,
                            thread KqSlice<F4> &w) {
    const float d = iq_f16(b + 208);
    const uint n = ib / 4, r = ib % 4;
    device const packed_ushort4 * ql = (device const packed_ushort4 *)
        (b + n * 64 + (r % 2) * 32);
    device const packed_ushort4 * qh = (device const packed_ushort4 *)
        (b + 128 + n * 32);
    const uint h = (p * F4) / 4;
    const float sc = d * (float) (char) b[192 + n * 8 + 2 * r + h];
    const uint shq = (r < 2) ? 0 : 4;
    const uint shh = 2 * r;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; j += 2) {
        const uint i = (p * F4 + j) / 2;
        const uint2 lw = as_type<uint2>(ushort4(ql[i]));
        const uint2 hw = as_type<uint2>(ushort4(qh[i]));
        w.f[j] = nib4(((lw.x >> shq) & 0x0F0F0F0Fu)
                      | (((hw.x >> shh) & 0x03030303u) << 4));
        w.f[j + 1] = nib4(((lw.y >> shq) & 0x0F0F0F0Fu)
                          | (((hw.y >> shh) & 0x03030303u) << 4));
    }
    w.a = sc;
    w.b = -32.0f * sc;
}

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_iq4xs_slice(device const uchar *b, uint ib, uint p,
                              thread KqSlice<F4> &w) {
    const float d = iq_f16(b);
    const ushort sh = iq_u16(b + 2);
    const uchar sl = b[4 + ib / 2];
    const int ls = (int) ((sl >> (4 * (ib % 2))) & 0xF)
                 | (int) (((sh >> (2 * ib)) & 3) << 4);
    device const uint * q1 = (device const uint *) (b + 8 + ib * 16);
    const uint h = (p * F4) / 4;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) {
        const uint at = (p * F4) % 4 + j;
        const uchar4 q = as_type<uchar4>((q1[at] >> (4 * h)) & 0x0F0F0F0Fu);
        w.f[j] = float4(kvalues_iq4nl[q.x], kvalues_iq4nl[q.y],
                        kvalues_iq4nl[q.z], kvalues_iq4nl[q.w]);
    }
    w.a = d * (float) (ls - 32);
    w.b = 0.0f;
}

// Q8_0 and IQ4_NL are 32-weight blocks; eight of them make the
// 256-weight span these kernels walk, so sub-block ib is block ib and
// `blk` is eight block strides. A block is 2-byte aligned, so its codes
// come in as packed shorts.
__attribute__((always_inline))
inline uint2 kq_bytes8(device const uchar *p) {
    return as_type<uint2>(ushort4(*(device const packed_ushort4 *) p));
}

__attribute__((always_inline))
inline float4 kq_iq4nl(uint nib) {
    const uchar4 q = as_type<uchar4>(nib);
    return float4(kvalues_iq4nl[q.x], kvalues_iq4nl[q.y],
                  kvalues_iq4nl[q.z], kvalues_iq4nl[q.w]);
}

__attribute__((always_inline))
inline void kq_dq_iq4_nl(device const uchar *b, uint ib, thread KqSub &w) {
    device const uchar * p = b + ib * 18;
    const float d = (float) *(device const half *) p;
    const uint2 lo = kq_bytes8(p + 2), hi = kq_bytes8(p + 10);
    const uint4 wd = uint4(lo.x, lo.y, hi.x, hi.y);
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < 4; ++j) {
        w.f[j] = kq_iq4nl(wd[j] & 0x0F0F0F0Fu);
        w.f[j + 4] = kq_iq4nl((wd[j] >> 4) & 0x0F0F0F0Fu);
    }
    w.a0 = d;
    w.a1 = d;
    w.b0 = 0.0f;
    w.b1 = 0.0f;
}

__attribute__((always_inline))
inline void kq_dq_q8_0(device const uchar *b, uint ib, thread KqSub &w) {
    device const uchar * p = b + ib * 34;
    const float d = (float) *(device const half *) p;
    #pragma clang loop unroll(full)
    for (ushort k = 0; k < 4; ++k) {
        const uint2 q = kq_bytes8(p + 2 + 8 * k);
        w.f[2 * k] = float4(as_type<char4>(q.x));
        w.f[2 * k + 1] = float4(as_type<char4>(q.y));
    }
    w.a0 = d;
    w.a1 = d;
    w.b0 = 0.0f;
    w.b1 = 0.0f;
}

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_iq4_nl_slice(device const uchar *b, uint ib, uint p,
                               thread KqSlice<F4> &w) {
    device const uchar * bp = b + ib * 18;
    const float d = (float) *(device const half *) bp;
    const uint2 lo = kq_bytes8(bp + 2), hi = kq_bytes8(bp + 10);
    const uint4 wd = uint4(lo.x, lo.y, hi.x, hi.y);
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) {
        const uint e = p * F4 + j;
        w.f[j] = kq_iq4nl((wd[e % 4] >> (4 * (e / 4))) & 0x0F0F0F0Fu);
    }
    w.a = d;
    w.b = 0.0f;
}

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_q8_0_slice(device const uchar *b, uint ib, uint p,
                             thread KqSlice<F4> &w) {
    device const uchar * bp = b + ib * 34;
    const float d = (float) *(device const half *) bp;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) {
        const uint e = p * F4 + j;
        const ushort2 q = *(device const packed_ushort2 *) (bp + 2 + 4 * e);
        w.f[j] = float4(as_type<char4>(as_type<uint>(q)));
    }
    w.a = d;
    w.b = 0.0f;
}

// Whether a decoder's span can end before a full 256: the packed blocks
// serve rows of any multiple of 32, the super-block types never do.
template <typename D> struct kq_tail {
    static constant constexpr bool value = false;
};

struct DqQ4K {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_q4k_slice<F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q4k(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_q4k(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqQ5K {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_q5k_slice<F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q5k(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_q5k(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqQ6K {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_q6k_slice<F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q6k(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_q6k(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqQ2K {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q2k(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqQ2K>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqQ3K {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqQ3K, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q3k(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqQ3K>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq2xxs {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq2xxs, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq2xxs(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq2xxs>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq2xs {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq2xs, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq2xs(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq2xs>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq2s {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq2s, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq2s(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq2s>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq3xxs {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq3xxs, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq3xxs(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq3xxs>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq3s {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq3s, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq3s(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq3s>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq1s {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq1s, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq1s(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq1s>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq1m {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq1m, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq1m(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq1m>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq4xs {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_iq4xs_slice<F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq4xs(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_iq4xs(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};

struct DqQ80 {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_q8_0_slice<F4>(b, ib, p, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_q8_0(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq4nl {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_iq4_nl_slice<F4>(b, ib, p, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_iq4_nl(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
template <> struct kq_tail<DqQ80> {
    static constant constexpr bool value = true;
};
template <> struct kq_tail<DqIq4nl> {
    static constant constexpr bool value = true;
};

// `blk` is the byte count of a 256-weight span, handed down from
// GGUF.rowByteCount so the stride has one home.
struct IQArgs { ulong woff; uint K; uint M; uint ty; uint blk; };
struct IQEmbedArgs {
    ulong woff; uint K; uint M; uint rowBytes; uint ty; uint blk;
};

template <typename D>
void iq_dq_row_impl(device const uchar * weights, device float * out,
                    constant IQArgs & a, uint gid) {
    if (gid < a.K / 256) {
        device const uchar * bp = weights + a.woff + (ulong) gid * a.blk;
        device float * o = out + (ulong) gid * 256;
        float w[32];
        for (uint sb = 0; sb < 8; ++sb) {
            D::dq(bp, sb, w);
            for (uint j = 0; j < 32; ++j) { o[sb * 32 + j] = w[j]; }
        }
    }
}

#define IQ_DQROW_KERNEL(NAME, D)                                   \
kernel void NAME(                                                  \
        device const uchar * weights [[buffer(0)]],                \
        device       float * out     [[buffer(1)]],                \
        constant IQArgs    & a       [[buffer(2)]],                \
        uint gid [[thread_position_in_grid]]) {                    \
    iq_dq_row_impl<D>(weights, out, a, gid);                       \
}

IQ_DQROW_KERNEL(q2_k_dequant_row, DqQ2K)
IQ_DQROW_KERNEL(q4_k_dequant_row, DqQ4K)
IQ_DQROW_KERNEL(q6_k_dequant_row, DqQ6K)

template <typename D>
void iq_embed_impl(device const uchar * weights, device const int * ids,
                   device float * out, constant IQEmbedArgs & a, uint gid) {
    const uint nblk = a.K / 256;
    if (gid < a.M * nblk) {
        const uint n = gid / nblk, ib = gid % nblk;
        device const uchar * bp = weights + a.woff
            + (ulong) ids[n] * a.rowBytes + (ulong) ib * a.blk;
        device float * o = out + (ulong) n * a.K + ib * 256;
        float w[32];
        for (uint sb = 0; sb < 8; ++sb) {
            D::dq(bp, sb, w);
            for (uint j = 0; j < 32; ++j) { o[sb * 32 + j] = w[j]; }
        }
    }
}

#define IQ_EMBED_KERNEL(NAME, D)                                   \
kernel void NAME(                                                  \
        device const uchar   * weights [[buffer(0)]],              \
        device const int     * ids     [[buffer(1)]],              \
        device       float   * out     [[buffer(2)]],              \
        constant IQEmbedArgs & a       [[buffer(3)]],              \
        uint gid [[thread_position_in_grid]]) {                    \
    iq_embed_impl<D>(weights, ids, out, a, gid);                   \
}

IQ_EMBED_KERNEL(q2_k_embed_batch, DqQ2K)
IQ_EMBED_KERNEL(q4_k_embed_batch, DqQ4K)
IQ_EMBED_KERNEL(q6_k_embed_batch, DqQ6K)

kernel void iq_dequant_row(
        device const uchar * weights [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        constant IQArgs    & a       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.K / 256) {
        device const uchar * bp = weights + a.woff + (ulong) gid * a.blk;
        device float * o = out + (ulong) gid * 256;
        float w[32];
        for (uint sb = 0; sb < 8; ++sb) {
            dq_sub(a.ty, bp, sb, w);
            for (uint j = 0; j < 32; ++j) { o[sb * 32 + j] = w[j]; }
        }
    }
}

kernel void iq_embed_batch(
        device const uchar   * weights [[buffer(0)]],
        device const int     * ids     [[buffer(1)]],
        device       float   * out     [[buffer(2)]],
        constant IQEmbedArgs & a       [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint nblk = a.K / 256;
    if (gid < a.M * nblk) {
        const uint n = gid / nblk, ib = gid % nblk;
        device const uchar * bp = weights + a.woff
            + (ulong) ids[n] * a.rowBytes + (ulong) ib * a.blk;
        device float * o = out + (ulong) n * a.K + ib * 256;
        float w[32];
        for (uint sb = 0; sb < 8; ++sb) {
            dq_sub(a.ty, bp, sb, w);
            for (uint j = 0; j < 32; ++j) { o[sb * 32 + j] = w[j]; }
        }
    }
}

// The prefill tile, the same 64(M) x 32(N) x 32(K) shape gemm_mm_impl uses,
// with the block walk made runtime: `blk` is the stride and `ty` picks the
// decoder, where the macro's Block is a compile-time type. A thread stages
// the 16 weights at `il*16` of its row, so it decodes the 32-weight
// sub-block `il / 2` and keeps half. [iq-prefill-tile]
template <typename Reg>
inline void iq_gemm_impl(
        device const uchar * weights,
        device const float * X,
        device       float * dst,
        constant IQArgs    & a,
        constant uint      & N,
        threadgroup uchar  * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort sgitg) {
    const int K = (int) a.K, M = (int) a.M;
    const int NR0 = 64, NR1 = 32, NK = 32, NL0 = NK / 16, NL1 = NK / 8;
    const ulong sbOff = sizeof(Reg) == 2 ? 4096 : 8192;
    threadgroup Reg * sa = (threadgroup Reg *) (shmem);
    threadgroup Reg * sb = (threadgroup Reg *) (shmem + sbOff);
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
    const ulong rowBytes = (ulong) (K / 256) * a.blk;
    device const uchar * row = weights + a.woff + rowBytes * (r0 + lr0);
    uint blkIdx = 0;
    const short iy = 8 * (tiitg % NL1);
    device const float * y = X + (ulong) (r1 + lr1) * K + iy;
    simdgroup_float8x8 mc[8];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    float w[32];
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        matrix<Reg, 4, 4> temp_a;
        dq_sub(a.ty, row + (ulong) blkIdx * a.blk, il / 2, w);
        const short half0 = (il % 2) * 16;
        for (short i = 0; i < 16; i++) {
            temp_a[i / 4][i % 4] = (Reg) w[half0 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short slot = 8 * sx + sy;
            sa[64 * slot + 8 * ly + lx] = temp_a[i / 4][i % 4];
        }
        {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short slot = 4 * sx + sy;
            threadgroup Reg * bp = sb + 64 * slot + 8 * ly;
            for (short i = 0; i < 8; i++) { bp[i] = (Reg) y[i]; }
        }
        il = (il + 2 < 16) ? il + 2 : il % 2;
        blkIdx = (il < 2) ? blkIdx + 1 : blkIdx;
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simd_mm_slice(sa, sb, mc, sgitg);
    }
    store_mm_tile(mc, dst, shmem, r0, r1, M, (int) N, nr0, nr1,
                  tiitg, sgitg);
}

kernel void iq_gemm_mm_h(
        device const uchar * weights [[buffer(0)]],
        device const float * X       [[buffer(1)]],
        device       float * dst     [[buffer(2)]],
        constant IQArgs    & a       [[buffer(3)]],
        constant uint      & N       [[buffer(4)]],
        threadgroup uchar  * shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    iq_gemm_impl<half>(weights, X, dst, a, N, shmem, tgpig, tiitg, sgitg);
}

// The K-quants get iq1_s_gemv's SHAPE without its ternary shortcut: eight
// threads cooperate on one 256-weight block, activations staged once per
// sub-block and reused across all NR0 rows, and ONE decoder inlined by the
// template rather than thirteen behind a runtime switch. [kquant-gemv]
// Lanes: 8/SPAN sub-block owners per block, two block stripes, and the rest
// of the 32 as row groups of 8/G rows each, so a K=2560 row's ten blocks
// divide into five passes with no tail and the reduction stays inside the
// contiguous lanes of one row group. [kquant-gemv]
template <typename D>
void kq_gemv_impl(
        device const uchar * weights,
        device const float * x,
        device       float * out,
        constant IQArgs    & a,
        uint3  tgpig,
        ushort tiisg) {
    const ushort SPAN = D::span;
    const ushort L = 8 / SPAN;
    const ushort LANES = L * 2;
    const ushort G = 32 / LANES;
    const ushort RPG = 8 / G;
    const ushort owner = tiisg % L;
    const ushort bg = (tiisg / L) % 2;
    const ushort rg = tiisg / LANES;
    const ushort sub = owner * SPAN;
    const uint nblk = (a.K + 255) / 256;
    const ulong rowBytes = (ulong) a.K * a.blk / 256;
    device const uchar * W = weights + a.woff;
    const uint r0 = tgpig.x * 8 + rg * RPG;
    float acc[4] = { 0, 0, 0, 0 };
    for (uint ib = bg; ib < nblk; ib += 2) {
        const bool live = !kq_tail<D>::value || ib * 256 + sub * 32 < a.K;
        device const float4 * y = (device const float4 *)
            (x + (ulong) ib * 256 + sub * 32);
        float4 yl[8 * SPAN];
        float sy16[2 * SPAN];
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < 8 * SPAN; ++i) { yl[i] = live ? y[i] : 0.0f; }
        #pragma clang loop unroll(full)
        for (ushort h = 0; h < 2 * SPAN; ++h) {
            sy16[h] = hsum(yl[h * 4] + yl[h * 4 + 1]
                           + yl[h * 4 + 2] + yl[h * 4 + 3]);
        }
        #pragma clang loop unroll(full)
        for (ushort rr = 0; rr < RPG; ++rr) {
            const uint row = r0 + rr;
            if (row < a.M && live) {
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * a.blk;
                float s;
                D::template dot<1, SPAN, 8>(bp, sub, yl, sy16, &s);
                acc[rr] += s;
            }
        }
    }
    #pragma clang loop unroll(full)
    for (ushort rr = 0; rr < RPG; ++rr) {
        float v = acc[rr];
        #pragma clang loop unroll(full)
        for (ushort sh = 1; sh < LANES; sh <<= 1) {
            v += simd_shuffle_xor(v, sh);
        }
        if (tiisg % LANES == 0 && r0 + rr < a.M) { out[r0 + rr] = v; }
    }
}


#define KQ_GEMV_KERNEL(NAME, D)                                    \
kernel void NAME(                                                  \
        device const uchar * weights [[buffer(0)]],                \
        device const float * x       [[buffer(1)]],                \
        device       float * out     [[buffer(2)]],                \
        constant IQArgs    & a       [[buffer(3)]],                \
        uint3  tgpig [[threadgroup_position_in_grid]],             \
        ushort tiisg [[thread_index_in_simdgroup]]) {              \
    kq_gemv_impl<D>(weights, x, out, a, tgpig, tiisg);             \
}


KQ_GEMV_KERNEL(q4_k_gemv, DqQ4K)
KQ_GEMV_KERNEL(q5_k_gemv, DqQ5K)
KQ_GEMV_KERNEL(q6_k_gemv, DqQ6K)
KQ_GEMV_KERNEL(q2_k_gemv, DqQ2K)
KQ_GEMV_KERNEL(q3_k_gemv, DqQ3K)
KQ_GEMV_KERNEL(iq2_xxs_gemv, DqIq2xxs)
KQ_GEMV_KERNEL(iq2_xs_gemv, DqIq2xs)
KQ_GEMV_KERNEL(iq2_s_gemv, DqIq2s)
KQ_GEMV_KERNEL(iq3_xxs_gemv, DqIq3xxs)
KQ_GEMV_KERNEL(iq3_s_gemv, DqIq3s)
KQ_GEMV_KERNEL(iq4_xs_gemv, DqIq4xs)
KQ_GEMV_KERNEL(q8_0_gemv, DqQ80)
KQ_GEMV_KERNEL(iq4_nl_gemv, DqIq4nl)

// The narrow-batch twin of kq_gemv: the same block walk, each weight
// sub-block decoded ONCE and dotted against R1 staged activation columns,
// so a verify pass streams the trunk once instead of once per column.
// [kquant-gemv]
template <typename D, ushort R1, ushort SG>
void kq_gemm_nb_impl(
        device const uchar * weights,
        device const float * X,
        device       float * out,
        constant IQArgs    & a,
        threadgroup float4 * ys,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    const ushort LANES = 16;
    const ushort RPG = 4;
    const ushort CS = 128;
    const ushort F4 = R1 >= 4 ? 2 : 4;
    const ushort sub = tiisg % 8;
    const ushort bg = (tiisg / 8) % 2;
    const ushort rg = tiisg / LANES;
    const uint nblk = (a.K + 255) / 256;
    const uint k4 = a.K / 4;
    const ulong rowBytes = (ulong) a.K * a.blk / 256;
    device const uchar * W = weights + a.woff;
    device const float4 * X4 = (device const float4 *) X;
    const uint r0 = tgpig.x * (8 * SG) + sgitg * 8 + rg * RPG;
    float acc[RPG * R1];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < RPG * R1; ++i) { acc[i] = 0.0f; }
    for (uint ib0 = 0; ib0 < nblk; ib0 += 2) {
        for (ushort e = tiitg; e < R1 * CS; e += 32 * SG) {
            const ushort c = e / CS, i = e % CS;
            const uint at = ib0 * 64 + i;
            if (at < k4) { ys[e] = X4[(ulong) c * k4 + at]; }
        }
        if (SG == 1) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint ib = ib0 + bg;
        const bool live = kq_tail<D>::value ? ib * 256 + sub * 32 < a.K
                                            : ib < nblk;
        threadgroup const float4 * yl = ys + bg * 64 + sub * 8;
        #pragma clang loop unroll(full)
        for (uint p = 0; p < 8 / F4; ++p) {
            float4 yp[R1 * F4];
            float sy[R1];
            #pragma clang loop unroll(full)
            for (ushort c = 0; c < R1; ++c) {
                float4 t = 0.0f;
                #pragma clang loop unroll(full)
                for (ushort j = 0; j < F4; ++j) {
                    yp[c * F4 + j] = yl[c * CS + p * F4 + j];
                    t += yp[c * F4 + j];
                }
                sy[c] = hsum(t);
            }
            #pragma clang loop unroll(full)
            for (ushort rr = 0; rr < RPG; ++rr) {
                const uint row = r0 + rr;
                if (row < a.M && live) {
                    device const uchar * bp = W + (ulong) row * rowBytes
                                                + (ulong) ib * a.blk;
                    KqSlice<F4> w;
                    D::template dqSlice<F4>(bp, sub, p, w);
                    #pragma clang loop unroll(full)
                    for (ushort c = 0; c < R1; ++c) {
                        float4 t = w.f[0] * yp[c * F4];
                        #pragma clang loop unroll(full)
                        for (ushort j = 1; j < F4; ++j) {
                            t += w.f[j] * yp[c * F4 + j];
                        }
                        acc[rr * R1 + c] += w.a * hsum(t) + w.b * sy[c];
                    }
                }
            }
        }
        if (SG == 1) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < RPG * R1; ++i) {
        float v = acc[i];
        #pragma clang loop unroll(full)
        for (ushort sh = 1; sh < LANES; sh <<= 1) {
            v += simd_shuffle_xor(v, sh);
        }
        const uint row = r0 + i / R1;
        if (tiisg % LANES == 0 && row < a.M) {
            out[(ulong) (i % R1) * a.M + row] = v;
        }
    }
}

// The narrow kernel in the old Q4_0 kernel's shape: no threadgroup
// staging, sixteen rows a simdgroup, a lane's activations read from device
// memory per quarter slice, and the row bound a select on the sum rather
// than a branch around the pair exchange.
template <typename D, ushort R1>
void kq_gemm_nw_impl(
        device const uchar * weights,
        device const float * X,
        device       float * out,
        constant IQArgs    & a,
        uint3  tgpig,
        ushort tiisg) {
    const ushort F4 = 2;
    const ushort RPG = 4;
    const ushort sub = tiisg % 8;
    const ushort rg = tiisg / 8;
    const uint nblk = (a.K + 255) / 256;
    const uint k4 = a.K / 4;
    const ulong rowBytes = (ulong) a.K * a.blk / 256;
    device const uchar * W = weights + a.woff;
    device const float4 * X4 = (device const float4 *) X;
    const uint r0 = tgpig.x * 16 + rg * RPG;
    float acc[RPG * R1];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < RPG * R1; ++i) { acc[i] = 0.0f; }
    for (uint ib = 0; ib < nblk; ++ib) {
        const bool live = kq_tail<D>::value ? ib * 256 + sub * 32 < a.K
                                            : true;
        const uint at = ib * 64 + sub * 8;
        #pragma clang loop unroll(full)
        for (uint p = 0; p < 8 / F4; ++p) {
            float4 yp[R1 * F4];
            float sy[R1];
            #pragma clang loop unroll(full)
            for (ushort c = 0; c < R1; ++c) {
                float4 t = 0.0f;
                #pragma clang loop unroll(full)
                for (ushort j = 0; j < F4; ++j) {
                    const uint e = at + p * F4 + j;
                    yp[c * F4 + j] = live ? X4[(ulong) c * k4 + e] : 0.0f;
                    t += yp[c * F4 + j];
                }
                sy[c] = hsum(t);
            }
            #pragma clang loop unroll(full)
            for (ushort rr = 0; rr < RPG; ++rr) {
                const uint row = min(r0 + rr, a.M - 1);
                const float keep = (r0 + rr < a.M && live) ? 1.0f : 0.0f;
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * a.blk;
                KqSlice<F4> w;
                D::template dqSlice<F4>(bp, sub, p, w);
                #pragma clang loop unroll(full)
                for (ushort c = 0; c < R1; ++c) {
                    float4 t = w.f[0] * yp[c * F4];
                    #pragma clang loop unroll(full)
                    for (ushort j = 1; j < F4; ++j) {
                        t += w.f[j] * yp[c * F4 + j];
                    }
                    acc[rr * R1 + c] += keep * (w.a * hsum(t) + w.b * sy[c]);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < RPG * R1; ++i) {
        float v = acc[i];
        v += simd_shuffle_down(v, 4);
        v += simd_shuffle_down(v, 2);
        v += simd_shuffle_down(v, 1);
        const uint row = r0 + i / R1;
        if (sub == 0 && row < a.M) {
            out[(ulong) (i % R1) * a.M + row] = v;
        }
    }
}

#define KQ_NW_KERNEL(NAME, D, R1)                                  \
kernel void NAME(                                                  \
        device const uchar * weights [[buffer(0)]],                \
        device const float * X       [[buffer(1)]],                \
        device       float * out     [[buffer(2)]],                \
        constant IQArgs    & a       [[buffer(3)]],                \
        uint3  tgpig [[threadgroup_position_in_grid]],             \
        ushort tiisg [[thread_index_in_simdgroup]]) {              \
    kq_gemm_nw_impl<D, R1>(weights, X, out, a, tgpig, tiisg);      \
}

KQ_NW_KERNEL(q4_k_gemm_nw_r3, DqQ4K, 3)
KQ_NW_KERNEL(q5_k_gemm_nw_r3, DqQ5K, 3)
KQ_NW_KERNEL(q6_k_gemm_nw_r3, DqQ6K, 3)
KQ_NW_KERNEL(iq4_xs_gemm_nw_r3, DqIq4xs, 3)

#define KQ_NB_KERNEL(NAME, D, R1, SG)                              \
kernel void NAME(                                                  \
        device const uchar * weights [[buffer(0)]],                \
        device const float * X       [[buffer(1)]],                \
        device       float * out     [[buffer(2)]],                \
        constant IQArgs    & a       [[buffer(3)]],                \
        threadgroup float4 * ys      [[threadgroup(0)]],           \
        uint3  tgpig [[threadgroup_position_in_grid]],             \
        ushort tiitg [[thread_index_in_threadgroup]],              \
        ushort tiisg [[thread_index_in_simdgroup]],                \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {         \
    kq_gemm_nb_impl<D, R1, SG>(weights, X, out, a, ys, tgpig,      \
                               tiitg, tiisg, sgitg);               \
}

#define KQ_NB_KERNELS(T, D)                                        \
KQ_NB_KERNEL(T##_gemm_nb_r2s1, D, 2, 1)                            \
KQ_NB_KERNEL(T##_gemm_nb_r2s8, D, 2, 8)                            \
KQ_NB_KERNEL(T##_gemm_nb_r3s1, D, 3, 1)                            \
KQ_NB_KERNEL(T##_gemm_nb_r3s8, D, 3, 8)                            \
KQ_NB_KERNEL(T##_gemm_nb_r4s8, D, 4, 8)                            \
KQ_NB_KERNEL(T##_gemm_nb_r5s8, D, 5, 8)

KQ_NB_KERNELS(q4_k, DqQ4K)
KQ_NB_KERNELS(q5_k, DqQ5K)
KQ_NB_KERNELS(q6_k, DqQ6K)
KQ_NB_KERNELS(iq4_xs, DqIq4xs)
KQ_NB_KERNELS(q8_0, DqQ80)
KQ_NB_KERNELS(iq4_nl, DqIq4nl)
KQ_NB_KERNELS(q3_k, DqQ3K)
KQ_NB_KERNELS(iq2_s, DqIq2s)
KQ_NB_KERNELS(iq3_xxs, DqIq3xxs)
KQ_NB_KERNELS(iq3_s, DqIq3s)
KQ_NB_KERNELS(iq2_xxs, DqIq2xxs)
KQ_NB_KERNELS(iq2_xs, DqIq2xs)
KQ_NB_KERNELS(iq1_s, DqIq1s)
KQ_NB_KERNELS(iq1_m, DqIq1m)

// iq1_s and iq1_m get their own gemv, outside dq_sub's thirteen-way switch,
// in q2_0_gemv's shape: EIGHT threads cooperate on one 256-weight block (one
// 32-weight sub-block each) with four blocks in flight, activations staged in
// registers, and the ternary grid ({0,1,2} nibbles) collapsed to one multiply
// per sub-block. [iq1-dedicated-gemv]
kernel void iq1_s_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant IQArgs    & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const ushort TPB = 8, STEP = 32 / TPB;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 256;
    const ulong rowBytes = (ulong) nblk * 50;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort sub = tiisg % TPB;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint ib = grp; ib < nblk; ib += STEP) {
        device const float * y = x + (ulong) ib * 256 + sub * 32;
        float yl[32];
        float sy = 0.0f;
        for (ushort i = 0; i < 32; i++) { yl[i] = y[i]; sy += y[i]; }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * 50;
                const float d = iq_f16(bp);
                const ushort h = iq_u16(bp + 34 + sub * 2);
                const float dl = d * (float) (2 * ((h >> 12) & 7) + 1);
                const float delta = (h & 0x8000) ? -0.125f : 0.125f;
                float lo = 0.0f, hi = 0.0f;
                for (ushort l = 0; l < 4; l++) {
                    const uint e = iq1s_grid_gpu[bp[2 + sub * 4 + l]
                        | (((h >> (3 * l)) & 7) << 8)];
                    for (ushort j = 0; j < 8; j++) {
                        const uint n = (e >> (8 * (j % 4) + 4 * (j / 4))) & 0xF;
                        const float v = yl[l * 8 + j];
                        if (n == 1) { lo += v; }
                        if (n == 2) { hi += v; }
                    }
                }
                acc[r] += dl * (lo + 2.0f * hi + (delta - 1.0f) * sy);
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

kernel void iq1_m_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant IQArgs    & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const ushort TPB = 8, STEP = 32 / TPB;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 256;
    const ulong rowBytes = (ulong) nblk * 56;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort sub = tiisg % TPB;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint ib = grp; ib < nblk; ib += STEP) {
        device const float * y = x + (ulong) ib * 256 + sub * 32;
        float yl[32];
        for (ushort i = 0; i < 32; i++) { yl[i] = y[i]; }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * 56;
                ushort sc[4];
                for (ushort i = 0; i < 4; i++) { sc[i] = iq_u16(bp + 48 + i * 2); }
                const ushort bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0)
                                  | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
                const float d = (float) as_type<half>(bits);
                const ushort sv = sc[sub / 2];
                const ushort shf = 6 * (sub % 2);
                const float dl1 = d * (float) (2 * ((sv >> shf) & 7) + 1);
                const float dl2 = d * (float) (2 * ((sv >> (shf + 3)) & 7) + 1);
                const uchar h0 = bp[32 + sub * 2], h1 = bp[33 + sub * 2];
                float s = 0.0f;
                for (ushort l = 0; l < 4; l++) {
                    const uchar h = (l < 2) ? h0 : h1;
                    const uint up = (l % 2 == 0) ? ((uint) h << 8)
                                                 : ((uint) h << 4);
                    const uint e = iq1s_grid_gpu[bp[sub * 4 + l] | (up & 0x700)];
                    const float delta = (h & ((l % 2 == 0) ? 0x08 : 0x80))
                                      ? -0.125f : 0.125f;
                    float lo = 0.0f, hi = 0.0f, sy = 0.0f;
                    for (ushort j = 0; j < 8; j++) {
                        const uint n = (e >> (8 * (j % 4) + 4 * (j / 4))) & 0xF;
                        const float v = yl[l * 8 + j];
                        sy += v;
                        if (n == 1) { lo += v; }
                        if (n == 2) { hi += v; }
                    }
                    s += ((l < 2) ? dl1 : dl2)
                       * (lo + 2.0f * hi + (delta - 1.0f) * sy);
                }
                acc[r] += s;
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

struct block_q4_K { uchar b[144]; };
struct block_q5_K { uchar b[176]; };
struct block_q6_K { uchar b[210]; };
struct block_iq4_XS { uchar b[136]; };

// Slice `il` (0..15) of a super-block is sub-block il/2, half il%2.
static inline void dq_q4_k_h(device const block_q4_K * xb, short il,
                             thread half4x4 & reg) {
    device const uchar * b = xb->b;
    const uint ib = il / 2, h = il % 2;
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    const float dv = d * sm.x, ov = dmin * sm.y;
    device const uint4 * q4 = (device const uint4 *) (b + 16 + (ib / 2) * 32);
    const uint4 v = q4[h];
    const uint sh = 4 * (ib & 1);
    const uint4 lo = (v >> sh) & 0x0F0F0F0Fu;
    reg[0] = half4(dv * nib4(lo.x) - ov);
    reg[1] = half4(dv * nib4(lo.y) - ov);
    reg[2] = half4(dv * nib4(lo.z) - ov);
    reg[3] = half4(dv * nib4(lo.w) - ov);
}

static inline void dq_q5_k_h(device const block_q5_K * xb, short il,
                             thread half4x4 & reg) {
    device const uchar * b = xb->b;
    const uint ib = il / 2, h = il % 2;
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    const float dv = d * sm.x, ov = dmin * sm.y;
    device const uint4 * q4 = (device const uint4 *) (b + 48 + (ib / 2) * 32);
    device const uint4 * h4 = (device const uint4 *) (b + 16);
    const uint4 v = q4[h];
    const uint4 hb = h4[h];
    const uint sh = 4 * (ib & 1);
    const uint4 n = ((v >> sh) & 0x0F0F0F0Fu)
                  | (((hb >> ib) & 0x01010101u) << 4);
    reg[0] = half4(dv * nib4(n.x) - ov);
    reg[1] = half4(dv * nib4(n.y) - ov);
    reg[2] = half4(dv * nib4(n.z) - ov);
    reg[3] = half4(dv * nib4(n.w) - ov);
}

static inline void dq_q6_k_h(device const block_q6_K * xb, short il,
                             thread half4x4 & reg) {
    device const uchar * b = xb->b;
    const uint ib = il / 2, h = il % 2;
    const float d = iq_f16(b + 208);
    const uint n = ib / 4, r = ib % 4;
    device const packed_ushort4 * ql = (device const packed_ushort4 *)
        (b + n * 64 + (r % 2) * 32);
    device const packed_ushort4 * qh = (device const packed_ushort4 *)
        (b + 128 + n * 32);
    const float s = d * (float) (int) (char) b[192 + n * 8 + 2 * r + h];
    const uint shq = (r < 2) ? 0 : 4;
    const uint shh = 2 * r;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        const uint2 lw = as_type<uint2>(ushort4(ql[2 * h + i]));
        const uint2 hw = as_type<uint2>(ushort4(qh[2 * h + i]));
        const uint q0 = ((lw.x >> shq) & 0x0F0F0F0Fu)
                      | (((hw.x >> shh) & 0x03030303u) << 4);
        const uint q1 = ((lw.y >> shq) & 0x0F0F0F0Fu)
                      | (((hw.y >> shh) & 0x03030303u) << 4);
        reg[2 * i] = half4(s * (nib4(q0) - 32.0f));
        reg[2 * i + 1] = half4(s * (nib4(q1) - 32.0f));
    }
}

static inline void dq_iq4_xs_h(device const block_iq4_XS * xb, short il,
                               thread half4x4 & reg) {
    device const uchar * b = xb->b;
    const uint ib = il / 2, h = il % 2;
    const float d = iq_f16(b);
    const ushort sh = iq_u16(b + 2);
    const uchar sl = b[4 + ib / 2];
    const int ls = (int) ((sl >> (4 * (ib % 2))) & 0xF)
                 | (int) (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * (float) (ls - 32);
    device const uint * q1 = (device const uint *) (b + 8 + ib * 16);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        const uchar4 q = as_type<uchar4>((q1[i] >> (4 * h)) & 0x0F0F0F0Fu);
        reg[i] = half4(dl * float4(kvalues_iq4nl[q.x], kvalues_iq4nl[q.y],
                                   kvalues_iq4nl[q.z], kvalues_iq4nl[q.w]));
    }
}

GEMM_MM_KERNEL(q4_k_gemm_mm_h, block_q4_K, half, 16, 256, dq_q4_k_h)
GEMM_MM_KERNEL(q5_k_gemm_mm_h, block_q5_K, half, 16, 256, dq_q5_k_h)
GEMM_MM_KERNEL(q6_k_gemm_mm_h, block_q6_K, half, 16, 256, dq_q6_k_h)
GEMM_MM_KERNEL(iq4_xs_gemm_mm_h, block_iq4_XS, half, 16, 256, dq_iq4_xs_h)

// The codebook types decode a whole sub-block and keep the staged half.
template <typename D, typename Block>
static inline void dq_generic_h(device const Block * xb, short il,
                                thread half4x4 & reg) {
    float v[32];
    D::dq(xb->b, il / 2, v);
    const short at = (il % 2) * 16;
    for (int i = 0; i < 4; i++) {
        reg[i] = half4(v[at + i * 4], v[at + i * 4 + 1], v[at + i * 4 + 2],
                       v[at + i * 4 + 3]);
    }
}

struct block_q3_K { uchar b[110]; };
struct block_iq2_S { uchar b[82]; };
struct block_iq3_XXS { uchar b[98]; };
struct block_iq3_S { uchar b[110]; };
struct block_iq2_XXS { uchar b[66]; };
struct block_iq2_XS { uchar b[74]; };
struct block_iq1_S { uchar b[50]; };
struct block_iq1_M { uchar b[56]; };

static inline void dq_q3_k_h(device const block_q3_K * xb, short il,
                             thread half4x4 & reg) {
    dq_generic_h<DqQ3K, block_q3_K>(xb, il, reg);
}
static inline void dq_iq2_s_h(device const block_iq2_S * xb, short il,
                              thread half4x4 & reg) {
    dq_generic_h<DqIq2s, block_iq2_S>(xb, il, reg);
}
static inline void dq_iq3_xxs_h(device const block_iq3_XXS * xb, short il,
                                thread half4x4 & reg) {
    dq_generic_h<DqIq3xxs, block_iq3_XXS>(xb, il, reg);
}
static inline void dq_iq3_s_h(device const block_iq3_S * xb, short il,
                              thread half4x4 & reg) {
    dq_generic_h<DqIq3s, block_iq3_S>(xb, il, reg);
}
static inline void dq_iq2_xxs_h(device const block_iq2_XXS * xb, short il,
                                thread half4x4 & reg) {
    dq_generic_h<DqIq2xxs, block_iq2_XXS>(xb, il, reg);
}
static inline void dq_iq2_xs_h(device const block_iq2_XS * xb, short il,
                               thread half4x4 & reg) {
    dq_generic_h<DqIq2xs, block_iq2_XS>(xb, il, reg);
}
static inline void dq_iq1_s_h(device const block_iq1_S * xb, short il,
                              thread half4x4 & reg) {
    dq_generic_h<DqIq1s, block_iq1_S>(xb, il, reg);
}
static inline void dq_iq1_m_h(device const block_iq1_M * xb, short il,
                              thread half4x4 & reg) {
    dq_generic_h<DqIq1m, block_iq1_M>(xb, il, reg);
}

GEMM_MM_KERNEL(q3_k_gemm_mm_h, block_q3_K, half, 16, 256, dq_q3_k_h)
GEMM_MM_KERNEL(iq2_s_gemm_mm_h, block_iq2_S, half, 16, 256, dq_iq2_s_h)
GEMM_MM_KERNEL(iq3_xxs_gemm_mm_h, block_iq3_XXS, half, 16, 256, dq_iq3_xxs_h)
GEMM_MM_KERNEL(iq3_s_gemm_mm_h, block_iq3_S, half, 16, 256, dq_iq3_s_h)
GEMM_MM_KERNEL(iq2_xxs_gemm_mm_h, block_iq2_XXS, half, 16, 256, dq_iq2_xxs_h)
GEMM_MM_KERNEL(iq2_xs_gemm_mm_h, block_iq2_XS, half, 16, 256, dq_iq2_xs_h)
GEMM_MM_KERNEL(iq1_s_gemm_mm_h, block_iq1_S, half, 16, 256, dq_iq1_s_h)
GEMM_MM_KERNEL(iq1_m_gemm_mm_h, block_iq1_M, half, 16, 256, dq_iq1_m_h)

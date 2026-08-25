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

inline void dq_sub_iq4xs(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const ushort sh = iq_u16(b + 2);
    const uchar sl = b[4 + ib / 2];
    const int ls = (int) ((sl >> (4 * (ib % 2))) & 0xF)
                 | (int) (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * (float) (ls - 32);
    for (uint j = 0; j < 16; ++j) {
        const uchar q = b[8 + ib * 16 + j];
        w[j] = dl * (float) kvalues_iq4nl[q & 0xF];
        w[j + 16] = dl * (float) kvalues_iq4nl[q >> 4];
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

inline void dq_sub_q4k(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b), dmin = iq_f16(b + 2);
    const float2 sm = dq_scale_min(b, ib);
    const float dv = d * sm.x, ov = dmin * sm.y;
    const uint qOff = 16 + (ib / 2) * 32;
    const bool high = (ib % 2) != 0;
    for (uint l = 0; l < 32; ++l) {
        const uchar q = b[qOff + l];
        w[l] = dv * (float) (high ? (q >> 4) : (q & 0xF)) - ov;
    }
}

inline void dq_sub_q5k(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b), dmin = iq_f16(b + 2);
    const float2 sm = dq_scale_min(b, ib);
    const float dv = d * sm.x, ov = dmin * sm.y;
    const uint qOff = 48 + (ib / 2) * 32;
    const uchar u = (uchar) (1u << ib);
    const bool high = (ib % 2) != 0;
    for (uint l = 0; l < 32; ++l) {
        const uchar q = b[qOff + l];
        const float lo = (float) (high ? (q >> 4) : (q & 0xF));
        w[l] = dv * (lo + ((b[16 + l] & u) ? 16.0f : 0.0f)) - ov;
    }
}

inline void dq_sub_q6k(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b + 208);
    const uint n = ib / 4, r = ib % 4;
    const uint qlOff = n * 64 + (r % 2) * 32;
    const uint qhOff = 128 + n * 32;
    const uint scOff = 192 + n * 8 + 2 * r;
    for (uint l = 0; l < 32; ++l) {
        const uchar byte = b[qlOff + l];
        const uint nib = (r < 2) ? (byte & 0xF) : (byte >> 4);
        const int q = (int) (nib | (((b[qhOff + l] >> (2 * r)) & 3) << 4));
        const int s = (int) (char) b[scOff + l / 16];
        w[l] = d * (float) s * (float) (q - 32);
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

struct DqQ4K {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q4k(b, ib, w);
    }
};
struct DqQ5K {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q5k(b, ib, w);
    }
};
struct DqQ6K {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q6k(b, ib, w);
    }
};
struct DqQ2K {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q2k(b, ib, w);
    }
};
struct DqQ3K {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q3k(b, ib, w);
    }
};
struct DqIq2xxs {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq2xxs(b, ib, w);
    }
};
struct DqIq2xs {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq2xs(b, ib, w);
    }
};
struct DqIq2s {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq2s(b, ib, w);
    }
};
struct DqIq3xxs {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq3xxs(b, ib, w);
    }
};
struct DqIq3s {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq3s(b, ib, w);
    }
};
struct DqIq4xs {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq4xs(b, ib, w);
    }
};

// `blk` is the super-block byte count, handed down from GGUF.rowByteCount so
// the stride has one home. K is a multiple of 256 for every type here.
struct IQArgs { ulong woff; uint K; uint M; uint ty; uint blk; };
struct IQEmbedArgs {
    ulong woff; uint K; uint M; uint rowBytes; uint ty; uint blk;
};

kernel void iq_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant IQArgs    & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 256;
    const ulong rowBytes = (ulong) nblk * a.blk;
    device const uchar * W = weights + a.woff;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float w[32];
    for (uint ib = tiisg; ib < nblk; ib += 32) {
        device const float * y = x + (ulong) ib * 256;
        for (uint r = 0; r < NR0; ++r) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * a.blk;
                float s = 0.0f;
                for (uint sb = 0; sb < 8; ++sb) {
                    dq_sub(a.ty, bp, sb, w);
                    device const float * yy = y + sb * 32;
                    for (uint j = 0; j < 32; ++j) { s += w[j] * yy[j]; }
                }
                acc[r] += s;
            }
        }
    }
    for (uint r = 0; r < NR0; ++r) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

// iq_gemm_nb: out[N,M] = X[N,K] @ W for a NARROW batch, the shape a verify
// pass has. A thread owns NR0 rows and R1 columns and decodes each 32-weight
// sub-block ONCE, dotting it against every column before moving on -- so the
// decode cost, which for these types is the whole cost, is paid once for R1
// columns instead of R1 times. [iq-narrow]
template<ushort R1, ushort NR0>
void iq_gemm_nb_impl(
        device const uchar * weights,
        device const float * X,
        device       float * out,
        constant IQArgs    & a,
        uint3  tgpig,
        ushort tiisg) {
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 256;
    const ulong rowBytes = (ulong) nblk * a.blk;
    device const uchar * W = weights + a.woff;
    float acc[NR0 * R1];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < NR0 * R1; ++i) { acc[i] = 0.0f; }
    float w[32];
    for (uint ib = tiisg; ib < nblk; ib += 32) {
        for (ushort r = 0; r < NR0; ++r) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * a.blk;
                float s[R1];
                #pragma clang loop unroll(full)
                for (ushort c = 0; c < R1; ++c) { s[c] = 0.0f; }
                for (uint sb = 0; sb < 8; ++sb) {
                    dq_sub(a.ty, bp, sb, w);
                    #pragma clang loop unroll(full)
                    for (ushort c = 0; c < R1; ++c) {
                        device const float * yy = X + (ulong) c * a.K
                            + (ulong) ib * 256 + sb * 32;
                        float t = 0.0f;
                        for (uint j = 0; j < 32; ++j) { t += w[j] * yy[j]; }
                        s[c] += t;
                    }
                }
                #pragma clang loop unroll(full)
                for (ushort c = 0; c < R1; ++c) { acc[r * R1 + c] += s[c]; }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < NR0 * R1; ++i) {
        const float s = simd_sum(acc[i]);
        const ushort r = i / R1, c = i % R1;
        if (tiisg == 0 && row0 + r < a.M) {
            out[(ulong) c * a.M + row0 + r] = s;
        }
    }
}

#define IQ_NB_KERNEL(NAME, R1, NR0)                                    \
kernel void NAME(                                                      \
        device const uchar * weights [[buffer(0)]],                    \
        device const float * X       [[buffer(1)]],                    \
        device       float * out     [[buffer(2)]],                    \
        constant IQArgs    & a       [[buffer(3)]],                    \
        uint3  tgpig [[threadgroup_position_in_grid]],                 \
        ushort tiisg [[thread_index_in_simdgroup]]) {                  \
    iq_gemm_nb_impl<R1, NR0>(weights, X, out, a, tgpig, tiisg);        \
}

IQ_NB_KERNEL(iq_gemm_nb_r2, 2, 8)
IQ_NB_KERNEL(iq_gemm_nb_r3, 3, 8)
IQ_NB_KERNEL(iq_gemm_nb_r4, 4, 4)
IQ_NB_KERNEL(iq_gemm_nb_r5, 5, 4)

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

kernel void iq_gemm_mm(
        device const uchar * weights [[buffer(0)]],
        device const float * X       [[buffer(1)]],
        device       float * dst     [[buffer(2)]],
        constant IQArgs    & a       [[buffer(3)]],
        constant uint      & N       [[buffer(4)]],
        threadgroup uchar  * shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    iq_gemm_impl<float>(weights, X, dst, a, N, shmem, tgpig, tiitg, sgitg);
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
template <typename D>
void kq_gemv_impl(
        device const uchar * weights,
        device const float * x,
        device       float * out,
        constant IQArgs    & a,
        uint3  tgpig,
        ushort tiisg) {
    const uint NR0 = 8;
    const ushort TPB = 8, STEP = 32 / TPB;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 256;
    const ulong rowBytes = (ulong) nblk * a.blk;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort sub = tiisg % TPB;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint ib = grp; ib < nblk; ib += STEP) {
        device const float * y = x + (ulong) ib * 256 + sub * 32;
        float yl[32];
        for (ushort i = 0; i < 32; ++i) { yl[i] = y[i]; }
        for (uint r = 0; r < NR0; ++r) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * a.blk;
                float w[32];
                D::dq(bp, sub, w);
                float s = 0.0f;
                for (ushort j = 0; j < 32; ++j) { s += w[j] * yl[j]; }
                acc[r] += s;
            }
        }
    }
    for (uint r = 0; r < NR0; ++r) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
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

// iq1_s narrow batch: the dedicated gemv's body with R1 columns carried per
// grid lookup. For these types the DECODE is the cost -- the grid indirection
// and the nibble walk -- and it does not depend on the column, so one lookup
// feeds R1 accumulations and a width-3 verify costs little more than a gemv.
// Activations are read from device rather than staged: yl[R1][32] would spill
// the registers this shape exists to keep. [iq1-narrow]
template<ushort R1, ushort NR0>
void iq1_s_gemm_nb_impl(
        device const uchar * weights,
        device const float * X,
        device       float * out,
        constant IQArgs    & a,
        uint3  tgpig,
        ushort tiisg) {
    const ushort TPB = 8, STEP = 32 / TPB;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 256;
    const ulong rowBytes = (ulong) nblk * 50;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort sub = tiisg % TPB;
    float acc[NR0 * R1];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < NR0 * R1; i++) { acc[i] = 0.0f; }
    for (uint ib = grp; ib < nblk; ib += STEP) {
        const ulong base = (ulong) ib * 256 + sub * 32;
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * 50;
                const float d = iq_f16(bp);
                const ushort h = iq_u16(bp + 34 + sub * 2);
                const float dl = d * (float) (2 * ((h >> 12) & 7) + 1);
                const float delta = (h & 0x8000) ? -0.125f : 0.125f;
                float lo[R1], hi[R1], sy[R1];
                #pragma clang loop unroll(full)
                for (ushort c = 0; c < R1; c++) {
                    lo[c] = 0.0f; hi[c] = 0.0f; sy[c] = 0.0f;
                }
                for (ushort l = 0; l < 4; l++) {
                    const uint e = iq1s_grid_gpu[bp[2 + sub * 4 + l]
                        | (((h >> (3 * l)) & 7) << 8)];
                    for (ushort j = 0; j < 8; j++) {
                        const uint n = (e >> (8 * (j % 4) + 4 * (j / 4))) & 0xF;
                        #pragma clang loop unroll(full)
                        for (ushort c = 0; c < R1; c++) {
                            const float v = X[(ulong) c * a.K + base
                                              + l * 8 + j];
                            sy[c] += v;
                            if (n == 1) { lo[c] += v; }
                            if (n == 2) { hi[c] += v; }
                        }
                    }
                }
                #pragma clang loop unroll(full)
                for (ushort c = 0; c < R1; c++) {
                    acc[r * R1 + c] += dl
                        * (lo[c] + 2.0f * hi[c] + (delta - 1.0f) * sy[c]);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < NR0 * R1; i++) {
        const float s = simd_sum(acc[i]);
        const ushort r = i / R1, c = i % R1;
        if (tiisg == 0 && row0 + r < a.M) {
            out[(ulong) c * a.M + row0 + r] = s;
        }
    }
}

#define IQ1S_NB_KERNEL(NAME, R1, NR0)                                  \
kernel void NAME(                                                      \
        device const uchar * weights [[buffer(0)]],                    \
        device const float * X       [[buffer(1)]],                    \
        device       float * out     [[buffer(2)]],                    \
        constant IQArgs    & a       [[buffer(3)]],                    \
        uint3  tgpig [[threadgroup_position_in_grid]],                 \
        ushort tiisg [[thread_index_in_simdgroup]]) {                  \
    iq1_s_gemm_nb_impl<R1, NR0>(weights, X, out, a, tgpig, tiisg);     \
}

IQ1S_NB_KERNEL(iq1_s_gemm_nb_r2, 2, 8)
IQ1S_NB_KERNEL(iq1_s_gemm_nb_r3, 3, 8)
IQ1S_NB_KERNEL(iq1_s_gemm_nb_r4, 4, 4)
IQ1S_NB_KERNEL(iq1_s_gemm_nb_r5, 5, 4)

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

// The drafting head's cheap output projection: score 2048 CLUSTERS, keep the
// best few, and score only the tokens they own. Included by Kernels.metal.
//
// The token table stays in ORIGINAL token order -- `token_ordering` is a
// permutation whose VALUES are token ids, so cluster c owns ordered positions
// [c*per, (c+1)*per) and the ids there are arbitrary. The gather is therefore
// scattered, and the winning candidate is already a token id with no inverse
// permutation to apply. [assist-centroids]

struct AssistTopArgs { uint n; uint k; };

// Top-k by k masked argmax passes over a staged copy. n is small (2048), so
// staging costs 8 KB of threadgroup memory and every pass is one reduction.
kernel void assist_top_clusters(
        device const float    * x   [[buffer(0)]],
        device       uint     * out [[buffer(1)]],
        constant AssistTopArgs & a  [[buffer(2)]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    threadgroup float s[2048];
    threadgroup float bv[32];
    threadgroup uint  bi[32];
    for (uint i = tpitg; i < a.n; i += ntg) { s[i] = x[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pass = 0; pass < a.k; pass++) {
        float best = -INFINITY;
        uint  at = 0;
        for (uint i = tpitg; i < a.n; i += ntg) {
            if (s[i] > best) { best = s[i]; at = i; }
        }
        const uint lane = tpitg % 32;
        const uint warp = tpitg / 32;
        for (uint off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(best, off);
            const uint  oi = simd_shuffle_down(at, off);
            if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
        }
        if (lane == 0) { bv[warp] = best; bi[warp] = at; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tpitg == 0) {
            float tv = bv[0];
            uint  ti = bi[0];
            for (uint w = 1; w < ntg / 32; w++) {
                if (bv[w] > tv || (bv[w] == tv && bi[w] < ti)) {
                    tv = bv[w];
                    ti = bi[w];
                }
            }
            out[pass] = ti;
            s[ti] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct AssistPickArgs {
    ulong woff;
    ulong ooff;
    uint  dim;
    uint  per;
    uint  clusters;
    uint  rowBytes;
};

// One Q4_0 row against the hidden vector. Byte j carries element j in its LOW
// nibble and j+16 in its HIGH one, w = (q - 8) * d -- the same layout
// q4_0_dequant_row reads.
inline float assist_row_dot(device const uchar * row, device const float * h,
                            uint dim) {
    float acc = 0.0f;
    const uint blocks = dim / 32;
    for (uint b = 0; b < blocks; b++) {
        device const uchar * bp = row + b * 18;
        const float d = (float) (*(device const half *) bp);
        device const uchar * qs = bp + 2;
        const uint base = b * 32;
        float s = 0.0f;
        for (uint i = 0; i < 16; i++) {
            const uchar byte = qs[i];
            s += ((float) (byte & 0x0F) - 8.0f) * h[base + i];
            s += ((float) (byte >> 4) - 8.0f) * h[base + i + 16];
        }
        acc += s * d;
    }
    return acc;
}

// The winning TOKEN ID over the tokens the chosen clusters own.
kernel void assist_cluster_argmax(
        device const uchar     * weights  [[buffer(0)]],
        device const float     * h        [[buffer(1)]],
        device const uchar     * obase    [[buffer(2)]],
        device const uint      * clusters [[buffer(3)]],
        device       int       * out      [[buffer(4)]],
        constant AssistPickArgs & a       [[buffer(5)]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    threadgroup float bv[32];
    threadgroup int   bi[32];
    device const uchar * table = weights + a.woff;
    device const float * ordering =
        (device const float *) (obase + a.ooff);
    const uint total = a.clusters * a.per;
    float best = -INFINITY;
    int   at = 0;
    for (uint i = tpitg; i < total; i += ntg) {
        const uint ordered = clusters[i / a.per] * a.per + (i % a.per);
        const int token = (int) rint(ordering[ordered]);
        const float score = assist_row_dot(
            table + (ulong) token * a.rowBytes, h, a.dim);
        if (score > best || (score == best && token < at)) {
            best = score;
            at = token;
        }
    }
    const uint lane = tpitg % 32;
    const uint warp = tpitg / 32;
    for (uint off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(best, off);
        const int   oi = simd_shuffle_down(at, off);
        if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
    }
    if (lane == 0) { bv[warp] = best; bi[warp] = at; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpitg == 0) {
        float tv = bv[0];
        int   ti = bi[0];
        for (uint w = 1; w < ntg / 32; w++) {
            if (bv[w] > tv || (bv[w] == tv && bi[w] < ti)) {
                tv = bv[w];
                ti = bi[w];
            }
        }
        out[0] = ti;
    }
}

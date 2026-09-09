---
type: File
title: LLM/metal/IQKernels.metal
description: GPU decode for the thirteen ggml super-block types, behind one
  32-weight sub-block primitive.
sources:
  - resource: LLM/metal/IQKernels.metal
tags: [orientation]
timestamp: 2026-09-06T20:00:00Z
---

`dq_sub` decodes one 32-weight group of a 256-weight block; `iq_dequant_row`
and `iq_embed_batch` are loops over the eight, and `iq_gemm_mm_h` is the
generic prefill tile that fills a threadgroup staging buffer from it. Q4_K,
Q5_K, Q6_K and IQ4_XS have their own compile-time tiles at the end of the
file, `GEMM_MM_KERNEL` over a super-block with sixteen-weight decoders. The
gemv and the narrow kernels do not materialize: `KqSub` and `KqSlice` carry
a sub-block, or a half or quarter of one, as float4 codes plus a scale pair,
and `kq_gemv_impl` / `kq_gemm_nb_impl` dot them against staged activations,
one decoder per type inlined by template; `kq_gemm_nw_impl` is the width-3
twin that reads its activations from device memory instead. The codebook
types reach both families through a generic slice over their materializing
decoders. Q8_0 and IQ4_NL join them as eight 32-weight blocks per 256-weight
span, with a compile-time tail flag for a row that is not a multiple of 256;
Q4_0 measured better on its own kernels in Kernels.metal. Included by
Kernels.metal BELOW `simd_mm_slice` and `store_mm_tile`, which the tile
shares; never compiled alone. The Swift codecs in `LLM/src/Quantize` are the
oracle and are themselves gated against ggml.

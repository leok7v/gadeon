---
type: File
title: LLM/metal/Kernels.metal
description: Every GPU kernel both Metal engines dispatch, in one file.
sources:
  - resource: LLM/metal/Kernels.metal
tags: [orientation]
timestamp: 2026-09-10T01:00:00Z
---

Compiled at build time into the default library, so there is no runtime
shader compile. Each kernel has a pure-Swift counterpart that stands as its
oracle.

A trap stays inline: anything silently wrong if you change it, a pairing
rule, a stride, a type width, keeps its warning at the code. The reasoning
and the numbers are concepts here instead, found with
`grep -rl Kernels.metal okf/`.

The dense and packed kernels come first, then the ggml super-block family,
then the drafting head. `dq_sub` decodes one 32-weight group of a 256-weight
block; `iq_dequant_row` and `iq_embed_batch` are loops over the eight, and
`iq_gemm_mm_h` is the generic prefill tile that fills a threadgroup staging
buffer from it. Q4_K, Q5_K, Q6_K and IQ4_XS have their own compile-time
tiles, `GEMM_MM_KERNEL` over a super-block with sixteen-weight decoders. The
gemv and the narrow kernels do not materialize: `KqSub` and `KqSlice` carry
a sub-block, or a half or quarter of one, as float4 codes plus a scale pair,
and `kq_gemv_impl` / `kq_gemm_nb_impl` dot them against staged activations,
one decoder per type inlined by template; `kq_gemm_nw_impl` is the width-3
twin that reads its activations from device memory instead. The codebook
types reach both families through a generic slice over their materializing
decoders. Q8_0 and IQ4_NL join them as eight 32-weight blocks per 256-weight
span, with a compile-time tail flag for a row that is not a multiple of 256;
Q4_0 and Q2_0 measured better on their own kernels. The Swift codecs in
`LLM/src/Quantize` are the oracle and are themselves gated against ggml.

The drafting head closes the file: k masked-argmax passes pick the best
clusters out of 2048, then one gathered dot-and-argmax runs over only the
tokens those clusters own. The token table stays in ORIGINAL token order, so
`token_ordering` yields token ids directly and the gather is scattered;
there is no inverse permutation to apply.

---
type: File
title: LLM/metal/IQKernels.metal
description: GPU decode for the thirteen ggml super-block types, behind one
  32-weight sub-block primitive.
sources:
  - resource: LLM/metal/IQKernels.metal
tags: [orientation]
timestamp: 2026-08-22T19:30:00Z
---

`dq_sub` decodes one 32-weight group of a 256-weight block; `iq_gemv`,
`iq_dequant_row` and `iq_embed_batch` are loops over the eight, and
`iq_gemm_mm`/`_h` is the prefill tile that fills a threadgroup staging
buffer from it. Included by Kernels.metal BELOW `simd_mm_slice` and
`store_mm_tile`, which the tile shares; never compiled alone. The Swift codecs in
`LLM/src/Quantize` are the oracle and are themselves gated against ggml.
See `one-sub-block-decode-carries-every-ggml-super-block-type`.

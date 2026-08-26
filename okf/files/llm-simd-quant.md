---
type: File
title: LLM/src/SIMD/Quant.swift
description: The 2-bit block types, their dequant and their mat-vec, and the
  type-dispatching front door the engines call.
sources:
  - resource: LLM/src/SIMD/Quant.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Q2_0 and Q2_X are scale first, 2-bit codes four per byte, sign living in the
code, verified against the real file and the reference dequantizer. Q2E8Row
reads the E8P codebook row instead, whose scales and codes are SEPARATE spans,
so a partial span is only addressable with the whole row's width in hand -- it
exists so the GPU kernel has a CPU oracle rather than only itself.

`QB` dispatches on the tensor's type and `GQ` covers the gemma block types.
A type missing from `QB` falls through to `GQ.gather`, which decodes an
unknown type AS BF16 rather than refusing -- so a new type must be added here,
not left to the default.
See `aneq-archive` for what this costs on the other engine,
and `aneq-archive` for the codebook itself.

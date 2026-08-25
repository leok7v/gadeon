---
type: File
title: LLM/src/Quantize/KQuants.swift
description: q2_K, q3_K and q4_K decode -- the non-IQ types that turn up
  inside unsloth's mixed files.
sources:
  - resource: LLM/src/Quantize/KQuants.swift
tags: [orientation]
timestamp: 2026-08-22T18:30:00Z
---

No codebooks here, only packed scales. Each type's field ORDER was taken
from ggml's own struct rather than guessed: q2_K puts scales first and its
`d`/`dmin` LAST at offset 80, q3_K ends with `d` at 108, q4_K starts with
them. q2_K also consumes TWO scale bytes per shift step, one per
16-element half -- missing the second is wrong by 0.18 and was caught only
by comparing against ggml.

THE FIXTURES. `IQ1CodecTests` compares against
`llama-quantize --allow-requantize <model> <out> F32`, ggml's own
dequantisation of the same weights. Types absent from the IQ1_S source model
get their own pair: requantize the 0.8B to that type, then that to F32, and
point the test at both. That is how q3_K is covered, and how any further
type should be.

---
type: File
title: LLM/src/Shared/Q2GPTQ.swift
description: GPTQ error compensation for the 2-bit block, on Accelerate.
sources:
  - resource: LLM/src/Shared/Q2GPTQ.swift
tags: [orientation]
timestamp: 2026-08-18T22:52:23Z
---

The Cholesky of a damped H inverse (dpotrf, dpotri, dpotrf), then per 128-column
block: fit the scale from the already-compensated weights, walk the columns
quantizing one at a time and pushing each residual onto the rest of the block
with a rank-1 sger, and finish with one sgemm that carries the block's error
onto every later column. Also the relative output error tr(D H D^T) / tr(W H W^T),
which is the only metric that ranks these builds correctly.

Why that metric and not a cosine, and what each lever is worth, is
`gptq-at-2b` and `q2x-block-and-qwen38-gguf`.

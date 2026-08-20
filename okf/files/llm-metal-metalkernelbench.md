---
type: File
title: LLM/src/Metal/MetalKernelBench.swift
description: One tensor, one kernel, many iterations, reported as GB/s.
sources:
  - resource: LLM/src/Metal/MetalKernelBench.swift
tags: [orientation]
timestamp: 2026-08-20T05:00:00Z
---

`--kernel-bench` times each GEMM variant on the model's own tensors and
divides the bytes a kernel MUST move by the time it took, so "how far from
bandwidth" is answerable without the whole-model bench's prefill, thermals
and page-cache noise. It is what set the per-type narrow thresholds in
`q2e8-narrow-is-not-amortizing`. Iterations ride one command buffer, so
the number is GPU time rather than per-commit latency.

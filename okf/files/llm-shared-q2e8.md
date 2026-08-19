---
type: File
title: LLM/src/Shared/Q2E8.swift
description: Build-time quantizer maths, from the Q2 scale fit through Q4_0,
  FP8 and BF16 encode.
sources:
  - resource: LLM/src/Shared/Q2E8.swift
tags: [orientation]
timestamp: 2026-08-18T21:32:34Z
---

Per-block scale fitting (absmax/3 seed, least-squares refit, optional
importance weighting and norm-preserving rescale), the AoS and SoA packings of
the 34-byte block, Q4_0, an FP8 E4M3 lookup table, and BF16 / F16 encode.
SIMD16<Float> inner loops under DispatchQueue.concurrentPerform over rows.

Why the scale is fitted rather than absmax/3, and why the levels are
mid-riser, is `q2x-block-and-qwen38-gguf`.

---
type: File
title: LLM/src/SIMD/Quant.swift
description: The ternary block type, its dequant and its mat-vec.
sources:
  - resource: LLM/src/SIMD/Quant.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Scale first, 2-bit codes four per byte, with the sign living in the code. The
layout is verified against the real file and the reference dequantizer.
See `ternary-ane-fp16-expansion` for what this costs on the other engine.

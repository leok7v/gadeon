---
type: File
title: LLM/src/SIMD/Tensor.swift
description: A strided f32 CPU tensor library over Accelerate.
sources:
  - resource: LLM/src/SIMD/Tensor.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Transliterated from the C the speech model was ported from: same declaration
order, same names, same arithmetic in the same sequence, and strides in
BYTES as in the C. Only the float data comes from the arena, so an arena
reset still invalidates it. See `tts-in-llm`.

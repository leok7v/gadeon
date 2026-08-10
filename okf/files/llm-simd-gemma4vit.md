---
type: File
title: LLM/src/SIMD/Gemma4ViT.swift
description: The gemma-4 vision tower on the CPU.
sources:
  - resource: LLM/src/SIMD/Gemma4ViT.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

It is NOT shaped like the Qwen tower next door: no conv patch embed, no square
grid, no LayerNorm and no merger. It is far closer to gemma's own text block,
which is why it reuses those primitives.

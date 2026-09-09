---
type: File
title: LLM/src/Qwen/QwenViT.swift
description: The Qwen3-VL vision tower on the CPU.
sources:
  - resource: LLM/src/Qwen/QwenViT.swift
tags: [orientation]
timestamp: 2026-09-05T02:00:00Z
---

Patch embed, position embed, uniform pre-norm blocks with vision rope, then
the merger into the language embedding space, on whatever h x w patch grid
an image resized to: the learned position table is resampled to the grid,
and the merge order and rope tables are built per grid. Every geometry
number comes from metadata or a tensor shape; the square path is gated byte
for byte against a numpy reference and the GPU tower is gated against this
one on a rectangle.

The merge order, rope tables, im2col and position resampling are STATIC so
`QwenMetalViT` builds its buffers from the same code; the two engines
cannot drift on layout.

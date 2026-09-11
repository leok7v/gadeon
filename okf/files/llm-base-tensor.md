---
type: File
title: LLM/Base/Tensor.swift
description: A strided f32 CPU tensor library over Accelerate.
sources:
  - resource: LLM/Base/Tensor.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Transliterated from the C the speech model was ported from: same declaration
order, same names, same arithmetic in the same sequence, and strides in
BYTES as in the C. Only the float data comes from the arena, so an arena
reset still invalidates it.

Several ops here have hand-rolled counterparts elsewhere: `tensorIm2col`
against `QwenViT`'s patch layout, the permute/reshape pair against
`Gemma4Audio` and `Gemma4Patchify`, the elementwise leaves against `Kern`
and `GK`. They cannot be collapsed into one another while this side is
compared bit for bit against the C: matching results is not enough, the
order the operations combine in has to match too, and `Kern` and `GK` are
each pinned to a different reference. Consolidation, if it ever happens,
moves the other callers onto these ops rather than the reverse.

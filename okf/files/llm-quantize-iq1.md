---
type: File
title: LLM/src/Quantize/IQ1.swift
description: ggml IQ1_S decode, gated bit-for-bit against ggml's own
  dequantisation of the same file.
sources:
  - resource: LLM/src/Quantize/IQ1.swift
tags: [orientation]
timestamp: 2026-08-22T17:40:00Z
---

`IQ1.dot` accumulates one 256-weight block against activations and
`IQ1.dequant` writes the block out. The grid values are ternary offset by
one, so the accumulation is `dl * (lo + 2*hi + (delta - 1) * sx)` -- the same
predicated-add shape `q2_0_gemv` already uses, with `(delta - 1)` where that
kernel has `-1`.

TWO THINGS THAT ARE EASY TO GET WRONG AND ARE GATED. The 11-bit index is a
`qs` byte plus three bits lifted out of `qh`, and the eight nibbles of a grid
entry are NOT consecutive: weight j sits at byte `j % 4`, half `j / 4`.
Reading them consecutively is wrong by 0.46 on real weights and looks
plausible until compared. See [[llm-quantize-iq1grid]].

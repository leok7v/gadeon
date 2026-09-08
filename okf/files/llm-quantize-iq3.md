---
type: File
title: LLM/src/Quantize/IQ3.swift
description: iq2_xs, iq2_s, iq3_xxs, iq3_s and iq4_xs decode -- the rest of
  the IQ family, all gated against ggml.
sources:
  - resource: LLM/src/Quantize/IQ3.swift
tags: [orientation]
timestamp: 2026-08-22T18:38:00Z
---

Four of these five share one shape: a grid of UNSIGNED magnitudes, a sign
pattern from `ksigns_iq2xs`, and a sub-scale per 32 weights off the block's
f16 `d`. iq4_xs is the odd one -- no grid, a 16-entry non-linear value table
and a 6-bit scale split across `scales_h` and `scales_l`.

EVERY OFFSET CAME FROM ggml's STRUCT, not from the dequant loop. iq2_s is
why: its signs are not a field at all, they live INSIDE `qs` at `qs + 32`,
and `qh` sits after them at 66. Reading the struct field order as if the
dequant's pointer arithmetic described it puts both in the wrong place.

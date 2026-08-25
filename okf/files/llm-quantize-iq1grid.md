---
type: File
title: LLM/src/Quantize/IQ1Grid.swift
description: The 2048-entry ggml iq1s_grid_gpu table, transcribed, plus the
  0.125 delta constant.
sources:
  - resource: LLM/src/Quantize/IQ1Grid.swift
tags: [orientation]
timestamp: 2026-08-22T17:40:00Z
---

2048 uint32 entries, 8 KB, extracted from `ggml-common.h`. Each entry packs
eight nibbles and EVERY nibble in the whole table is 0, 1 or 2 -- checked,
not assumed -- so the grid is ternary offset by one and needs no multiply
per weight. That is what makes IQ1_S land on the cheap decode path rather
than the E8 one. See [[llm-quantize-iq1]].

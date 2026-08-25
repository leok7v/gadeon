---
type: File
title: LLM/metal/IQTables.metal
description: The ggml IQ codebooks as MSL constant arrays, generated from
  the Swift grids.
sources:
  - resource: LLM/metal/IQTables.metal
tags: [orientation]
timestamp: 2026-08-22T19:30:00Z
---

Generated, not hand-edited: the 64-bit grids are split into low/high uint
pairs because MSL has no 64-bit literal arrays. Included by Kernels.metal.
See `one-sub-block-decode-carries-every-ggml-super-block-type`.

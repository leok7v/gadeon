---
type: File
title: LLM/src/Quantize/IQ2XXS.swift
description: ggml IQ2_XXS and IQ1_M decode, both gated bit-for-bit against
  ggml's own dequantisation.
sources:
  - resource: LLM/src/Quantize/IQ2XXS.swift
tags: [orientation]
timestamp: 2026-08-22T18:30:00Z
---

IQ2_XXS reads eight unsigned magnitudes from a 256-entry grid and takes each
sign from a 128-entry table indexed by seven bits of the second aux word. Its
decode shape is the SAME as our own E8 -- table, explicit signs, a real
multiply per weight -- so porting it buys no ALU win, only the codebook.

IQ1_M shares IQ1_S's grid but NOT its header: the block carries no
`ggml_half` at all. The f16 scale is reassembled four nibbles at a time from
the four `scales` shorts, and each 32-weight group takes two sub-scales
rather than one. Assuming the IQ1_S layout here silently yields noise.

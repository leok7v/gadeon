---
type: File
title: LLM/src/Quantize/RowScales.swift
description: Reads the .rsc row-scale file and multiplies a tensor's output
  rows before quantization.
sources:
  - resource: LLM/src/Quantize/RowScales.swift
tags: [orientation]
timestamp: 2026-08-20T23:25:00Z
---

A per-output-row gain learned outside the converter, applied inside it. The
converter calls `scale()` right after `apply()` and before the quantizer, so
the fit sees the scaled weights and picks `d' = c*d` for that row's blocks --
which is why the correction costs no bits: it rides in the fp16 scales Q2_E8
already stores.

Format is "RSC1", a count, then name and float vector per entry. It stops at
reading and applying; producing the scales is the tuning step, which lives
outside this repo for now. See
`calibration-corpus-and-recovery`.

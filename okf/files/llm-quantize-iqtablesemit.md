---
type: File
title: LLM/src/Quantize/IQTablesEmit.swift
description: The MSL codebook header, emitted from the Swift grids the SIMD
  engine already carries.
sources:
  - resource: LLM/src/Quantize/IQTablesEmit.swift
tags: [orientation]
timestamp: 2026-09-10T03:00:00Z
---

A pure function returning the text of `LLM/metal/IQTables.h`, so the GPU's
codebooks and the CPU's are one transcription rather than two. `gadeon
--emit-iq-tables` writes it and `IQTablesTests` fails when the committed
header and the grids disagree.

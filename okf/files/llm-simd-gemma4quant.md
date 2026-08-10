---
type: File
title: LLM/src/SIMD/Gemma4Quant.swift
description: The block types the gemma-4 repack emits.
sources:
  - resource: LLM/src/SIMD/Gemma4Quant.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Layouts byte for byte as the repack writes them, and as its probe proves
against the source checkpoint. The ternary block type lives next door and is
reused unchanged.

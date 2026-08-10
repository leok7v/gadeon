---
type: File
title: LLM/src/SIMD/ViT.swift
description: The Qwen3-VL vision tower on the CPU.
sources:
  - resource: LLM/src/SIMD/ViT.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Patch embed, position embed, uniform pre-norm blocks with vision rope, then
the merger into the language embedding space. Every geometry number comes
from metadata or a tensor shape, and it is gated byte for byte against a
numpy reference.

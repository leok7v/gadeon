---
type: File
title: LLM/src/Shared/Safetensors.swift
description: A multi-shard safetensors reader that decodes to f32.
sources:
  - resource: LLM/src/Shared/Safetensors.swift
tags: [orientation]
timestamp: 2026-08-18T21:32:34Z
---

Every *.safetensors in a directory mapped and merged into one name-to-entry
table, with F32 / F16 / BF16 / F8_E4M3 decoded to f32 on demand. A tensor with
a sibling .weight_scale_inv is expanded through its 128x128 block scales.

Each shard owns its descriptor and pages so a header parse that throws still
frees them, the same shape as [[llm-shared-gguf]].

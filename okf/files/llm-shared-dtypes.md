---
type: File
title: LLM/src/Shared/Dtypes.swift
description: BF16 and FP8 e4m3 decode, the two safetensors dtypes the
  loader meets that Swift has no native type for.
sources:
  - resource: LLM/src/Shared/Dtypes.swift
tags: [orientation]
timestamp: 2026-08-25T12:10:00Z
---

Both are read-only decoders into f32, used by `Safetensors.tensor`. They
lived in `Q2E8.swift` until the Q2_E8 removal
(`q2x-q2z-q2e8-are-removed`) and were the only part of that file with a
live caller: the drafter graft reads safetensors, and an origin checkpoint
is bf16 or fp8. FP8's 256-entry e4m3 table is built once and shared.

---
type: File
title: LLM/src/SIMD/BonsaiEngine.swift
description: The ternary forward loop on the CPU.
sources:
  - resource: LLM/src/SIMD/BonsaiEngine.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The hybrid layer stack token by token, threading per-layer conv, recurrent
and KV state. This is the oracle every GPU kernel is gated against.
See `bonsai-1.7b-dense-qwen3`.

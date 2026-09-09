---
type: File
title: LLM/src/Qwen/QwenAttention.swift
description: The full-attention layer, one token at a time.
sources:
  - resource: LLM/src/Qwen/QwenAttention.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Fused projection, per-head QK norm, partial rope, GQA, causal softmax,
output gate. Its page pool is append only, so a bookmark shares completed
pages by reference and only the tail page copies: a mark costs the tail, not
the context.

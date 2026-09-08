---
type: File
title: LLM/src/Qwen/QwenMetalEngine.swift
description: The ternary forward loop on the GPU.
sources:
  - resource: LLM/src/Qwen/QwenMetalEngine.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The same layer stack as the CPU engine, token by token, threading per-layer
state in resident buffers. All kernels for one token ride one command buffer
and only the logits are read back. The CPU engine is the oracle.
See `metal-perf-ceilings` and `metal-longcontext-attention`.

A `Bookmark` copies the GDN recurrence whole, which cannot be paged, and
shares the attention KV as append-only page snapshots, so only a partial
tail copies; it mirrors `QwenEngine.Bookmark`. `ropeShift` is added to the
sequence index for rope angles only: a vision prefill compresses each
image's M-RoPE span and the text after it ropes at the compressed scalar,
while causality stays on raw positions. On the token-by-token path
the stop gate is the loop condition and `QwenMetalBackend.extend` turns it
into `EngineError.stopped`, so a prefill Stop rolls the turn back.

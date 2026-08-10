---
type: File
title: LLM/src/Metal/MetalKVPool.swift
description: The paged KV cache for one attention layer, on the GPU.
sources:
  - resource: LLM/src/Metal/MetalKVPool.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Fixed-size pages allocated on first write, so context grows without one giant
contiguous buffer and without realloc copies. The attention kernel gathers
across them through a bindless page table.
See `metal-longcontext-attention` for why windowing this storage is a
regression rather than a saving.

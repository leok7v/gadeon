---
type: File
title: LLM/src/Base/MetalKVPool.swift
description: The paged KV cache for one attention layer, on the GPU.
sources:
  - resource: LLM/src/Base/MetalKVPool.swift
tags: [orientation]
timestamp: 2026-09-05T12:00:00Z
---

Fixed-size pages of P positions, each its own MTLBuffer allocated on first
write, so context grows toward the page table's bound without one giant
contiguous buffer and without realloc copies. The attention kernel gathers
across them through a bindless page table. Append only: completed pages are
never mutated, so a bookmark shares them by reference and copies only the
partial tail page.

Pages are half precision. Decode cost is linear in cached positions, so past
a few hundred tokens the KV read is most of what a token moves, and on the
dense 1.7B the pages (~112 MB per 512 positions at f32) were what bounded
context on a 3 GB phone; llama.cpp defaults to f16 KV as well.

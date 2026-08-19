---
type: File
title: LLM/src/Metal/MetalContext.swift
description: One device, one queue, the kernel library and the mapped
  weights.
sources:
  - resource: LLM/src/Metal/MetalContext.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Tensors are addressed by byte offset into a no-copy window buffer over the
mmap'd file, so there is no per-tensor buffer and no offset-alignment
constraint. Only this file builds a window, so a dispatch cannot bind one
window and address into another.

It also owns the two shared read-only buffers no single caller should
allocate: the all-empty vision block table, and Q2_E8's 256-entry codebook. The
codebook is a plain device buffer rather than threadgroup memory because it is
512 bytes in its 2-bit packed form and stays in L2 -- a per-threadgroup copy of
the 4 KB float table would have doubled the weight traffic of a 1024x6144
tensor. See `e8-lattice-codebook`.

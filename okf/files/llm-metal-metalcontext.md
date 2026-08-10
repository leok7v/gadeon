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

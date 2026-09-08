---
type: File
title: LLM/src/Base/MetalContext.swift
description: One device, one queue, the kernel library and the mapped
  weights.
sources:
  - resource: LLM/src/Base/MetalContext.swift
tags: [orientation]
timestamp: 2026-09-05T12:00:00Z
---

Tensors are addressed by byte offset into a no-copy window buffer over the
mmap'd file, so there is no per-tensor buffer and no offset-alignment
constraint. Only this file builds a window, so a dispatch cannot bind one
window and address into another. The kernel library is the build-time
`default.metallib` in the module's bundle; nothing compiles shader source at
run time.

Every pipeline this GPU can build is built before any encoding starts, and
the simdgroup-matrix kernels are skipped on a GPU without matrix units,
because building one there takes the shader compiler service down rather
than returning an error; `matrixUnits` is what the vision and audio paths
gate on. It also owns the one shared read-only buffer no single caller should
allocate: the all-empty vision block table.

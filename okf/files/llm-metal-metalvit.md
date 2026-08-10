---
type: File
title: LLM/src/Metal/MetalViT.swift
description: The Qwen3-VL vision tower on the GPU.
sources:
  - resource: LLM/src/Metal/MetalViT.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The same op sequence as the CPU tower, which stays the oracle. Weights
dequantize ONCE at load into resident half buffers and are kept, so an image
turn pays neither the reload nor the dequant, and matrices stay in the
native row-major layout the kernel reads directly.

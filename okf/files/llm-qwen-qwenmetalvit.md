---
type: File
title: LLM/src/Qwen/QwenMetalViT.swift
description: The Qwen3-VL vision tower on the GPU.
sources:
  - resource: LLM/src/Qwen/QwenMetalViT.swift
tags: [orientation]
timestamp: 2026-09-05T02:00:00Z
---

The same op sequence as the CPU tower, which stays the oracle. Weights
dequantize ONCE at load into resident half buffers and are kept, so an image
turn pays neither the reload nor the dequant, and matrices stay in the
native row-major layout the kernel reads directly. Per-grid tables and
scratch are built on first use and kept for the grid that repeats.

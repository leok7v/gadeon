---
type: File
title: LLM/src/Metal/Gemma4MetalViT.swift
description: The gemma-4 vision tower on the GPU.
sources:
  - resource: LLM/src/Metal/Gemma4MetalViT.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

It reuses the TEXT tower's norm kernels rather than the ViT ones next door,
because gemma's encoder block IS gemma's text block. Weights stay quantized in
the mapped file and expand in registers inside the GEMM.

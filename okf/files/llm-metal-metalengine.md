---
type: File
title: LLM/src/Metal/MetalEngine.swift
description: The ternary forward loop on the GPU.
sources:
  - resource: LLM/src/Metal/MetalEngine.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The same layer stack as the CPU engine, token by token, threading per-layer
state in resident buffers. All kernels for one token ride one command buffer
and only the logits are read back. The CPU engine is the oracle.
See `metal-perf-ceilings` and `metal-longcontext-attention`.

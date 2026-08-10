---
type: File
title: LLM/src/Metal/Gemma4MetalEngine.swift
description: The gemma-4 text forward on the GPU.
sources:
  - resource: LLM/src/Metal/Gemma4MetalEngine.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Every kernel for one token is encoded onto ONE command buffer and committed
once, so a decoded token costs one dispatch stream and one sync. The weights
are MIXED, so a dispatch names a tensor and lets the encoder pick the kernel
rather than assuming a type. The CPU engine is the oracle.

---
type: File
title: LLM/src/Base/MetalGolden.swift
description: Byte-exact golden buffers for the kernels.
sources:
  - resource: LLM/src/Base/MetalGolden.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Bytes rather than cosine on purpose: a consolidation claims the identical
sequence of floating-point operations, so equality is the honest test.
Cosine cannot see a one-ulp change, and a rounding step function puts a high
noise floor under every comparison anyway.

Its pseudo-random fill matches `QwenMetalSelfTest.fill`, so a case
reproduces from either harness.

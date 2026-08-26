---
type: File
title: LLM/src/Metal/MetalEnc.swift
description: The typed one-token dispatch surface.
sources:
  - resource: LLM/src/Metal/MetalEnc.swift
tags: [orientation]
timestamp: 2026-08-26T12:00:00Z
---

Each method encodes ONE kernel with the bindings its shader counterpart
expects. Kept apart from the engine so the forward loop reads as an op
sequence and the binding detail lives here.

`gemm` routes by WIDTH: N=1 to the per-type gemv, 1 < N <= narrowMax to the
per-type narrow kernel, wider to the simdgroup-matrix tile. All three are
dedicated per type; nothing super-blocked reaches `dq_sub`'s runtime switch
below tile width (`the-narrow-kernel-was-the-whole-mtp-verdict`).


---
type: File
title: LLM/src/Metal/MetalEnc.swift
description: The typed one-token dispatch surface.
sources:
  - resource: LLM/src/Metal/MetalEnc.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Each method encodes ONE kernel with the bindings its shader counterpart
expects. Kept apart from the engine so the forward loop reads as an op
sequence and the binding detail lives here.

`LLM_IQ_ROWS=1` routes super-blocked types at narrow N to `gemmRows`
instead of the prefill tile -- the A/B instrument that measured the MTP
verify gap, default OFF (`iq-verify-has-no-narrow-kernel`).


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

---
type: File
title: LLM/Qwen/QwenMetalBackend.swift
description: The GPU engine adapted to the backend seam.
sources:
  - resource: LLM/Qwen/QwenMetalBackend.swift
tags: [orientation]
timestamp: 2026-09-05T12:00:00Z
---

`EngineBackend` over the GPU Qwen engine plus everything vision: it owns
the resident vision tower, lends it to `QwenMedia`, and splices soft spans
over the template's `<|image_pad|>` runs onto the engine's M-RoPE vision
prefill, one grid per span. `QwenMetalChat` is the GPU counterpart of
`QwenChat` and loads everything from the one GGUF, the vision tower beside
it or inside it.

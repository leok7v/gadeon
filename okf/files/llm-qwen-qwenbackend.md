---
type: File
title: LLM/src/Qwen/QwenBackend.swift
description: The ternary CPU engine adapted to the backend seam.
sources:
  - resource: LLM/src/Qwen/QwenBackend.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

`QwenBackend` is `EngineBackend<QwenEngine, Tokenizer>` with nothing
added, and `QwenChat` is the loaded ternary model on the CPU: engine,
tokenizer, template and sampling presets out of the one GGUF.

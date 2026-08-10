---
type: File
title: LLM/src/Shared/ModelShape.swift
description: What a loaded model IS, before any turn has produced a
  number.
sources:
  - resource: LLM/src/Shared/ModelShape.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Every field is read from the model's own files: a metadata key, a tensor's
size, a compiled program on disk. Nothing is keyed by model name, so a new
checkpoint describes itself.
See `feedback-no-hardcoded-model-constants` and `qwenpaw-4b-9b-geometry`.

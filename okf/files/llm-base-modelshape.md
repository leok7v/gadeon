---
type: File
title: LLM/src/Base/ModelShape.swift
description: What a loaded model IS, before any turn has produced a
  number.
sources:
  - resource: LLM/src/Base/ModelShape.swift
tags: [orientation]
timestamp: 2026-09-05T12:00:00Z
---

Every field is read from the model's own file: a metadata key or a tensor's
size. Nothing is keyed by model name, so a new checkpoint describes itself.
A tower that ships as its own file arrives as a sidecar.

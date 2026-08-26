---
type: File
title: LLM/src/Shared/EngineTypes.swift
description: The error and the speculative-decode tally every backend
  shares.
sources:
  - resource: LLM/src/Shared/EngineTypes.swift
tags: [orientation]
timestamp: 2026-08-26T02:00:00Z
---

Two types that outlived the engine they were declared in: the error a
backend throws when it cannot serve a turn, and the per-turn count a
speculative decoder drains. Both are spoken by Metal and SIMD alike, which
is why they sit here rather than inside any one engine.

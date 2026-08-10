---
type: File
title: LLM/src/SIMD/Gemma4Model.swift
description: gemma-4 geometry and per-layer tensor handles.
sources:
  - resource: LLM/src/SIMD/Gemma4Model.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Every scalar comes from a metadata key and every width from a tensor shape,
so a re-emit with different dimensions loads unchanged. It is its own type
rather than a variant of the ternary config, because the key set genuinely
differs. See `gemma4-port`.

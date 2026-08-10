---
type: File
title: LLM/src/CoreML/PagedAttention.swift
description: Paged KV attention over a lazily allocated page pool.
sources:
  - resource: LLM/src/CoreML/PagedAttention.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Flash-style tiles whose un-normalized partials merge on the host with an fp32
online softmax. Tiles emit their output transposed because the natural form
lowers to CPU. Every model-shaped number arrives from the engine's own
geometry rather than being written here.
See `feedback-no-hardcoded-model-constants`.

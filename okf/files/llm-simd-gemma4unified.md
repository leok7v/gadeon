---
type: File
title: LLM/src/SIMD/Gemma4Unified.swift
description: The multimodal path of a unified gemma-4, which has no
  towers.
sources:
  - resource: LLM/src/SIMD/Gemma4Unified.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

An image becomes pixel blocks and audio becomes sample frames, each through
one small stack into the language model's embedding space. A few matrix-vector
products per token rather than an encoder. See `gemma4-port`.

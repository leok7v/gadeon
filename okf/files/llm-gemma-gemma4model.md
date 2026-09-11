---
type: File
title: LLM/Gemma/Gemma4Model.swift
description: gemma-4 geometry and per-layer tensor handles.
sources:
  - resource: LLM/Gemma/Gemma4Model.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Every scalar comes from a metadata key and every width from a tensor shape,
so a re-emit with different dimensions loads unchanged. It is its own type
rather than a variant of the ternary config, because the key set genuinely
differs.

`GemmaChat` is the public door: engine, tokenizer and chat template out of
the one file, the counterpart of `QwenChat` and `QwenMetalChat`.
`chunkLength`, the prefill chunk walk, is gated directly by
`VisionBlockTests`: a walk over a turn must cover every id exactly once and
never cut a vision block, since a block reads forward across itself and a
split block answers fluently and wrongly.

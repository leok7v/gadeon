---
type: File
title: LLM/src/Base/EngineBackend.swift
description: The one adapter from any text engine to the backend seam.
sources:
  - resource: LLM/src/Base/EngineBackend.swift
tags: [orientation]
timestamp: 2026-09-05T07:00:00Z
---

`TextEngine` names what an engine must offer (position, sampler, reset,
extend, decode, bookmark and restore, the parked-bytes codec, and the
optional stop, queue and spec counters, defaulted for a plain engine) and
`Tokenizing` what a tokenizer must. `EngineBackend<E, T>` is every seam
method the four lineage adapters used to spell by hand: the stop-to-throw on
extend, mark and rewind, the state, checkpoint and turn structs, the bytes
codec. A lineage subclass keeps only what differs: soft tokens, vision, the
matrix-units gate. `loadState` takes either the adapter's own `State` or the
engine's parked type, which for an engine without a separate parked form is
its bookmark.

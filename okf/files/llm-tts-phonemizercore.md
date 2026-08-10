---
type: File
title: LLM/src/TTS/PhonemizerCore.swift
description: Where the phonemizer port starts, in the C's own order.
sources:
  - resource: LLM/src/TTS/PhonemizerCore.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Primitives and leaf helpers first, then the rule engine, the prosody pass,
the IPA renderer and the stateful engine in the files beside it. Its buffers
are BYTES rather than text, because the engine stores in-band sentinels
inside what look like strings.

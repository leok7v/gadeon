---
type: File
title: LLM/src/TTS/Kittens.swift
description: The speech model itself, transliterated from its reference C.
sources:
  - resource: LLM/src/TTS/Kittens.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Same declaration order, same names in Swift casing, same loop bounds, same
arithmetic in the same sequence, so the WAV byte comparison stays meaningful.
Only the tracing and the host memory choreography were dropped, and both are
bit neutral by construction. See `tts-in-llm`.

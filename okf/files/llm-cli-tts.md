---
type: File
title: LLM/cli/TTS.swift
description: The speech probe. Text in, WAV out, no model load.
sources:
  - resource: LLM/cli/TTS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

It writes the same 24 kHz mono PCM WAV the reference binary does, because the
acceptance test for the speech engine is a byte comparison against it.
See `tts-in-llm`.

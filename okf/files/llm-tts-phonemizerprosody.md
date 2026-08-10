---
type: File
title: LLM/src/TTS/PhonemizerProsody.swift
description: The prosody sub-steps, in the order they run.
sources:
  - resource: LLM/src/TTS/PhonemizerProsody.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Each operates in place on the working phoneme buffer. The order is the C's
order and is load bearing, since each step assumes the last one ran.

---
type: File
title: LLM/src/SIMD/Gemma4Mel.swift
description: The log-mel frontend the gemma-4 audio tower expects.
sources:
  - resource: LLM/src/SIMD/Gemma4Mel.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Pure DSP with no weights, so it gates on its own against the reference input
features. Three details are not the obvious defaults and each shifts every
frame: semicausal padding, a periodic Hann window, and the log taken of mel
plus floor rather than of their maximum.

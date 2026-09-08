---
type: File
title: LLM/src/TTS/Phonemizer.swift
description: The stateful half of the phonemizer port.
sources:
  - resource: LLM/src/TTS/Phonemizer.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The loaders, the dispatch chain, the rule-scan main loop and the per-token
pipeline. Everything in the C that took the phonemizer struct became this
class.

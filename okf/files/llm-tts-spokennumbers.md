---
type: File
title: LLM/src/TTS/SpokenNumbers.swift
description: Numerals turned into the words a voice says for them.
sources:
  - resource: LLM/src/TTS/SpokenNumbers.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

It runs BEFORE the phonemizer, and that placement is the design: the
phonemizer's tokenizer would break a number at its separators and end a
sentence inside it, and fixing that there would mean editing a file gated byte
for byte against its C.

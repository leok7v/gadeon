---
type: File
title: LLM/Gemma/Gemma4Backend.swift
description: The gemma-4 CPU engine adapted to the backend seam.
sources:
  - resource: LLM/Gemma/Gemma4Backend.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

`EngineBackend` over the CPU gemma engine plus the two things gemma adds
on the seam: soft tokens are always on, and `extendSoft` feeds tower rows
through `SoftFeed`. The SentencePiece tokenizer and the three-id stop set
come in through `Tokenizing`.

---
type: File
title: LLM/src/SIMD/Gemma4Backend.swift
description: The gemma-4 CPU engine adapted to the backend seam.
sources:
  - resource: LLM/src/SIMD/Gemma4Backend.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Two things differ from the ternary backend and both are gemma facts rather
than style: the tokenizer is the SentencePiece one, and a turn ends on a SET
of ids rather than one, which the seam carries.

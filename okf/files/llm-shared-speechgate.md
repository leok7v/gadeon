---
type: File
title: LLM/src/Shared/SpeechGate.swift
description: Which live audio ever becomes tokens.
sources:
  - resource: LLM/src/Shared/SpeechGate.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The same energy and adaptive-floor detector AudioChunks uses on a finished
file, turned around: there it marks pauses to cut at and keeps everything,
here it marks speech to keep and the silence never reaches the tower. The
margin is asymmetric because a symmetric one eats word-final consonants.
See `audio-chunking-and-vad`.

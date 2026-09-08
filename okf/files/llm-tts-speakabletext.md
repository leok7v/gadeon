---
type: File
title: LLM/src/TTS/SpeakableText.swift
description: Markdown answer text turned into sentences a voice should
  say.
sources:
  - resource: LLM/src/TTS/SpeakableText.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

push() returns only the segments that are COMPLETE, so speech starts on
sentence one rather than at the end of the turn. Two block kinds are
DESCRIBED rather than read out, because a fenced code block read aloud is
unbearable and a table read cell by cell is worse.
See `voice-conversation-landed`.

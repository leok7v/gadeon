---
type: File
title: App/VoiceSession.swift
description: The speaking half of a voice conversation.
sources:
  - resource: App/VoiceSession.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

It takes the answer as it streams, decides what of it is worth saying, and
says it. Text runs ahead of speech by design, because generation is several
times faster than the voice and throttling generation to match buys only
buffered text.

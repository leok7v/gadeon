---
type: File
title: LLM/src/TTS/Speech.swift
description: Offline text to speech. Text in, samples out.
sources:
  - resource: LLM/src/TTS/Speech.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The facade is deliberately a PURE SYNCHRONOUS FUNCTION of text, voice and
speed. Nothing here queues, plays or knows about a turn, which is what keeps
this side comparable byte for byte against the reference binary. The speaking
session lives in the app. See `tts-in-llm`.

---
type: File
title: LLM/src/Shared/AudioChunks.swift
description: Cut a recording at its pauses, not at arbitrary offsets.
sources:
  - resource: LLM/src/Shared/AudioChunks.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

A tower has a hard ceiling, so a long clip must be split whatever happens;
splitting on a timer cuts words in half, and splitting on silences the
speaker already left costs nothing. The floor is a low percentile of the
clip's own frame energies, because a fixed dB threshold is wrong in both a
quiet room and a noisy one. See `audio-chunking-and-vad`.

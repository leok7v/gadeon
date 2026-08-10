---
type: File
title: LLM/src/Shared/AudioFile.swift
description: Any container the system reads, decoded to mono f32.
sources:
  - resource: LLM/src/Shared/AudioFile.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The target rate is a PARAMETER, because it belongs to a model's frontend
rather than to audio, so this file never learns which model asked. It reads
through AVAssetReader because the source may be a video container whose audio
is one track among several.

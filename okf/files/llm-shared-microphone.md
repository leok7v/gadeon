---
type: File
title: LLM/src/Shared/Microphone.swift
description: The microphone as mono f32 samples at a requested rate.
sources:
  - resource: LLM/src/Shared/Microphone.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The rate is a parameter for the same reason it is in AudioFile. The tap fires
on a real-time audio thread and does allocate there, which is a measured
compromise against a 4096-frame buffer; what must never appear in that
callback is a model, the GPU, or an await.

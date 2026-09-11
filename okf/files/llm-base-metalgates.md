---
type: File
title: LLM/Base/MetalGates.swift
description: The two lock-backed flags every GPU forward polls.
sources:
  - resource: LLM/Base/MetalGates.swift
tags: [orientation]
timestamp: 2026-09-05T05:00:00Z
---

`MetalStopSignal` is the stop lever for a synchronous Metal forward that
holds the session actor with no suspension point: the app raises it, the
prefill loops poll it between chunks and the decode loop polls it through
the backend. `BackgroundGate` parks every command-buffer commit while iOS
has the app in the background, where a GPU submit is aborted; macOS never
raises it. Both engines share them, which is why they live here and not in
either engine.

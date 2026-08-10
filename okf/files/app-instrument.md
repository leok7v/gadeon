---
type: File
title: App/Instrument.swift
description: Debug instrumentation, wired once at launch.
sources:
  - resource: App/Instrument.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Markdown render timing into the shared log, and a background watchdog
measuring how long the main queue goes undrained, which is the direct signal
for a render stall independent of what blocked it. Both are threshold gated,
so a smooth session is silent.

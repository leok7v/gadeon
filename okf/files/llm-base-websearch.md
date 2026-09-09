---
type: File
title: LLM/src/Base/WebSearch.swift
description: The two web-search providers, and the one shape they both
  come back as.
sources:
  - resource: LLM/src/Base/WebSearch.swift
tags: [orientation]
timestamp: 2026-09-09T02:00:00Z
---

`SearchProvider` is the switch pair Settings renders and the model's tool
list is gated on. `WebSearch` normalizes either provider into `SearchHit`
and renders one format, so the wire the model learns never depends on which
answered. `ParallelSearch` is a small MCP client over their HTTP endpoint,
holding the handshake and the advertised tool schema for the process.

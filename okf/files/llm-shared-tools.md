---
type: File
title: LLM/src/Shared/Tools.swift
description: The cross-platform half of the agentic tool layer.
sources:
  - resource: LLM/src/Shared/Tools.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Tool specs, the tool-call parser, a pure pre-exec sanitizer that touches
neither filesystem nor network, and the sandbox-safe tools over URLSession.

It stops at the sandbox. Shell and filesystem tools need process spawning and
live in the CLI target. See `web-tools-rl-aligned` and
`generic-toolcall-handover`.

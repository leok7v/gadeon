---
type: File
title: LLM/src/Metal/MetalSelfTest.swift
description: Bring-up validation, before the full engine exists.
sources:
  - resource: LLM/src/Metal/MetalSelfTest.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Each check dispatches one kernel and diffs its output against the pure-Swift
reference on identical inputs, so a kernel is proven in isolation before it is
wired into the forward loop.

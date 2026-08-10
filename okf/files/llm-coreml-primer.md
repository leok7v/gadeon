---
type: File
title: LLM/src/CoreML/Primer.swift
description: Compiling a set while it is still downloading.
sources:
  - resource: LLM/src/CoreML/Primer.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Loading a model triggers the daemon compile and the cache entry outlives the
discarded model, so the one-time optimize overlaps the download. In process,
because a sandboxed app cannot spawn a copy of its own binary, and one
program at a time, because the compiler serializes globally. Purely additive:
any failure leaves the cache cold.
See `ane-compile-perf-decisions` and `ane-compile-monitoring`.

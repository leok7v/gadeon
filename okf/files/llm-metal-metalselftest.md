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

The dispatches go through `MetalEnc`, never through a kernel name written
here: a second dispatch table in the gate can disagree with the thing under
test and still read green. Bounds are RELATIVE for the same reason an absolute
one is meaningless on a K-deep dot product.
See `metal-gguf-forward-vs-hf` for both, and for what a stale golden set
looks like.

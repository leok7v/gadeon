---
type: File
title: LLM/src/Shared/Sampler.swift
description: Every sampling filter, in a fixed order.
sources:
  - resource: LLM/src/Shared/Sampler.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

top-k, typical, top-p, min-p, XTC, then temperature LAST, so a temperature at
or below zero is the only greedy path. Config resolves user over model over
default before init, and a logit mask lets a grammar forbid tokens without
the sampler knowing. It also holds the per-mode preset matrix and its two
readers, which return nil rather than another model's card. See
`sampler-preset-testing` and `no-cross-model-sampler-fallback`.

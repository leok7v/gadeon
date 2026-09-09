---
type: File
title: LLM/src/Base/Sampler.swift
description: Every sampling filter, in a fixed order.
sources:
  - resource: LLM/src/Base/Sampler.swift
tags: [orientation]
timestamp: 2026-09-05T12:00:00Z
---

top-k, typical, top-p, min-p, XTC, then temperature LAST, so a temperature
at or below zero is the only greedy path. Config resolves user over model
over default before init, and a logit mask lets a grammar forbid tokens
without the sampler knowing. `sample` mutates the logits it is handed, in
place, so the caller's buffer is the per-token scratch; a reader that needs
the raw logits back must use a penalty-free, mask-free sampler, under which
no step mutates. It also holds the per-mode preset matrix and its two
readers, which return nil rather than another model's card.

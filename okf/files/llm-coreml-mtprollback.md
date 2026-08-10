---
type: File
title: LLM/src/CoreML/MtpRollback.swift
description: Rolling one recurrent layer's state back to an accepted
  prefix.
sources:
  - resource: LLM/src/CoreML/MtpRollback.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

It unrolls the per-token recurrence from the verify trunk's own rollback
ingredients, so a single verify pass serves every accepted length and the base
state rolls back with no second trunk pass. That is the whole reason
self-speculation is a win on a weight-read-bound engine.
See `mtp-crossover-and-verify-width`.

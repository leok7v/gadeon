---
type: File
title: LLM/src/SIMD/GDN.swift
description: The gated delta-net layer, one token at a time.
sources:
  - resource: LLM/src/SIMD/GDN.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The per-token recurrence is EXACT, so looping it over a prompt reproduces the
chunked GPU path to floating-point tolerance. That equivalence is what lets
the CPU path stand as an oracle at all.

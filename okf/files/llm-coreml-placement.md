---
type: File
title: LLM/src/CoreML/Placement.swift
description: Where did each op actually land.
sources:
  - resource: LLM/src/CoreML/Placement.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

A per-op compute-device audit, and the only honest answer to whether a
program runs on the Neural Engine: a silent CPU fallback is invisible at
runtime because the tokens are still correct. Same rules as the Python gate,
so the two agree. See `ane-placement-audit` and `decode-placement-100-ne`.

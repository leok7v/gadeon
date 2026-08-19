---
type: File
title: LLM/cli/Q2E8Gate.swift
description: Deterministic Q2 packings on a fixed input, for cross-checking.
sources:
  - resource: LLM/cli/Q2E8Gate.swift
tags: [orientation]
timestamp: 2026-08-18T21:32:34Z
---

`gadeon-cli x --q2e8-gate DIR` writes the input values, the AoS and SoA
packings at both fit settings, their dequantized images, and the FP8 table, so
another implementation can be compared byte for byte rather than by eye.

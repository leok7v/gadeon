---
type: File
title: LLM/src/SIMD/Gemma4Engine.swift
description: The gemma-4 text forward on the CPU.
sources:
  - resource: LLM/src/SIMD/Gemma4Engine.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Sandwich norms around attention and MLP, a per-layer embedding projection,
and a layer scalar, mirroring the reference decoder layer exactly. It is the
oracle for the GPU twin and is itself gated against a Python dump.

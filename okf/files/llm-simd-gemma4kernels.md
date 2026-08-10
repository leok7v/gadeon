---
type: File
title: LLM/src/SIMD/Gemma4Kernels.swift
description: Elementwise, norm and rope primitives for gemma-4.
sources:
  - resource: LLM/src/SIMD/Gemma4Kernels.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Kept apart from the ternary primitives so that path's numerics cannot drift
when these change. Two differences are load bearing: the norm weight arrives
as an array because every gemma norm is BF16, and rope takes a rotated PAIR
COUNT because a head rotates only part of its width.

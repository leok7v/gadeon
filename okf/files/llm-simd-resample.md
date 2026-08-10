---
type: File
title: LLM/src/SIMD/Resample.swift
description: Pillow's separable resampler, because that is what the config
  means.
sources:
  - resource: LLM/src/SIMD/Resample.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

CoreGraphics offers no bicubic at all, and the two bicubics in common use are
DIFFERENT CURVES with different coefficients. Against the reference pixels
this reads about one quantisation step where the system's best reads an order
of magnitude more.

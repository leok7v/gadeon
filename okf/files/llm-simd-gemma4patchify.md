---
type: File
title: LLM/src/SIMD/Gemma4Patchify.swift
description: An image into the patches and positions the tower consumes.
sources:
  - resource: LLM/src/SIMD/Gemma4Patchify.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

NATIVE RESOLUTION is the whole design: the image is resized to the largest
size that fits the patch budget and divides the pooling grid, then padded to
that budget with sentinel positions. So the patch count follows the aspect
ratio, which is where the variable soft-token count comes from.

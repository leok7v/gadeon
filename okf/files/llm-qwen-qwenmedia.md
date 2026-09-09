---
type: File
title: LLM/src/Qwen/QwenMedia.swift
description: The Qwen VL family's attachment encoder over the resident
  Metal tower.
sources:
  - resource: LLM/src/Qwen/QwenMedia.swift
tags: [orientation]
timestamp: 2026-09-05T03:30:00Z
---

An image is resized at its own aspect ratio to the token budget, runs the
backend's vision tower on that grid, and becomes one span of `<|image_pad|>`
rows carrying its grid, which the template's own vision triple already
brackets. A video is sampled at two frames a second, consecutive frames pair
up through the tower's two temporal kernels, and each pair is its own
bracketed span behind a `<t seconds>` stamp, the whole strip replacing the
template's triple. Audio throws a sentence the user can read.

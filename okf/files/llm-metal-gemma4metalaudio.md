---
type: File
title: LLM/src/Metal/Gemma4MetalAudio.swift
description: The gemma-4 audio tower on the GPU.
sources:
  - resource: LLM/src/Metal/Gemma4MetalAudio.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

It mirrors the CPU tower op for op, with that tower as the oracle. Every norm
here is F32 where the text and vision towers carry BF16 ones, so a BF16 norm
kernel pointed at one walks the weight array at half stride and produces
something wrong everywhere and plausible nowhere. The type is asserted rather
than trusted. See `gemma4-port`.

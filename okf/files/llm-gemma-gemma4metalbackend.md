---
type: File
title: LLM/src/Gemma/Gemma4MetalBackend.swift
description: The gemma GPU engine adapted to the backend seam.
sources:
  - resource: LLM/src/Gemma/Gemma4MetalBackend.swift
tags: [orientation]
timestamp: 2026-09-05T05:00:00Z
---

`EngineBackend` over the GPU gemma engine. Soft tokens exist only where
the GPU has matrix units: both towers reach the GPU through the
simdgroup-matrix GEMM, and on a GPU without them building the pipeline
does not fail, it takes the shader compiler service down. That gate costs
the whole A13 line the attachment UI; text is unaffected. A parked
conversation restores through the engine's own `Parked` form, which the
base adapter hands to `adopt`.

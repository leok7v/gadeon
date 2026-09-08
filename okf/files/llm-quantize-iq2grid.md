---
type: File
title: LLM/src/Quantize/IQ2Grid.swift
description: The ggml iq2xxs_grid, ksigns_iq2xs and kmask_iq2xs tables,
  transcribed.
sources:
  - resource: LLM/src/Quantize/IQ2Grid.swift
tags: [orientation]
timestamp: 2026-08-22T18:30:00Z
---

256 uint64 grid entries of eight unsigned magnitudes each, 128 sign patterns,
and the eight bit masks that select one. Unlike the IQ1 grid these values are
NOT ternary, which is why IQ2_XXS costs a multiply per weight where IQ1_S
costs a select. See [[llm-quantize-iq2xxs]].

---
type: File
title: LLM/src/Quantize/Blocks.swift
description: The one ggml-type-to-decoder table, and the predicate every
  dispatch site asks.
sources:
  - resource: LLM/src/Quantize/Blocks.swift
tags: [orientation]
timestamp: 2026-08-22T19:30:00Z
---

`decoder(_:)` is the single place a super-block type is added; `superBlocked`
is the same table read as a yes/no, which is what MetalEnc and GQ branch on.
Duplicating either list is how the gather and the stride both went wrong -
see `a-super-block-type-has-no-per-element-read`.

---
type: File
title: LLM/cli/Splice.swift
description: Rebuilds a GGUF taking named roles from a donor file, so a
  leave-one-out arm costs a copy instead of a 30-minute re-emit.
sources:
  - resource: LLM/cli/Splice.swift
tags: [orientation]
timestamp: 2026-08-22T03:05:00Z
---

`gadeon-cli x --splice <base.gguf> <donor.gguf> <leaf,leaf,...> <out.gguf>`
copies base tensor for tensor, substituting the donor's copy of every tensor
whose leaf key matches, and fails when nothing matched. Metadata and tensor
order come from the base.

Its reason to exist is attribution: with one Q2_E8 emit and one BF16 emit in
hand, "what does quantizing ffn_down alone cost" is a 30-second splice
rather than another convert. See
`perplexity-is-deterministic-and-corpus-bound` for why one scoring run per
arm is enough.

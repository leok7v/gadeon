---
type: File
title: LLM/metal/AssistKernels.metal
description: The drafting head's cheap output projection, over clusters.
sources:
  - resource: LLM/metal/AssistKernels.metal
tags: [orientation]
timestamp: 2026-08-30T02:00:00Z
---

Two kernels: k masked-argmax passes to pick the best clusters out of 2048,
then one gathered dot-and-argmax over only the tokens those clusters own.

The token table stays in ORIGINAL token order, so `token_ordering` yields
token ids directly and the gather is scattered; there is no inverse
permutation to apply. Included by `Kernels.metal` and excluded from compiling
on its own, like `IQTables` and `IQKernels`.
See `the-e4b-drafter-is-cheap-and-mispaired`.

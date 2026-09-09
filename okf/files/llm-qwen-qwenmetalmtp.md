---
type: File
title: LLM/src/Qwen/QwenMetalMTP.swift
description: The nextn drafter block on the GPU, with its own KV pool that
  follows the base position.
sources:
  - resource: LLM/src/Qwen/QwenMetalMTP.swift
tags: [orientation]
timestamp: 2026-09-06T14:00:00Z
---

One attention layer plus the eh_proj/enorm/hnorm/shared_head_norm quartet,
encoded onto a caller's command buffer. It folds a token embedding onto the
base hidden that predicted it and leaves h_nextn for the tied lm_head, so a
draft costs one layer rather than the whole stack. The block is blk.<nLayer>
of the SAME weight file, which is why self-speculation here needs no second
model.

The pool has an `origin`, the absolute position of its row 0, and `end` is
the next position it lacks; the engine cycles only when `end` equals its own
position. `encodeSeed` writes a prefill chunk's rows and `encodeRow` one
token's, both K and V only, since this single layer's K and V depend on
nothing but the row's input.

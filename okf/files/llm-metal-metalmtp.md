---
type: File
title: LLM/src/Metal/MetalMTP.swift
description: The nextn drafter block on the GPU, with its own KV pool.
sources:
  - resource: LLM/src/Metal/MetalMTP.swift
tags: [orientation]
timestamp: 2026-08-20T01:05:00Z
---

One attention layer plus the eh_proj/enorm/hnorm/shared_head_norm quartet,
encoded onto a caller's command buffer. It folds a token embedding onto the
base hidden that predicted it and leaves h_nextn for the tied lm_head, so a
draft costs one layer rather than the whole stack. The block is blk.<nLayer>
of the SAME weight file, which is why self-speculation here needs no second
model. See `mtp-on-metal`.

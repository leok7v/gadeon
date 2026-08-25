---
type: File
title: LLM/cli/Graft.swift
description: The --graft and --drafter subcommands; argument parsing
  over GGUFGraft.
sources:
  - resource: LLM/cli/Graft.swift
tags: [orientation]
timestamp: 2026-08-24T02:20:00Z
---

`gadeon-cli x --graft <text.gguf> <donor.gguf> <out.gguf>`. Thin, like the
other cli entry points: parse three paths, call
[[llm-quantize-ggufgraft]], report tensor and key counts.

It exists so re-tuning a text trunk does not force re-emitting a vision tower
that the tuning never touched -- 21 minutes at 4B, 40 at 9B, 2.4 hours at
27B.

`gadeon-cli x --drafter <text.gguf> <donor.gguf> <out.gguf>` is the same
call with the nextn block's prefixes: tensors `blk.` (which the `!present`
guard narrows to the block the base lacks) and keys `qwen35.block_count` +
`qwen35.nextn_predict_layers`, which must move TOGETHER. It adds MTP to a
third-party file that shipped without it. See `drafter-graft-from-one-shard`.

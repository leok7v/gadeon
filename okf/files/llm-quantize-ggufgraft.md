---
type: File
title: LLM/src/Quantize/GGUFGraft.swift
description: Grafts a vision tower from a donor GGUF onto a text-only one,
  header rewrite plus verbatim tensor copy.
sources:
  - resource: LLM/src/Quantize/GGUFGraft.swift
tags: [orientation]
timestamp: 2026-08-24T02:20:00Z
---

`GGUFGraft.graft(text:donor:to:)` takes the language tensors and KV of one
file, adds the donor's `v.*` / `mm.*` tensors and `clip.*` keys, and writes a
new GGUF. Keys are ordered as the converter orders them, so a diff against a
re-convert stays readable.

The tower is f16/f32 and untouched by quantization, so the result is EXACT:
validated byte-for-byte against a full re-convert of the same weights, 1.7 s
against 21 minutes. It stops at reading raw KV bytes and copying tensor
payloads; it never inspects or rewrites a tensor.

The two prefix lists are PARAMETERS defaulting to the tower's
(`["v.", "mm."]` / `["clip."]`), so the same body grafts the nextn drafter
(`drafter-graft-from-one-shard`). Tensors are add-only -- a name the base
already has is never taken, which is what stops a donor's `output.weight`
from landing twice -- while a listed KEY the base already has is REPLACED by
the donor's, because the donor is the authority on the subsystem it supplies.

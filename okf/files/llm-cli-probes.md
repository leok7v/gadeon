---
type: File
title: LLM/cli/Probes.swift
description: Standalone probe and bench modes.
sources:
  - resource: LLM/cli/Probes.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Each runs before or instead of the chat path and exits the process itself.
Bodies are verbatim blocks in the order main.swift dispatches them.

`--vit` runs `QwenViT` over the exact pixels the numpy reference
preprocessed and cosine-compares the merged embeddings with no LM load;
pixels and reference come from `scripts/convert/qwen35/bonsai27b_vit_ref.py`.

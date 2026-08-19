---
type: File
title: LLM/cli/Perplexity.swift
description: Held-out perplexity over a text corpus, the end-to-end quality
  scalar.
sources:
  - resource: LLM/cli/Perplexity.swift
tags: [orientation]
timestamp: 2026-08-19T16:00:00Z
---

`gadeon-cli <model.gguf> --ppl <corpus.txt> [--ppl-ctx 512] [--ppl-chunks N]`.
Non-overlapping chunks, scoring only the second half of each so every scored
token has context, which is llama.cpp's protocol and makes the numbers
comparable to published ones.

The chunk runs through `MetalEngine.chunkCost` -- one batched forward with the
head suppressed, then the lm_head swept over the scored half. `--ppl-serial`
takes the token-by-token `MetalEngine.step` instead, which is the reference the
batched path is gated against. The corpus must be HELD OUT: scoring on the
Hessian calibration text measures nothing.
See `perplexity-is-the-missing-instrument`.

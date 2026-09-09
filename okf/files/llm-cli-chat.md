---
type: File
title: LLM/cli/Chat.swift
description: The GGUF entry point, the one chat driver every lineage runs
  through, and the ternary bring-up modes that hang off it.
sources:
  - resource: LLM/cli/Chat.swift
tags: [orientation]
timestamp: 2026-09-04T09:30:00Z
---

`runGgufMain` opens a file by its own `general.architecture` into a
`LoadedChat` (backend, template, vocab, presets, encoded attachments), then
`--bench`, `--probe` and `runChat` run over that struct with no idea which
lineage filled it. `runTurn` picks the stream per turn (`img:PATH` through
the model's media encoder at `--image-budget`, the attachment turn's soft
spans, or text) and prints the one metrics line. The ternary-only bring-up
modes (`--kernel-bench`, `--metal-selftest`, `--slugs`, `--hess`, `--imat`,
`--mtp-*`, `--tap`) sit in `runBonsaiModes` and exit themselves. `traceTurn`
is the single place the `--trace` record is shaped.

The `--tap` dump runs the SIMD `QwenEngine` over 64 synthetic tokens from a
fresh reset and taps GDN layer 0's input and output per token, the raw
tensors an offline emit gates against.

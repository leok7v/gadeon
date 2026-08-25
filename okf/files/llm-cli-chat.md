---
type: File
title: LLM/cli/Chat.swift
description: The GGUF chat entry point plus the probes that hang off it, and
  the one place a --trace record is built.
sources:
  - resource: LLM/cli/Chat.swift
tags: [orientation]
timestamp: 2026-08-22T16:45:00Z
---

`runGgufMain` routes a `.gguf` argument: gemma-4 by architecture, then the
bring-up modes (`--ppl`, `--kernel-bench`, `--metal-selftest`, `--slugs`,
`--hess`, `--imat`, `--mtp-verify` / `--mtp-bench`, `--tap`, `--bench`,
`--title`), then the chat loop over `ChatSession`.

TWO TURN FUNCTIONS, not one: `runBonsai` for text and `runBonsaiVision` for
an `img:PATH question` turn -- the GGUF path has no `--vl-image` flag, which
is the CoreML path's spelling. Both build the same accumulators and hand
them to `traceTurn`, which is the single place the JSON record is shaped;
`runSession` on the CoreML side calls the same builder. Keeping the
dictionary in one function is deliberate, so a field added on one path
cannot go missing on the other.

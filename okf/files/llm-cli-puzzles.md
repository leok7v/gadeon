---
type: File
title: LLM/cli/Puzzles.swift
description: Drives the puzzle corpus over a loaded model and rescores a
  saved run.
sources:
  - resource: LLM/cli/Puzzles.swift
tags: [orientation]
timestamp: 2026-08-21T22:30:00Z
---

`--puzzle-gate [label] [--json out]` loads the model ONCE and resets between
items, where the Python it replaces spawned a process per item;
`--puzzle-rescore <runs.json>` re-applies the scorer to a saved run without
decoding anything. See [[llm-shared-puzzlegate]].

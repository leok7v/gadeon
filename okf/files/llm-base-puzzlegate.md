---
type: File
title: LLM/Base/PuzzleGate.swift
description: The reasoning corpus and the scoring rule that decides PASS,
  FAIL or INCOMPLETE.
sources:
  - resource: LLM/Base/PuzzleGate.swift
tags: [orientation]
timestamp: 2026-08-21T22:30:00Z
---

Six short single-answer puzzles and the scorer both the CLI gate and the
unit tests read, so neither carries its own copy. The scoring rule is the load-bearing part: accept-only,
markdown stripped, and only the tail after the reasoning is scored.

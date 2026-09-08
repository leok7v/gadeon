---
type: File
title: LLM/cli/Replay.swift
description: Scoring an instruction model on its own output, because wikitext
  is out of distribution for it.
sources:
  - resource: LLM/cli/Replay.swift
tags: [orientation]
timestamp: 2026-08-29T20:30:00Z
---

`--replay-make` renders prompts through the chat template, generates greedily
and freezes prompt+reply token ids with the index where the reply starts.
`--replay` teacher-forces any model over those exact ids, scoring only from
that index, and reuses the top-64 dump format so KL, top1/top5 and tau come
out unchanged. See `wikitext-cannot-score-an-rl-tuned-gemma`.

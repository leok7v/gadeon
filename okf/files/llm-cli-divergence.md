---
type: File
title: LLM/cli/Divergence.swift
description: Scores a student against a cached BF16 teacher -- KL, top-1
  and top-5 agreement, Kendall tau over the top-64, and max |dp|.
sources:
  - resource: LLM/cli/Divergence.swift
tags: [orientation]
timestamp: 2026-08-22T04:35:00Z
---

`--kld-dump <top.bin>` records the teacher's top-64 ids, logits and
logsumexp per scored position, ~4 MB for 8192 positions; `--kld <top.bin>`
replays the same chunks on another model and reports the divergence beside
its perplexity. Caching the teacher is what makes an arm cost one 4B run
rather than a 4B and a 7.6 GB BF16 run side by side.

Two shapes here are load-bearing. `rank` selects the top-64 by insertion
against a running floor rather than sorting: a full sort of the 248320-wide
vocabulary at every scored position is 36G comparisons per run. And the
divergence is read on the TEACHER's top-64, so the student is never asked
which tokens it would have ranked instead -- the tail is one lumped bucket
and the covered mass is reported beside the number so the approximation
stays visible. See
`perplexity-is-deterministic-and-corpus-bound`.

---
type: File
title: LLM/cli/Assist.swift
description: The two measurements that say whether the drafting head pays.
sources:
  - resource: LLM/cli/Assist.swift
tags: [orientation]
timestamp: 2026-08-29T02:30:00Z
---

`--assist-probe` scores the head's greedy top-1 against the trunk's own next
token; `--assist-bench` runs draft/verify/rollback and reports t/s against
plain decode. Both render the chat template, because an untemplated -it
prompt measures degenerate text.

`--assist-bench --sampler` installs the file's own preset, one fresh sampler
per arm so both draw the same RNG stream. Without it the bench measures a
path the app never runs (`spec-decode-is-exact-under-sampling`).

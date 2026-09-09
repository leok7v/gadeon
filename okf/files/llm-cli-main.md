---
type: File
title: LLM/cli/main.swift
description: The CLI entry point, its flag prologue and its mode dispatch.
sources:
  - resource: LLM/cli/main.swift
tags: [orientation]
timestamp: 2026-09-04T09:30:00Z
---

A multi-turn chat over any GGUF: `gadeon model.ggxf "turn" "turn"`, or
turns from stdin lines. Reasoning streams to stderr and the answer to
stdout. `-n N` caps a turn, `--think` (or `--reasoning-effort on|<level>`)
turns thinking on, `--soft-reasoning` / `--max-reasoning` set the caps the
app derives from the measured rate, `--system PROMPT` (`@path` reads a
file) sets the system message with NO date tail, `--wiki-model PATH`
enables the wikipedia and network tools (without it only the local
get_current_time and calculator are advertised), `--precook FILE` primes
or cooks the prefix, `--title` and `--hint` run the meta turns, `--greedy`
and `--seed N` steer sampling, and `--image-budget N` (default 280) is the
soft-token budget an `img:PATH` turn spends on its picture. The offline modes (`--meta`, `--graft`,
`--assist*`, `--splice`, `--replay*`, `--kld`, `--puzzle-*`) run first
and exit.

Fixture defaults resolve from the REPO ROOT, not from the package directory.

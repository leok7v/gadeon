---
type: File
title: LLM/src/Base/Flags.swift
description: The one registry a launch knob is declared in, and the
  argv > UserDefaults > default resolution every reader of it shares.
sources:
  - resource: LLM/src/Base/Flags.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

Every knob this repo used to reach only through an environment variable
(Metal/SIMD engine tuning, diagnostic logging, CLI-only overrides) is now
one `Knob` in `Flags.registry`: its kebab-case argv name, whether it takes
a value, a help string, a scope, and a default. `Flags.value` / `.int` /
`.double` / `.uint64` / `.on` resolve argv, then a `Flags.store`d
`UserDefaults` write, then the registry default -- no environment
fallback. See `a-knob-is-a-flag-not-a-variable` for why, the arity-aware
`--name value` / `--name=value` parse, and the three knobs whose semantics
needed more than a straight bool swap.

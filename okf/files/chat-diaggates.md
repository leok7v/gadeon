---
type: File
title: Chat/DiagGates.swift
description: Which diagnostic channels write to the log, and how each one
  is asked for.
sources:
  - resource: Chat/DiagGates.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

One case per channel, each carrying its own label, its explanation, and its
`Flags` registry answer (argv, then Settings' stored value, then the
registry default). Settings > Diagnostics renders the list straight off
`allCases`, so a new channel is a case here and nothing else.
See `a-knob-is-a-flag-not-a-variable` for the registry and
`a-diagnostic-gate-is-a-setting-not-a-variable` for why a gate has a
Settings toggle at all.

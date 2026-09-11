---
type: File
title: LLM/Base/DiagGate.swift
description: Which diagnostic categories write to the log, and how each one
  is asked for.
sources:
  - resource: LLM/Base/DiagGate.swift
tags: [orientation]
timestamp: 2026-09-08T23:30:00Z
---

One case per category plus `fault`, which is never gated. Each carries its
own label, its explanation, and its `Flags` registry answer (argv, then
Settings' stored value, then the verbosity ladder). `debug` is the master
and resolves once at launch. Settings > Diagnostics renders `switchable`, so
a new category is a case here and nothing else. It lives beside `Flags` and
`Diag` because the cli links no `Chat`, and a gate the engine cannot see is
a gate nothing in `LLM/` can use.

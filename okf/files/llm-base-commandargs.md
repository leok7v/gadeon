---
type: File
title: LLM/Base/CommandArgs.swift
description: One owner of a command line, so what a flag took is known
  rather than reconstructed.
sources:
  - resource: LLM/Base/CommandArgs.swift
tags: [orientation]
timestamp: 2026-09-02T13:00:00Z
---

A class, not a struct: every CLI mode reads from the one instance, and a
value taken deep in a mode has to be missing from the leftovers back at the
top. That is the whole point -- `turns` is computed from what nothing
claimed, so no mode has to work out which words a flag ate.

Three rules the callers rely on. A flag's value is never another flag, so a
flag missing its argument reads nothing rather than eating what follows. A
number is taken only when it PARSES, so a malformed one leaves its word
alone. And `@path` is resolved in one place, so a flag cannot be the one
that forgot.

Lives in the library rather than in `LLM/cli/` only because that is what
makes it testable, the same reason `PuzzleGate` is here.

A name the `Flags` registry also owns (`--spec-n`, `--seed`) is never parsed
here twice: `consume(_:hasValue:)` only reserves the token so it does not
leak into `turns`, and the value itself comes from `Flags`.

---
type: File
title: LLM/src/Base/Diag.swift
description: The one diagnostics sink, shared by App and LLM.
sources:
  - resource: LLM/src/Base/Diag.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Every report carries a `diagnostics-are-gated-by-debug` category and lands,
when that category is on, in the unified log, on stderr, and in a per-run file
under Caches tagged with a timestamp and the caller's file and line. The file
is created on the first line that passes the gate, so a run with Debug off
writes none. One file per launch, pruned after about a day.
See `ios-stderr-write-crash` and `devicectl-read-device-files`.

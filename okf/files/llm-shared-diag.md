---
type: File
title: LLM/src/Shared/Diag.swift
description: The one diagnostics sink, shared by App and LLM.
sources:
  - resource: LLM/src/Shared/Diag.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Every report lands on stderr, in the unified log, and in a per-run file under
Caches tagged with a timestamp and the caller's file and line. One file per
launch, pruned after about a day.
See `ios-stderr-write-crash` and `devicectl-read-device-files`.

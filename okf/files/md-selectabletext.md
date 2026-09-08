---
type: File
title: MD/src/SelectableText.swift
description: Atomic units, and the incremental splice that streams into
  them.
sources:
  - resource: MD/src/SelectableText.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Atomic spans mark what selection should snap around rather than cut through.
The incremental apply keeps the shared attributed prefix and suffix and
replaces only the middle, so a streaming update re-lays out proportional to
the delta and a selection outside the edit survives.

The prefix comparison is ATTRIBUTE aware on purpose: closing a markdown span
changes attributes on characters already emitted, so a string-only diff
leaves them stale. See `md-single-surface-transcript`.

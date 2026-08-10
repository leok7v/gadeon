---
type: File
title: MD/src/Parser.swift
description: The batch parser.
sources:
  - resource: MD/src/Parser.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

It carries per-column table alignment and an internal seam the streaming
parser reuses, so streaming and batch results cannot diverge. Reference link
definitions ride a task-local so the streaming parser can inject what it has
accumulated so far.

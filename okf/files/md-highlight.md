---
type: File
title: MD/Highlight.swift
description: Regex-driven syntax highlighting.
sources:
  - resource: MD/Highlight.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Language specs and both colour themes live in a bundled ini rather than in
code. Spans are mask tracked, so an earlier and more specific match is never
re-coloured by a later one.

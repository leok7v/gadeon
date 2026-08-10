---
type: File
title: MD/src/TableMetrics.swift
description: Shared column-width maths for every table renderer.
sources:
  - resource: MD/src/TableMetrics.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

And for the monospaced copy serialisation, so a copied table matches the
rendered one. Widths use a sqrt-damped character count so a wide column does
not starve the narrow ones.

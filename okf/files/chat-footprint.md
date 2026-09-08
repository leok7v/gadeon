---
type: File
title: Chat/Footprint.swift
description: What the app costs in memory, in the terms iOS kills on.
sources:
  - resource: Chat/Footprint.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

phys_footprint rather than resident size, because jetsam decides on
footprint and the two differ by exactly what this app leans on: weights
mmap'd read-only are clean file-backed pages. Each report carries footprint,
resident, the dirty split and the file-backed part.
See `gpu-weights-are-wired` for the case this measurement CANNOT see.

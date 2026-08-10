---
type: File
title: MD/src/ImagePrefetch.swift
description: Fit an image into a box.
sources:
  - resource: MD/src/ImagePrefetch.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Explicit dimensions win; otherwise the intrinsic size is scaled and clamped
with the aspect preserved. Shared by every renderer and export path, so they
cannot disagree about size.

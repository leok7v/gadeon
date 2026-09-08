---
type: File
title: App/Cover-iOS.swift
description: Present a view full-screen with no app chrome behind it.
sources:
  - resource: App/Cover-iOS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

fullScreenCover. The macOS twin uses a sheet, so the shared call site stays
free of a #if os.

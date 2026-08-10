---
type: File
title: App/Composer-macOS.swift
description: The in-card model picker as a borderless label on macOS.
sources:
  - resource: App/Composer-macOS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

.borderlessButton is a macOS-only menu style, so it lives in an SDK-split
modifier matched symbol for symbol by the iOS twin rather than in a #if os
branch inside the shared Composer.

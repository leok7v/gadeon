---
type: File
title: App/AppFont-macOS.swift
description: A text style at the app's text size, on macOS.
sources:
  - resource: App/AppFont-macOS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

AppKit has no content size category, so the base point size is fixed and this
scale is the only thing that moves it. That is the whole reason the app
carries a multiplier at all.

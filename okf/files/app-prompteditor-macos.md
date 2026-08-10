---
type: File
title: App/PromptEditor-macOS.swift
description: The prompt editor on macOS.
sources:
  - resource: App/PromptEditor-macOS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

An NSTextView rather than a vertical TextField, which appends a Shift+Return
newline at the string END instead of the caret and scrolls choppily. Return
submits and Shift+Return breaks at the caret.

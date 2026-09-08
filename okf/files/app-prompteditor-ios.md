---
type: File
title: App/PromptEditor-iOS.swift
description: The prompt editor on iOS.
sources:
  - resource: App/PromptEditor-iOS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

A UITextView inserts at the caret, scrolls smoothly, and grows between a
minimum and maximum line count. The soft keyboard has no Shift+Return, so
Return inserts a newline and Send submits. Focus comes in as a
`FocusRequest` and goes out as the `editing` report, one writer each, see
`the-composer-focus-is-a-request-and-a-report`; a paste at or over
`ChatModel.pasteAttachBytes` is refused by the should-change delegate and
becomes a Text attachment instead (`attachment-inline-refs`).
See `attributegraph-cycle-responder`.

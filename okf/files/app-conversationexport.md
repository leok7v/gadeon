---
type: File
title: App/ConversationExport.swift
description: One saved conversation as a PDF, and the file document both
  export surfaces share.
sources:
  - resource: App/ConversationExport.swift
tags: [orientation]
timestamp: 2026-08-11T04:00:00Z
---

`ConversationPDF` is the Transferable a ShareLink takes; its exporting
closure is async, so nothing renders until a destination is chosen.

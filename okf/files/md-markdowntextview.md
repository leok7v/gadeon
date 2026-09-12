---
type: File
title: MD/MarkdownTextView.swift
description: The whole document as one selectable native surface.
sources:
  - resource: MD/MarkdownTextView.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Drag selection snaps around atomic code, table and image units, and images
are prefetched and embedded as text attachments. Passing a find controller
enables find. It reads its own width off the geometry and hands it to the
builder, which is what lets a table solve its columns.

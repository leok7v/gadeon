---
type: File
title: MD/src/TableView.swift
description: A GFM table in SwiftUI, with per-column alignment.
sources:
  - resource: MD/src/TableView.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Natural widths are measured from the rendered cells and redistributed when
they exceed the container, and the cells WRAP. Horizontal scrolling is a last
resort kept only for the transient first layout before the width is known.

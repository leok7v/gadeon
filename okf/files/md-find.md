---
type: File
title: MD/src/Find.swift
description: Find across the transcript's per-message text surfaces.
sources:
  - resource: MD/src/Find.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The app registers each bubble's text view and the transcript order; this
highlights every match in every view, tracks ONE cursor across them, and
scrolls to the active match's message before selecting it there. A single
registered view degenerates to plain single-view find.

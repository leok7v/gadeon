---
type: File
title: MD/src/Stream.swift
description: The incremental parser for live output.
sources:
  - resource: MD/src/Stream.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

One instance per channel. Appending costs work proportional to the open block
rather than the whole document: boundaries are re-derived over the unsealed
tail only, and a block seals once a later block proves its boundary settled.
So finishing equals a batch parse of everything appended.

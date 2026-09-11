---
type: File
title: App/AppTests/SentencesTests.swift
description: The sentence splitter's cut rules, pinned.
sources:
  - resource: App/AppTests/SentencesTests.swift
tags: [orientation]
timestamp: 2026-09-08T00:00:00Z
---

Terminators and newlines cut, abbreviations, list markers and decimals do
not, closing quotes do not hide a mark, a runaway line is capped, and the
last piece is the partial one whose predecessors never move.

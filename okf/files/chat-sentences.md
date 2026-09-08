---
type: File
title: Chat/Sentences.swift
description: The reasoning stream cut into sentences for the marquee.
sources:
  - resource: Chat/Sentences.swift
tags: [orientation]
timestamp: 2026-09-08T00:00:00Z
---

A byte scanner over a growing text: a newline or a terminated word before
whitespace ends a piece, list markers, decimals, initials and a short
abbreviation list do not, and a piece is cut at the first whitespace past
240 bytes so a code line or URL never becomes one enormous line. Only
whitespace cuts, so the last piece is the one still being written and the
earlier ones keep their indices. See `the-thinking-marquee-runs-to-rest`.

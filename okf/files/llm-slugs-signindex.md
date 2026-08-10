---
type: File
title: LLM/src/Slugs/SignIndex.swift
description: A 384-d embedding turned into a short list of matching
  articles.
sources:
  - resource: LLM/src/Slugs/SignIndex.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

A projection to fewer dimensions, one bit per sign, then a Hamming match
against every article key. The index rides a trailer appended to the GGUF,
which ordinary readers ignore.

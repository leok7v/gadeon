---
type: File
title: LLM/src/Shared/GGUFWriter.swift
description: A GGUF v3 container writer, the mirror of the reader.
sources:
  - resource: LLM/src/Shared/GGUFWriter.swift
tags: [orientation]
timestamp: 2026-08-18T21:32:34Z
---

declare() every tensor, finishHeader(), then append() them in the declared
order. append validates the name and the byte count, and close() throws when
a declared tensor was never written, so a truncated or misordered emit cannot
reach disk quietly.

Byte sizes come from GGUF.rowByteCount, shared with the reader, so the two
sides of a private block type cannot drift. See `aneq-archive`.

---
type: File
title: LLM/src/Shared/ChatWire.swift
description: The chat markup a model's own template speaks, derived by
  diffing it.
sources:
  - resource: LLM/src/Shared/ChatWire.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Rendering an assistant message that carries reasoning against the same
message carrying none leaves exactly the marker bytes behind. Never matched
against a model name and never looked up in a table of dialects, because
every lineage spells the boundaries differently and two of them differ by one
byte while meaning different things.

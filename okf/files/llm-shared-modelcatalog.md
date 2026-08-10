---
type: File
title: LLM/src/Shared/ModelCatalog.swift
description: Where each offered model comes from, and what it costs.
sources:
  - resource: LLM/src/Shared/ModelCatalog.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Multi-file CoreML sets and single-file ternary GGUFs alike, both downloading
through the same flow into a sha-pinned directory so a later Hub push cannot
swap weights under a released binary. Backend agnostic, which is why it lives
here rather than under CoreML.

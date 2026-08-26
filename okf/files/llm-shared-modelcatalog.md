---
type: File
title: LLM/src/Shared/ModelCatalog.swift
description: Where each offered model comes from, and what it costs.
sources:
  - resource: LLM/src/Shared/ModelCatalog.swift
tags: [orientation]
timestamp: 2026-08-26T02:30:00Z
---

One entry per offered model: HF repo, a PINNED commit sha, the on-disk byte
count the consent prompt shows, and the single GGUF file to fetch. Everything
lands in modelStore/<name>/<sha>/, so a later Hub push cannot swap weights
under a released binary. Which of these the picker offers, and at what
installed-RAM floor, is `App/Bundle+Models.swift`, not here.

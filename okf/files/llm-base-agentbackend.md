---
type: File
title: LLM/src/Base/AgentBackend.swift
description: The seam every turn loop renders and decodes against.
sources:
  - resource: LLM/src/Base/AgentBackend.swift
tags: [orientation]
timestamp: 2026-09-05T12:00:00Z
---

The protocol ChatSession speaks and every compute backend implements, plus
the message and metrics model they carry, so the concrete engine stays behind
it. Async where the real backend is actor isolated. `extendSoft` is the one
attachment path: the turn's ids with every span already expanded, and the
tower rows to lay over the placeholder positions; the default throws, and
the lineage adapters over `EngineBackend` override it where they have a
tower. `supportsVision` is the tower gate itself, which the Qwen adapter
also answers `supportsSoftTokens` with.

Its serializeState default is a silent no-op and a backend that inherits it
re-prefills forever. See `never-reprefill`.

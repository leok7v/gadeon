---
type: File
title: LLM/src/Shared/AgentBackend.swift
description: The seam every turn loop renders and decodes against.
sources:
  - resource: LLM/src/Shared/AgentBackend.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The protocol ChatSession speaks and every compute backend implements, plus
the message and metrics model they carry, so the concrete engine stays behind
it. Async where the real backend is actor isolated.

Its serializeState default is a silent no-op and a backend that inherits it
re-prefills forever. See `never-reprefill`.

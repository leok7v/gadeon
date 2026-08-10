---
type: File
title: LLM/src/Shared/ChatSession.swift
description: The multi-turn loop that seeds a turn, decodes it, and runs
  its tools.
sources:
  - resource: LLM/src/Shared/ChatSession.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

It owns the conversation's marks, renders only the delta each turn and
appends it, splits reasoning from answer using the wire the template
declares, drives the tool rounds, and parks and restores state.

It stops at the backend protocol. No model, no KV layout and no CoreML or
Metal call is visible here.
See `continuation-append-render`, `agentic-tool-loop-landed` and
`never-reprefill`.

---
type: File
title: LLM/src/Shared/StateBytes.swift
description: The wire format a parked conversation is written in.
sources:
  - resource: LLM/src/Shared/StateBytes.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Little-endian counts and f32 payloads, so a state file means the same thing
whatever an engine stores internally. One definition because three engines
write it, and it carries no geometry of its own since a file is only ever read
back by the same model. See `never-reprefill`.

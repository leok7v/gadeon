---
type: File
title: Chat/TraceFile.swift
description: A plain-text mirror of the session trace.
sources:
  - resource: Chat/TraceFile.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

Every event appends as a timestamped line with its payload indented under it,
so the file can be pasted instead of screenshotted. Overwritten each launch,
and constructed only when the transcript gate is on, because its content is
the user's own conversation.

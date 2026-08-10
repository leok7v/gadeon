---
type: File
title: App/TraceFile.swift
description: A plain-text mirror of the session trace.
sources:
  - resource: App/TraceFile.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Every event appends as a timestamped line with its payload indented under it,
so the file can be pasted instead of screenshotted. Overwritten each launch,
and NOT constructed in a Release build, because its content is the user's own
conversation.

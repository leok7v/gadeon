---
type: File
title: MD/src/Diag.swift
description: A diagnostics sink the host app wires in.
sources:
  - resource: MD/src/Diag.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

So render timing lands in the same pullable log as everything else. Absent by
default and frame-budget gated, so steady rendering stays silent and only
stalls surface.

---
type: File
title: App/DiagGates.swift
description: Which diagnostic channels write to the log.
sources:
  - resource: App/DiagGates.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Environment switches declared in project.yml, so turning one on is a
checkbox in the scheme editor rather than an edit here. The two loudest
default off; they were 91% of one captured session and are already rate
limited, so this is a gate rather than a throttle.

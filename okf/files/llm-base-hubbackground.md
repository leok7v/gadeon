---
type: File
title: LLM/Base/HubBackground.swift
description: The iOS background-session download path and the piece
  assembler that keeps resume equal to the .part file's length.
sources:
  - resource: LLM/Base/HubBackground.swift
tags: [orientation]
timestamp: 2026-08-24T18:35:00Z
---

`Relay` owns the one background `URLSession`, whose identifier must be
stable because iOS relaunches the app against it. `drain` is the polling
loop that tops up a bounded window of ranged tasks and assembles what lands.

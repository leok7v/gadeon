---
type: File
title: LLM/src/Shared/VisionPositions.swift
description: Where an image sits in the position stream, for mrope.
sources:
  - resource: LLM/src/Shared/VisionPositions.swift
tags: [orientation]
timestamp: 2026-08-26T02:00:00Z
---

An image occupies one scalar step per merged row rather than one per token,
so the text around it has to be numbered as though the picture were short.
This plans that numbering for a run of text and images together.

---
type: File
title: LLM/src/Shared/ChatTemplate.swift
description: A conversation rendered through the model's own jinja
  template.
sources:
  - resource: LLM/src/Shared/ChatTemplate.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

It bridges messages and tool specs into the shape the template walks and
drives the jinja host callbacks over an interned node tree. renderPrompt is
the pure, engine-free core that both the delta render and the offline gates
call. See `multiturn-boundary-thinktag`.

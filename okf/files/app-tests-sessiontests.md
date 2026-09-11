---
type: File
title: App/AppTests/SessionTests.swift
description: The driver's turn lifecycle, hosted by the app target.
sources:
  - resource: App/AppTests/SessionTests.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

Drives `Chat.Session.sendText` through a mock `AgentBackend` (the same
script shapes `LLM/LLMTests/ChatSessionTests.swift` drives `ChatSession`
with directly, one layer up): a plain turn, a turn stopped mid-prefill, an
empty-decode turn that ends answerless, and a tool round reaching the event
stream. Uses the `installBackend`/`toolRunnerOverride` test seams -- The
mock's pause gate is stored under a lock before the pause is visible:
publishing `isPaused` first let a `release` racing the pause find no
continuation, which hung a run for 57 minutes once and crashed the test host
another time (task #25).

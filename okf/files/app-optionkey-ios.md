---
type: File
title: App/OptionKey-iOS.swift
description: There is no Option key on iOS.
sources:
  - resource: App/OptionKey-iOS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The identity modifier that keeps the shared call site compiling. On a
phone the same flag is raised by a two-second press on the New Chat button
(`ChatModel.revealHidden`), with Debug on, which opens Settings at Misc,
and lowered when Settings closes; the key monitor never fires here.

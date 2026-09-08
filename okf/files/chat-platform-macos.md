---
type: File
title: Chat/Platform-macOS.swift
description: isOS, false on this SDK.
sources:
  - resource: Chat/Platform-macOS.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

Duplicates `App/App-macOS.swift`'s `isOS`: `Bundle+Models.swift` and
`Footprint.swift` need it and `Chat` cannot depend on `App`. Paired with
`Platform-iOS.swift`; `config/platform.xcconfig` excludes one by SDK.

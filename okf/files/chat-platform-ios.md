---
type: File
title: Chat/Platform-iOS.swift
description: isOS, true on this SDK.
sources:
  - resource: Chat/Platform-iOS.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

Duplicates `App/App-iOS.swift`'s `isOS`: `Bundle+Models.swift` and
`Footprint.swift` need it and `Chat` cannot depend on `App`. Paired with
`Platform-macOS.swift`; `config/platform.xcconfig` excludes one by SDK.

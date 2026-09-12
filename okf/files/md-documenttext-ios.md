---
type: File
title: MD/DocumentText-iOS.swift
description: Tables in the flattened document, on iOS.
sources:
  - resource: MD/DocumentText-iOS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Tab stops, because UIKit has no text table. They are absolute locations, so
this builder has to be told the width it is building for and breaks its own
cells:

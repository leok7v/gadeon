---
type: File
title: LLM/metal/Kernels.metal
description: Every GPU kernel both Metal engines dispatch.
sources:
  - resource: LLM/metal/Kernels.metal
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Compiled at build time into the default library, so there is no runtime
shader compile. Each kernel has a pure-Swift counterpart that stands as its
oracle. See `mm-kernel-multiblock`.

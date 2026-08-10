---
type: File
title: LLM/src/Shared/VideoFrames.swift
description: Frames sampled from a video, each with its timestamp.
sources:
  - resource: LLM/src/Shared/VideoFrames.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The frame count is a parameter, because it belongs to a model's processor
rather than to video. Fewer frames than asked for is not an error here, since
the soft-token count is already variable per item.

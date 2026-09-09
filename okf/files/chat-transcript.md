---
type: File
title: Chat/Transcript.swift
description: The transcript value types both the driver and the view model
  read and build.
sources:
  - resource: Chat/Transcript.swift
tags: [orientation]
timestamp: 2026-09-05T00:30:00Z
---

`Message`, `Doc`, `DocRef`, `ImageAttachment`, `ClipAttachment` and
`ToolRound`, moved out of `ChatModel` so the driver's persistence and turn
code can read and build them without depending on the App target. A
`ClipAttachment` resolves itself through any `MediaEncoder`. An image or
clip attachment carries two names: `name` is the per-chat ordinal the prompt
refers to and `file` the basename it came from. A message's `clips` are the
live turn's playable files and its `posters` the frame that stands for each
once the file is gone,

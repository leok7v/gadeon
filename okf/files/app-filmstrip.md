---
type: File
title: App/FilmStrip.swift
description: The frames of an attached video, playing while the model
  encodes them.
sources:
  - resource: App/FilmStrip.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

It sits on the composer's bottom edge at a fixed height, so the card growing
as someone types never resizes it. A new frame flips one state value and the
cross-fade is an opacity, which is about 32 state changes for the whole show
rather than a redraw every display refresh.
See `voice-turn-and-render-stalls` for why that distinction is load bearing.

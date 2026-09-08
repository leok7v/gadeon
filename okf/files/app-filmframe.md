---
type: File
title: App/FilmFrame.swift
description: One video frame as it appears behind the composer.
sources:
  - resource: App/FilmFrame.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The look alone: a mask gradient, a dim level and a fit, kept apart from
FilmStrip so it can be put in front of a renderer, which a view wired to
ChatModel cannot be. Frames compare by INDEX, since two frames of one clip
can be identical without being the same frame. See `composer-film-strip`.

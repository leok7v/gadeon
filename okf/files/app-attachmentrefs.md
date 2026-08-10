---
type: File
title: App/AttachmentRefs.swift
description: An attachment as an inline @name in the prompt text.
sources:
  - resource: App/AttachmentRefs.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Two invisible sentinels bracket the reference so it renders as plain text yet
stays a detectable, removable unit. Deleting the reference drops the
attachment, and at send each document reference is replaced in place by the
file's content. See `attachment-inline-refs`.

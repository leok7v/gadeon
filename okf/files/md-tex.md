---
type: File
title: MD/src/TeX.swift
description: Inline maths, mapped to Unicode rather than laid out.
sources:
  - resource: MD/src/TeX.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Dollar and backslash delimited spans are split out, then LaTeX tokens become
Greek letters, operators, scripts and simple fractions. Complex layouts
degrade to readable inline text rather than failing.

It is not a layout engine and is not meant to become one.

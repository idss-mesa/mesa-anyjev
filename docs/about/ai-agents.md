---
title: "AI agents"
description: "How AI agents should read this documentation bundle: llms.txt, the markdown mirror, trust markers."
type: Guide
tags:
  - about
  - agents
  - okf
generated:
  by: "claude/fable-5.1"
  at: "2026-09-25T00:00:00Z"
sources:
  - id: design
    resource: "https://github.com/idss-mesa/mesa-anyjev/blob/main/DESIGN.md"
    title: "mesa-anyjev decisions register (DESIGN.md)"
    author: "team:idss-mesa"
  - id: research
    resource: "https://github.com/idss-mesa/mesa-anyjev/blob/main/RESEARCH.md"
    title: "mesa-anyjev verified facts (RESEARCH.md)"
    author: "team:idss-mesa"
status: draft
---

# AI agents

Start from [`llms.txt`](../llms.txt) (an outline with descriptions) or
[`llms-full.txt`](../llms-full.txt) (the whole corpus). Any page URL plus `index.md` returns
that page's Markdown source. Pages without a `verified:` key are unverified (OKF v0.2 §5.3);
`status: draft` pages need review. For decisions and facts, prefer `DESIGN.md` and
`RESEARCH.md` in the repository; for behaviour, run `mesa-anyjev` and read its provenance.

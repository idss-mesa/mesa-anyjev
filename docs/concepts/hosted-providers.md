---
title: "Hosted providers"
description: "Hosted Jev through MotherDuck prompt_jev as a third provider for the same questions, and the data-residency switch that keeps it off by default."
type: Guide
tags:
  - concepts
  - motherduck
  - hosted
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

# Hosted providers

MotherDuck's `prompt_jev` SQL function answers the same three primitives (choice, score,
yes/no) and returns a choice, a full probability distribution and a confidence (DESIGN D16).
mesa-anyjev can ask it the identical questions, using the frozen wording and option lists as
SQL constants, and record the answers with `level='none'` and `calibration='typesafe'`.

Because the state leaves the host, hosted providers are off by default
(`policy.hosted_providers: off`). With `allowlist`, a project must carry
`mesa.hosted_inference=allow` on its root or be listed in `hosted_allow_project_roots`;
runs record `data_left_host`. Hosted decisions may be proposed but never auto-write until a
bench row with its own results file cites their coverage at 5% risk. This provider ships in
milestone M4b.

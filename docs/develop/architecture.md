---
title: "Architecture"
description: "The mesa-anyjev modules, the annotate and apply phases, and the boundaries with AnyJev, mesa-mcp and mesa-ducklake."
type: Reference
tags:
  - develop
  - architecture
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

# Architecture

Two phases (DESIGN D10): `annotate` decides and proposes (no writes); `apply` writes the
accepted AVUs to iRODS through mesa-mcp's shared helpers and records one mesa-ducklake
snapshot per run and path. Decisions are written to the sidecar before any iRODS write.

Layers: backends (AnyJev's protocol: fake, gateway, local weights, composite) below
providers (`DecisionRecord` with level and calibration) below the pipeline; the planner is a
separate role that only proposes. `mesa_mcp.ols` provides the OLS client and the canonical
AVU transform; `mesa_ducklake.DuckLakeClient` records snapshots.

The code map is in `CLAUDE.md`; the decisions in `DESIGN.md`; the facts in `RESEARCH.md`.

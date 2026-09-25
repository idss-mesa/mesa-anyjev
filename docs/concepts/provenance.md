---
title: "Provenance"
description: "The mesa-anyjev sidecar: runs, decisions, options, groups, AVU links, human overrides and labels, and how it joins the mesa-ducklake AVU history."
type: Reference
tags:
  - concepts
  - provenance
  - ducklake
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

# Provenance

Decision provenance is a sidecar owned by mesa-anyjev (DESIGN D4): Postgres schema
`mesa_anyjev` (its own migrations, `src/mesa_anyjev/provenance/migrations/`) or a DuckDB file
for development. It is never a column on mesa-ducklake's `avu_changes` (the AVU triple is a
hard contract) and never a table in schema `mesa`.

Tables: `runs`, `decisions` (one row per answered question with state, provider, level,
calibration, probabilities, thresholds in force and outcome), `decision_options`,
`decision_groups` (a ranking over candidates), `avu_links` (the AVU a decision produced and
its write status), `human_overrides`, `labels`.

The join into the AVU history is `(project_id, snapshot_id, irods_path, attribute, value,
unit)`; `snapshot_id` is null for decisions and links that wrote nothing. Every AVU written by
mesa-anyjev also carries a per-row `source` tag of the form `mesa-anyjev:<tool>:<outcome>` in
mesa-ducklake, so `get_history` shows which rows were written automatically and which by a
curator without a join.

Decisions are never updated; only a link's write status and snapshot id, and a run's status,
change. Corrections are new rows.

## Postgres and exports

`postgresql://` DSNs use the packaged migration (`migrations/0001_mesa_anyjev.sql`, schema
`mesa_anyjev`, JSONB, foreign keys, `NULLS NOT DISTINCT` uniqueness) through a psycopg store
with the same protocol as the DuckDB file; `mesa-anyjev provenance migrate --dsn ...` applies
it, and the store may share a database with mesa-ducklake's `mesa` schema (D4). The
`requires_postgres` test tier runs the DuckDB round trip against a real server.

`mesa-anyjev provenance export --run-id <id> --out <project>/.mesa/anyjev` writes `runs`,
`decisions`, `decision_groups` and `avu_links` as Parquet (never under `.mesa/ducklake/`).
`provenance reconcile --run-id <id> --local-ducklake <dsn>` repairs links whose snapshot id
is NULL because the mirror step failed after the iRODS write, by matching the DuckLake
history rows whose source starts with `mesa-anyjev:` on the exact AVU triple.

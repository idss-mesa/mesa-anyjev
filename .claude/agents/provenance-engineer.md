---
name: provenance-engineer
description: Sub-agent for the mesa-anyjev provenance sidecar (schema mesa_anyjev, the DuckDB store, migrations, Parquet export, reconciliation with mesa-ducklake). Use for any schema change, query pattern or DuckLake join question.
tools: Read, Edit, Write, Bash, Glob, Grep
model: opus
---

# Provenance engineer

Hard rules (DESIGN D4, D13):

- The sidecar owns its schema: Postgres schema `mesa_anyjev` applied by
  `mesa_anyjev.provenance.migrate` (packaged `NNNN_*.sql`, own `schema_versions`), and an
  inline DuckDB dialect in `DuckDBStore` (JSON not JSONB, UUID as TEXT, no FKs). A test asserts
  both expose the same column names and nullability.
- Never a column on `avu_changes`, never a table in schema `mesa`, never a file under
  `.mesa/ducklake/`, never the mesa-ducklake catalog file (single-writer lock).
- Join key into DuckLake: `(project_id, snapshot_id, irods_path, attribute, value, unit)`;
  `snapshot_id` is NULL for abstain, rejected, proposed and dry-run links and for planner and
  rule decisions; `provenance reconcile` repairs NULLs from `get_history(source LIKE 'mesa-anyjev:%')`.
- Decisions are never updated; only `avu_links.write_status` / `snapshot_id` and `runs.status`
  change. Human picks and labels are appended.
- Migrations are numbered, append-only, one transaction per file, with a header comment that
  explains what the change records and why it is provenance about a decision rather than data
  inside the AVU.

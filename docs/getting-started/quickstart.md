---
title: "Quickstart"
description: "Annotate one dataset card end to end with the fake backend, apply the proposals to a local DuckLake, and read the provenance back."
type: Tutorial
tags:
  - getting-started
  - quickstart
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

# Quickstart

Everything below runs offline on the synthetic backend and a local DuckLake catalog; the
recorded OLS responses under `tests/fixtures/ols/` stand in for the EMBL-EBI API.

```bash
export MESA_ANYJEV_OLS__FIXTURES=replay MESA_ANYJEV_POLICY__PROFILE=dev
uv run mesa-anyjev annotate --card tests/fixtures/cards/DP1.10003.001.brd_countdata.md \
    --backend fake --planner static \
    --provenance duckdb:///$PWD/.local/prov.duckdb --out .local/run.json
```

`annotate` decides and proposes; it writes nothing to iRODS. The run id, the proposals with
their probability and level, and the neon-avu-eval result shape are in `.local/run.json`.
The shipped policy is proposed-only, so the curator accepts the run:

```bash
RUN=$(jq -r .run_id .local/run.json)
uv run mesa-anyjev apply --run-id $RUN --accept proposed \
    --provenance duckdb:///$PWD/.local/prov.duckdb \
    --local-ducklake duckdb:///$PWD/.local/lake.duckdb \
    --irods-path /local/mesa-anyjev/DP1.10003.001/brd_countdata.csv --actor $USER
uv run mesa-anyjev explain --provenance duckdb:///$PWD/.local/prov.duckdb \
    --path /local/mesa-anyjev/DP1.10003.001/brd_countdata.csv
```

`apply` writes one mesa-ducklake snapshot for the run and path with the source tag
`mesa-anyjev:apply:human`; `explain` joins each AVU back to the decision that produced it,
its level and its snapshot id.

To decide with a real model, see [Gateway](gateway.md).

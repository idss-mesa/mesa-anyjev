# mesa-anyjev

Declarative, calibrated ontology and schema decisions for the MESA stack.

**mesa-anyjev** turns every choice on the way from a dataset to an ontology-grounded AVU into a
typed question with a probability and an explicit calibration level, using
[AnyJev](https://github.com/nokia-applied-research/AnyJev) over an open model: which aspect
of a dataset a column describes, which ontology to search, which OLS candidate term fits,
what the AVU value should be, whether to keep the annotation. A reasoning model (a CARC model
on sparky-2 through the LiteLLM gateway, or Claude) only *plans*: which ontologies are in play
and what to search. Deterministic code does the rest through
[mesa-mcp](https://github.com/idss-mesa/mesa-mcp)'s OBO/OLS tools, and every decision is
recorded in a provenance sidecar next to the AVU history kept by
[mesa-ducklake](https://github.com/idss-mesa/mesa-ducklake). Hosted Jev through MotherDuck's
`prompt_jev` is a third, policy-gated provider for the same questions.

Status: pre-alpha, milestone M0 (scaffold and contracts). See `DESIGN.md` for the decisions,
`RESEARCH.md` for the verified facts the design rests on, and `CLAUDE.md` for how to work here.

## Why

In the neon-avu-eval experiment four agentic models annotated seven NEON tables with only
valid OBO terms, yet agreed with each other on 11-23% of exact CURIEs and with themselves on
39-50% between runs. The choices inside that loop were implicit and unrepeatable. Here each
choice is a fixed question whose variable content lives in the state, so it accumulates a
batch prior, then a temperature (L1), then a closed-form head (L2) from curator feedback, and
nothing auto-writes below a level and threshold that a leave-one-card-out bench has cited.

## Install

```bash
uv sync --all-extras --no-extra hf        # everything except local CUDA weights
uv run mesa-anyjev doctor                 # what this host can reach
```

Requires Python 3.11+. `mesa-mcp`, `mesa-ducklake` and `anyjev` are installed from pinned git
commits (none is on PyPI at a usable version).

## Quick start (milestone M1)

```bash
uv run mesa-anyjev annotate --card path/to/card.md --backend fake --planner static \
    --provenance duckdb:///$PWD/.local/prov.duckdb --out .local/run.json
uv run mesa-anyjev apply --run-id <run_id> --profile dev --accept proposed \
    --provenance duckdb:///$PWD/.local/prov.duckdb \
    --local-ducklake duckdb:///$PWD/.local/lake.duckdb \
    --irods-path /local/mesa-anyjev/DP1.10003.001/brd_countdata.csv --actor $USER
uv run mesa-anyjev explain --provenance duckdb:///$PWD/.local/prov.duckdb \
    --path /local/mesa-anyjev/DP1.10003.001/brd_countdata.csv
```

## License

MIT, Copyright (c) 2026 The Regents of the University of New Mexico. AnyJev is Apache-2.0 and is
not affiliated with TypeSafe AI or Jev; see `THIRD_PARTY.md`.

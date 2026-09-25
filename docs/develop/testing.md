---
title: "Testing"
description: "How to run the hermetic mesa-anyjev test suite and the opt-in engine, Postgres, end-to-end and hosted tiers."
type: Guide
tags:
  - develop
  - testing
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

# Testing

```bash
uv run pytest -q                                   # hermetic: FakeBackend, DuckDB under tmp_path
uv run ruff check src tests scripts && uv run mypy --strict src
```

Opt-in tiers are excluded by default and selected with markers and environment variables:
`engine` (`MESA_ANYJEV_ENGINE=gateway|hf`), `requires_postgres` (`MESA_ANYJEV_TEST_PG_*`),
`e2e` (`MESA_ANYJEV_E2E=1`), `live` (`MESA_ANYJEV_LIVE=1`), `hosted` (`MESA_ANYJEV_MOTHERDUCK=1`).
Unit tests never touch the network or a GPU. Every question key is pinned by
`tests/test_questions_lock.py`; every numeric write threshold is checked by
`tests/test_policy_citations.py`.

## The Postgres and end-to-end tiers

```bash
docker run -d --rm --name mesa-anyjev-pg -e POSTGRES_PASSWORD=mesa -e POSTGRES_USER=mesa \
  -e POSTGRES_DB=mesa_anyjev_test -p 127.0.0.1:55432:5432 postgres:16
MESA_ANYJEV_TEST_PG_DSN=postgresql://mesa:mesa@127.0.0.1:55432/mesa_anyjev_test \
  uv run pytest -q -m requires_postgres tests/test_provenance_postgres.py
```

The `e2e` tier (`tests/e2e/test_chooser_harness.py`) spawns the real `mesa-mcp --transport
stdio` with the copied elicitation broker and lets the chooser answer `mesa_avu_apply_term`'s
picker. It needs `MESA_E2E_IRODS_ROOT`, iRODS credentials in the environment, the `e2e`
extra and live EBI OLS; it is skipped otherwise.

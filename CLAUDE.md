# CLAUDE.md — mesa-anyjev

mesa-anyjev makes MESA's ontology and schema choices declarative and calibrated: AnyJev typed
decisions (choice / noul / score) over the OBO/OLS tools of mesa-mcp, with every decision
recorded in a sidecar next to the mesa-ducklake AVU history. Python 3.11+, hatchling `src/`
layout, package `mesa_anyjev`, console script `mesa-anyjev`, MCP tools `mesa_decide_*`.
House style follows `idss-mesa/neon-mcp`. `DESIGN.md` records every decision (the D-numbers)
and `RESEARCH.md` the verified facts; read both before changing behaviour. The full plan that
produced this repo is summarised in `DESIGN.md` § Plan.

## Commands

```bash
uv sync --all-extras --no-extra hf                 # dev install without CUDA weights
uv run pytest -q                                   # hermetic suite (fake backend, DuckDB)
uv run ruff check src tests scripts && uv run ruff format --check src tests
uv run mypy --strict src
uv run mesa-anyjev questions --check               # Question keys match questions.lock.json
uv run mesa-anyjev doctor                          # what this host can reach
export MESA_LLM_API_KEY=$(grep '^MESA_LLM_API_KEY=' ../mesa-mcp/.env | cut -d= -f2-)   # never `source` that file
MESA_ANYJEV_ENGINE=gateway uv run pytest -q -m engine tests/engine   # needs the tunnel (see RESEARCH.md)
MESA_ANYJEV_ENGINE=hf uv run pytest -q -m engine tests/engine/test_hf_l2.py   # CUDA host, hf extra, ~2 min load
MESA_ANYJEV_TEST_PG_DSN=postgresql://... uv run pytest -q -m requires_postgres  # docker postgres:16 (docs/develop/testing.md)
```

## Fixed decisions (do not re-litigate; details in DESIGN.md)

- Questions are fixed-key: the option list is frozen (registry, aspects, value kinds, Yes/No)
  and the variable content lives in the state. Open candidate sets are one noul per candidate
  under one key, ranked by p_true. Any wording or option edit rotates the key and must update
  `questions.lock.json` (D1, D7).
- `level` is the AnyJev level only (raw, L0, L1, L2) or `none`; `calibration` is separate
  (anyjev, typesafe, none). `probs` is NULL iff calibration is none. Nothing with calibration
  `none` ever auto-writes; hosted (typesafe) records never auto-write until a cited bench row
  exists (D3, D16). Never fabricate a distribution.
- The provider resolves the level per exact key itself and never trusts `Decider.route()` for
  noul questions (amendment A1). `observe()` never fits inside a serving process (A3).
- Every numeric `auto` threshold in `policy_defaults.yaml` cites a leave-one-card-out bench
  JSON cell; a test refuses uncited numbers and positives-only cells (D6, A2).
- Two phases: `annotate` decides and proposes; `apply` writes (one DuckLake snapshot per run
  and path). Decisions are written to the sidecar before any iRODS write (D10, D13).
- Provenance is a mesa-anyjev-owned sidecar (Postgres schema `mesa_anyjev` or a DuckDB file),
  never a column on `avu_changes`, never a table in schema `mesa` (D4).
- Gateway backend: `logprobs=20` on every request, never `allowed_token_ids`; choices with
  K > `max_choice_k` (8) use the noul twin; L2 only from local weights (D11).
- The reasoning model plans (ontologies, columns, queries); it never answers a question and
  never writes (D2). Gateway planner by default, Claude from M2, Static as fallback.
- Tool prefix `mesa_decide_`, env prefix `MESA_ANYJEV_`, source tag `mesa-anyjev:<tool>:<outcome>`
  (D9). The name "anyjev" appears only in the package name.
- MIT, Copyright (c) 2026 The Regents of the University of New Mexico.

## Code map

`config.py` (settings) · `registry.py` (frozen vocabularies) · `questions.py` + `questions.lock.json`
(the only place Question objects are built) · `cards.py`, `states.py` (dataset cards -> JSON
states) · `providers/` (DecisionRecord, AnyJev / Claude / hosted providers) · `backends/`
(gateway, composite, factory) · `planner/` · `ols.py`, `avu.py` (deterministic candidate and AVU
code over mesa_mcp) · `policy.py` (outcomes) · `pipeline.py` (annotate) · `apply.py` (write) ·
`service.py` · `health.py` (doctor) · `provenance/` (sidecar) · `artifacts.py` · `learn/` ·
`bench/` · `chooser.py` · `mcp_tools/` · `cli.py`.

## Testing rules

- Unit tests never touch the network or a GPU: `FakeBackend`, `RecordingOLS` fixtures, DuckDB
  files under `tmp_path`. Markers `live`, `engine`, `requires_postgres`, `e2e`, `hosted` are
  opt-in through environment variables and are excluded by default.
- Every number in docs or tables names the results JSON it came from (AnyJev ground rule 1).
- No token, key or password is ever logged, printed or committed.

## Git

Agents commit on `feat/`, `fix/`, `docs/` branches with Conventional Commits and open PRs;
never amend shared branches. `CHANGELOG.md` (Keep a Changelog) for the package, `docs/log.md`
for the docs.

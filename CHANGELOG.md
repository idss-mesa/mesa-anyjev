# Changelog

All notable changes to the mesa-anyjev package. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Documentation changes are logged separately in
`docs/log.md`.

## [Unreleased]

### Added

- Milestone M2: labels, bench and L1. `learn ingest` writes the neon-avu-eval silver labels
  with their exact states (OLS term records recorded as fixtures); `bench run` evaluates raw,
  L0 and L1 with leave-one-card-out, option-reversal and phrasing flip probes, coverage at 5%
  and 10% risk, the `neon_term_choice26` control task, and writes results JSON with the
  environment; `learn fit` fits L1 temperatures with class-count guards and `learn promote`
  gates on held-out regressions; the Claude planner (structured outputs, adaptive thinking,
  static fallback on refusal). First gateway bench and L1 fit recorded under
  `bench/results/2026-09-25/` and RESEARCH.md; thresholds stay proposed-only.
- Milestone M1: one card end to end. `mesa-anyjev annotate` (the decide phase, staged per
  question, every decision and proposal in the sidecar), `apply` (local DuckLake mode, one
  snapshot per run and path, `--accept auto|proposed|all|<ids>`), `explain`, `plan`, and the
  `doctor`'s gateway checks. The CARC gateway backend (logprobs=20, no allowed_token_ids,
  missing labels counted, loopback only, circuit breaker), the AnyJev provider with honest
  per-key level resolution, the static and gateway planners, the deterministic OLS candidate
  layer with recorded fixtures, and the canonical AVU builder with a parity test against
  mesa-mcp.
- Milestone M0: repository scaffold and contracts. Configuration model (`MESA_ANYJEV_*`),
  frozen vocabularies (`registry.py`), the production `Question` set with `questions.lock.json`
  and a drift test, dataset-card parser and JSON state builders, the `DecisionRecord` provider
  contract with mandatory `level` and `calibration`, the write policy with cited thresholds,
  the provenance sidecar (DuckDB store; Postgres migration `0001_mesa_anyjev.sql`), the artifact
  store, and the vendored AnyJev metrics with the bench `Task` pattern.

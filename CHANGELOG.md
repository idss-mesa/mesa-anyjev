# Changelog

All notable changes to the mesa-anyjev package. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Documentation changes are logged separately in
`docs/log.md`.

## [Unreleased]

### Added

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

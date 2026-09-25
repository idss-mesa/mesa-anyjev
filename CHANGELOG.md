# Changelog

All notable changes to the mesa-anyjev package. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Documentation changes are logged separately in
`docs/log.md`.

## [Unreleased]

### Added

- Milestone M4: mesa-mcp integration. `DecisionService` (one lock with a bounded wait,
  human picks as overrides, labels and link changes, candidates read back from the sidecar);
  the `mesa_decide_annotate|apply|explain|feedback|health` tools registered into mesa-mcp's
  registry with the surface `decision` (entry point `mesa_mcp.tools`; mesa-mcp PR #6 adds the
  loader), `apply` asking one elicitation per open candidate group with an ids-only state
  and a tamper guard; `ElicitationChooser` for mesa-mcp's own term picker with the
  `tests/e2e` harness; the Postgres sidecar store on the packaged migration; `provenance
  export` (Parquet) and `provenance reconcile`; the `feedback` CLI verb; iRODS-mode apply
  resolves the nearest registered project and never registers one.
- Milestone M3: local L2 on the GB10. The `hf` extra (torch 2.14 CUDA 13, transformers 5,
  accelerate) loads `Qwen/Qwen3-8B` through AnyJev's `HFBackend`; `doctor --backend hf`
  checks the device, the model's depth and width, single-token labels, answer mass and the
  block loop; `learn fit --level L2` fits closed-form heads with the same leave-one-card-out
  guards as L1, and the bench's L1/L2 cells share that fitting path and its pooling rule (the
  M2 ECE disagreement came from the fitter averaging per-fold ECE and coverage); `annotate` and `bench` load the promoted bundle so `--level auto` serves L2 per
  exact question key; the composite backend pairs gateway logprobs with local hidden states
  after asserting label-token parity; `backend.hf_native_triton` (default off) deregisters
  torch's Triton eager overrides so hosts without `Python.h` run the stock aten kernels.
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

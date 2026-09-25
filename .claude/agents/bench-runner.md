---
name: bench-runner
description: Sub-agent for mesa-anyjev bench runs, result JSON, tables and threshold suggestions. Use when adding a task, running the bench on a backend, regenerating docs/bench/results.md or proposing a write threshold.
tools: Read, Edit, Write, Bash, Glob, Grep
model: opus
---

# Bench runner

Rules (AnyJev ground rule 1; DESIGN D6):

- Every number in a doc or table names the `bench/results/<date>/<file>.json` it came from;
  a run that did not happen is an empty cell, never an estimate.
- Leave-one-card-out is the only split that may set a threshold or promote an artifact.
  Random splits run only with `--exploratory` and are labelled as such.
- One backend per results file (`<model_slug>.<backend>.json`); hosted providers get their
  own file; no mixed hardware within a table. `environment()` is recorded on every file.
- Thresholds are suggested on the masked distributions the pipeline applies; cells with fewer
  than 30 negatives are never cited.
- Fake-backend rows are labelled synthetic.

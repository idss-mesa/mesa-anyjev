---
title: "Hosted providers"
description: "Hosted Jev through MotherDuck prompt_jev as a third provider for the same questions, and the data-residency switch that keeps it off by default."
type: Guide
tags:
  - concepts
  - motherduck
  - hosted
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

# Hosted providers

MotherDuck's `prompt_jev` SQL function answers the same three primitives (choice, score,
yes/no) and returns a choice, a full probability distribution and a confidence (DESIGN D16).
mesa-anyjev asks it the identical questions: the frozen question text is the instruction,
the frozen option list is the `choice := [...]` constant, and the `State:` text is the same
rendering AnyJev prefills, so a hosted answer and a local answer to one state share the
`state_sha256`. The SQL literal for every question is pinned in `questions.lock.json`
(`hosted_sql`) beside its key, without entering the lock sha.

## What a hosted record is

`provider='motherduck'`, `method='hosted:prompt_jev'`, `level='none'` (no AnyJev level
exists for it), `calibration='typesafe'` with the distribution re-ordered into the frozen
option order. Three things never become a probability: a `value` the frozen options do not
contain, a distribution that does not sum to one, and a NULL answer (MotherDuck returns NULL
for a failed request and the query continues). Those records abstain with
`calibration='none'` and the reason in `diagnostics`; a NULL is recorded as
`decider_unavailable`. The observed return shape (`STRUCT(choice, probabilities[],
confidence)` for choice, a DOUBLE for yes/no, a weighted position for score) is pinned in
the parser and in the doctor, so a change in the preview fails loudly.

## Scoring stored states

```bash
export MOTHERDUCK_TOKEN=...      # read by the DuckDB extension; never stored in config
MESA_ANYJEV_POLICY__HOSTED_PROVIDERS=allowlist \
  uv run mesa-anyjev hosted score --question term.fits --source labels --dry-run
MESA_ANYJEV_POLICY__HOSTED_PROVIDERS=allowlist \
  uv run mesa-anyjev hosted score --question term.fits,column.aspect --source labels
```

`--dry-run` prints the rows, requests (32 rows per request) and approximate input tokens
without sending anything. A real run records a normal sidecar run with
`backend_kind='motherduck'`, `data_left_host=true` and the egress region, one decision row
per state, and, when the states carry labels, a results file of its own
(`bench/results/<date>/motherduck__prompt_jev.motherduck.json`), never mixed with gateway
or local rows. `--source run --run-id <id>` re-scores the decisions of a local run so the
two providers can be compared state by state. Re-scoring effective AVUs straight from the
DuckLake history waits for states that carry the dataset card.

## Data residency

Because the state leaves the host, hosted providers are off by default
(`policy.hosted_providers: off`) and the CLI, the service and the doctor refuse them. With
`allowlist`, a local source (labels, fixtures, a sidecar run on this host) is allowed when
`hosted_allow_local_sources` is true, and a project only when its root is listed in
`hosted_allow_project_roots` or carries `mesa.hosted_inference=allow`. The `prod` profile
allows `auto` for AnyJev-calibrated records only, so a hosted decision may be proposed but
never auto-writes until a bench row in its own results file cites its coverage at 5% risk.
`mesa-anyjev doctor` adds the MotherDuck checks (extension, token present by name only,
`md:` attach, a two-option fixture, the pinned shape) whenever the policy is not `off`.

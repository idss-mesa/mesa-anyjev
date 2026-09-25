---
title: "Learning"
description: "Where labels come from, how L1 temperatures and L2 heads are fitted with leave-one-card-out, the guards against positives-only data, and how an artifact is promoted."
type: Guide
tags:
  - concepts
  - learning
  - calibration
  - labels
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

# Learning

Labels are option indices per `(question_key, state)` in the sidecar's `labels` table, each
with a source and a weight; when a state has several, the highest weight wins (DESIGN D6):

| Source | Weight | Where it comes from |
|---|---|---|
| `curator` | 1.0 | an explicit pick or "none of these" in the term picker (M4) |
| `curator_implicit` | 0.7 | the other candidates offered next to a pick |
| `accepted_avu` | 0.8 | an auto-written AVU later deleted through mesa-mcp |
| `consensus_all`, `consensus_majority` | 0.8, 0.6 | a term proposed by every, or by at least two, of the four agentic models in neon-avu-eval |
| `consensus_negative` | 0.5 | a term proposed by a single model, or an ontology no model used for a column |
| `teacher`, `hosted_jev` | 0.5 | allowed in the schema; not produced in 0.1.0 |
| `gold` | 1.0 | hand-verified rows |

`mesa-anyjev learn ingest --eval-root <neon-avu-eval>` writes the silver from the
evaluation: `term.fits` states for every unique (card, CURIE) pair with the candidate's OLS
record in the state, `column.annotate`, `column.aspect`, `column.ontology` and its yes/no
twin, `avu.value_kind` and `avu.keep`. The silver is *agreement between models*, not truth,
and it is circular with the agentic baselines the bench compares against.

## Fitting

`mesa-anyjev learn fit --question term.fits --level L1` fits a temperature on the L0
probabilities. Two guards come from the plan's critique: the per-question `min_weight` in
`policy_defaults.yaml` lets the single-model negatives in (at the default 0.6 the term labels
are positives-only, which would make temperature scaling degenerate and coverage trivially
100%), and every leave-one-card-out fold needs at least 30 labels per class in training and
5 in the held-out card, else it is skipped and reported. Fitting runs on a Decider without
adaptive shifts so the artifact freezes the prior it was fitted with.

`--level L2` fits a closed-form head (LDA or ridge, chosen by the same held-out score) on the
hidden state of one block, on a backend that exposes hidden states (`hf`, `composite`, or the
fake backend in tests); the gateway raises `LevelUnavailable`. Both levels go through one
fitting path, `learn/fit.py`, which the bench's leave-one-card-out cells now share, so a bench
L1 or L2 cell and a `learn fit` report on the same labels are the same computation, and both
pool the held-out decisions before computing ECE and coverage (M2's two paths disagreed on
term.fits ECE, 0.072 against 0.231, because the fitter averaged per-fold values of metrics
that are not linear in the items). L2 needs at least 40 labels and the same per-class fold
guards; the manifest records the head's block (`layer_abs`), method and, for L1, the
temperature. At serving time a head answers only its own question key (A1): a fitted
`term.fits` head is never routed to another yes/no question.

The fit is saved as a new immutable artifact version whose manifest carries the pooled
held-out numbers; `learn promote --version N` moves `CURRENT` only when held-out accuracy and
ECE do not regress against the current version (D15). Serving processes never fit.

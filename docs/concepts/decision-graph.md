---
title: "Decision graph"
description: "The fixed-key questions mesa-anyjev asks from a dataset card to a written AVU, which provider answers each, and what happens below threshold."
type: Guide
tags:
  - concepts
  - decisions
  - questions
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

# Decision graph

Every question is fixed-key (DESIGN D1): its wording and option list are frozen in
`questions.py` and pinned in `questions.lock.json`; the variable content (the column, the
site, the candidate term, the sibling AVUs) lives in the *state*. That is what lets one
question accumulate a batch prior, an L1 temperature and an L2 head across every column and
dataset. Open candidate sets are asked as one yes/no per candidate under one key and ranked by
`p_true`, which also removes AnyJev's 26-option cap.

| Step | Question | Kind | Decides |
|---|---|---|---|
| Plan | (the reasoning model) | planner | which ontologies are in play, which columns to annotate, what to search; hints only |
| Q1 | `column.annotate` | noul | should this column be annotated at all (identifiers are a rule, never a model call) |
| Q2 | `column.aspect` | choice, K=8 | taxon, environment, method, measurement, unit, data_type, location, other |
| Q3 | `column.ontology` / `column.ontology_fits` | choice K=12, or its noul twin on the gateway | which registry ontology to search; masked by aspect and plan |
| S | candidates | code | OLS search, prefix filter, dedup, cap |
| Q4 | `term.fits` | noul per candidate | is this candidate the right term at the right specificity (one key for column, site, unit and taxon scopes) |
| Q7 | `avu.value_kind` | choice, K=4 | term label, site code, column name, or top data value (after deterministic pre-rules) |
| Q8 | `avu.keep` | noul | keep this AVU given its siblings |
| H | human pick | MRTR | the curator's choice is authoritative and becomes a label |
| W | write | code | one DuckLake snapshot per run and path |

The registry, the aspects and the value kinds are in `mesa_anyjev.registry`; the ontology
registry is the ten ontologies allowed by the neon-avu-eval prompt plus TAXRANK and GENEPIO
(every prefix with at least five valid evaluation AVUs, DESIGN D7).

---
title: "Second opinions"
description: "Claude as a recorded second opinion on proposed terms (never a probability, never auto), the specificity step over child terms, and the planner audit."
type: Guide
tags:
  - concepts
  - claude
  - specificity
  - audit
generated:
  by: "claude/fable-5.1"
  at: "2026-09-25T00:00:00Z"
sources:
  - id: design
    resource: "https://github.com/idss-mesa/mesa-anyjev/blob/main/DESIGN.md"
    title: "mesa-anyjev decisions register (DESIGN.md)"
    author: "team:idss-mesa"
status: draft
---

# Second opinions

Three M5 steps refine a proposal without ever raising its level.

## Claude as a recorded second opinion

The Claude Messages API exposes no logits, so Claude can never be an AnyJev backend. What it
can do is answer the same rendered `State` / `Question` / `Options` text through structured
outputs (`messages.parse` with an answer model whose only field is a literal over the frozen
options, adaptive thinking, no forced tool choice). Every such record is `provider='claude'`,
`level='none'`, `calibration='none'`, `probs=None`; a refusal, an error or an answer outside the
options abstains. With `claude.second_opinion` on (or `annotate --second-opinion`), a
`proposed` candidate group gets Claude's yes/no over its top candidates, recorded in the same
group with the winner as parent. Claude agreeing is noted on the proposal; Claude saying no
to the winner turns the outcome into `escalated`, which stays a proposal for a human to
settle. Nothing Claude says can make a group `auto` (the profile requires an AnyJev
calibration), and nothing it says changes a probability.

## Specificity

When a proposed term has children in its ontology, the pipeline asks `term.fits` over those
children (a second group that records which term it refines). A child replaces the parent
only when p(child) is at least `policy.specificity_delta` (0.10) above p(parent) and the child
clears the proposal threshold on its own. Recorded fixtures may lack the children calls; the
parent then stands. On the first end-to-end run at L0 (carc-fast) 23 child groups were
opened and none replaced its parent ([Bench results](../bench/results.md)).

## Planner audit

Before any column is looked at, `dataset.ontology_applies` is asked once per registry entry
over the card header. The answers are recorded (they never write) and compared with the
planner's ontology list: `annotate` prints the agreement, and the run's result carries
`audit.planner_only` and `audit.model_only`, so a planner that keeps naming ontologies the
model rejects, or missing ones it accepts, shows up in the sidecar rather than in a hunch.

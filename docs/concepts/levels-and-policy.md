---
title: "Levels and policy"
description: "What an AnyJev level and the calibration field promise, and the rules that turn a decision into auto, proposed, human or abstain."
type: Guide
tags:
  - concepts
  - calibration
  - policy
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

# Levels and policy

Every stored decision carries two fields (DESIGN D3):

* `level`: AnyJev's level (`raw`, `L0` zero-label debiasing, `L1` temperature on labels,
  `L2` closed-form head on hidden states) or `none` for anything that is not an AnyJev
  decision: a rule, a planner hint, a Claude structured answer, a hosted Jev answer.
* `calibration`: where the probabilities come from: `anyjev`, `typesafe` (hosted Jev), or
  `none`. `probs` is null exactly when calibration is `none`; nothing is ever one-hot.

The write policy (`policy_defaults.yaml`, `mesa_anyjev.policy`) reads `p_true` for yes/no
questions (never `confidence`, which is the larger of p and 1-p) and `confidence` for
choices. A decision becomes:

* `auto` only with an AnyJev level at or above both the question's and the profile's floor
  (`prod`: L1), a calibration the profile allows, a numeric threshold that cites the
  leave-one-card-out bench cell it came from, and no label missing from the gateway's top-20
  in that batch;
* `proposed` at or above the propose threshold (the curator sees a ranked picker);
* `abstain` otherwise.

The shipped defaults are proposed-only (`auto: null` everywhere) until a bench run exists.

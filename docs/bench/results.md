---
title: "Bench results"
description: "Measured raw, L0 and L1 numbers per question on the CARC gateway and the synthetic backend, every cell naming its results JSON."
type: Reference
tags:
  - bench
  - results
  - calibration
generated:
  by: "claude/fable-5.1"
  at: "2026-09-25T00:00:00Z"
sources:
  - id: gateway-results
    resource: "https://github.com/idss-mesa/mesa-anyjev/blob/main/bench/results/2026-09-25/RedHatAI__Qwen3-8B-NVFP4.gateway.json"
    title: "bench results 2026-09-25, carc-fast through the CARC gateway"
    author: "team:idss-mesa"
  - id: fake-results
    resource: "https://github.com/idss-mesa/mesa-anyjev/blob/main/bench/results/2026-09-25/fake.fake.json"
    title: "bench results 2026-09-25, synthetic backend"
    author: "team:idss-mesa"
status: draft
stale_after: "2026-12-31T00:00:00Z"
---

# Bench results

Labels are the neon-avu-eval silver (agreement between four agentic models, not truth;
see [Learning](../concepts/learning.md)). `L1 (LOCO)` cells are pooled held-out decisions
over seven leave-one-card-out folds. `cov@5%` is the share of items that can be answered
before the error rate on the answered set exceeds 5%; `flip` is the answer change under
option reversal (choice) or phrasing swap (yes/no). Every number below names its JSON.

## carc-fast (RedHatAI/Qwen3-8B-NVFP4) through the CARC gateway, 2026-09-25

| task | n | level | acc | ece | brier | cov@5% | cov@10% | flip | n_neg | prompts | ms/dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| neon_term_fits | 285 | raw | 0.502 | 0.372 | 0.780 | 0.004 | 0.004 |  | 199 | 285 | 100.6 |
| neon_term_fits | 285 | L0 | 0.611 | 0.262 | 0.605 | 0.007 | 0.049 | 0.393 | 199 | 570 | 126.5 |
| neon_term_fits | 285 | L1 (LOCO) | 0.621 | 0.072 | 0.459 | 0.007 | 0.042 |  | 199 | 3990 | 728.9 |
| neon_term_choice26 | 44 | raw | 0.227 | 0.689 | 1.443 | 0.000 | 0.000 |  | 28 | 44 | 151.6 |
| neon_term_choice26 | 44 | L0 | 0.205 | 0.689 | 1.460 | 0.000 | 0.000 | 0.343 | 28 | 152 | 354.5 |
| neon_term_choice26 | 44 | L1 | n/a | | | | | | | | |
| neon_annotate | 98 | raw | 0.378 | 0.517 | 1.062 | 0.000 | 0.000 |  | 35 | 98 | 46.8 |
| neon_annotate | 98 | L0 | 0.398 | 0.490 | 1.020 | 0.000 | 0.000 | 0.316 | 35 | 196 | 39.1 |
| neon_annotate | 98 | L1 (LOCO) | 0.294 | 0.467 | 0.560 | 0.000 | 0.000 |  | 5 | 196 | 40.4 |
| neon_aspect | 60 | raw | 0.600 | 0.376 | 0.775 | 0.000 | 0.000 | 0.433 | 0 | 120 | 83.2 |
| neon_aspect | 60 | L0 | 0.583 | 0.373 | 0.736 | 0.000 | 0.000 | 0.283 | 0 | 444 | 242.0 |
| neon_aspect | 60 | L1 (LOCO) | 0.600 | 0.181 | 0.552 | 0.017 | 0.017 |  | 0 | 1680 | 718.0 |
| neon_ontology_fits | 190 | raw | 0.411 | 0.575 | 1.140 | 0.047 | 0.063 |  | 114 | 190 | 47.4 |
| neon_ontology_fits | 190 | L0 | 0.474 | 0.417 | 0.868 | 0.032 | 0.116 | 0.289 | 114 | 380 | 47.6 |
| neon_ontology_fits | 190 | L1 (LOCO) | 0.458 | 0.182 | 0.494 | 0.068 | 0.105 |  | 114 | 2660 | 310.7 |
| neon_keep_avu | 313 | raw | 0.965 | 0.025 | 0.064 | 1.000 | 1.000 |  | 11 | 313 | 171.9 |
| neon_keep_avu | 313 | L0 | 0.895 | 0.044 | 0.160 | 0.786 | 0.974 | 0.323 | 11 | 626 | 178.4 |
| neon_keep_avu | 313 | L1 | n/a | | | | | | | | |
| neon_value_kind | 278 | raw | 0.507 | 0.457 | 0.937 | 0.004 | 0.004 | 0.259 | 0 | 556 | 73.0 |
| neon_value_kind | 278 | L0 | 0.475 | 0.462 | 0.942 | 0.004 | 0.004 | 0.194 | 0 | 1578 | 124.2 |
| neon_value_kind | 278 | L1 (LOCO) | 0.468 | 0.124 | 0.642 | 0.004 | 0.004 |  | 0 | 4759 | 377.2 |

How to read it (`bench/results/2026-09-25/RedHatAI__Qwen3-8B-NVFP4.gateway.json`):

* **term.fits** (285 candidate states, 199 negatives): accuracy 0.502 raw, 0.611 at L0,
  0.621 at L1; ECE 0.372, 0.262, 0.072. Calibration meets the pre-registered 0.10 bar at
  L1, but coverage at 5% risk is 0.007 (two items), so every `auto` threshold stays null.
* **neon_term_choice26** is the control: one choice per column over its candidates with a
  new key per item, so it can only run raw/L0 and never learns (accuracy 0.23 / 0.21).
* **column.aspect** (K=8) already lost 19 labels to the gateway's top-20 readout at L0;
  the wider the choice, the more the gateway erases (DESIGN D11).
* **avu.keep** has only 11 negatives, so no leave-one-card-out fold is allowed to fit; its
  raw/L0 numbers reflect a positives-heavy label set, not a calibrated gate.
* The bench's L1 cells were fitted with adaptive shifts (the provider's default) whereas
  `learn fit` fits without them; on term.fits they disagree on ECE (0.072 here, 0.231 in the
  promoted bundle's manifest). Milestone M3 makes both use one fitting path.

## Synthetic backend (planted biases; numbers are not a claim about any model)

| task | n | level | acc | ece | brier | cov@5% | cov@10% | flip | n_neg | prompts | ms/dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| neon_term_fits | 285 | raw | 0.442 | 0.504 | 1.017 | 0.007 | 0.007 |  | 199 | 285 | 0.1 |
| neon_term_fits | 285 | L0 | 0.442 | 0.488 | 0.946 | 0.007 | 0.007 | 0.004 | 199 | 570 | 0.2 |
| neon_term_fits | 285 | L1 (LOCO) | 0.442 | 0.178 | 0.508 | 0.007 | 0.007 |  | 199 | 3990 | 1.2 |
| neon_term_choice26 | 44 | raw | 0.250 | 0.249 | 0.811 | 0.000 | 0.000 |  | 28 | 44 | 0.2 |
| neon_term_choice26 | 44 | L0 | 0.159 | 0.317 | 0.841 | 0.000 | 0.000 | 0.411 | 28 | 221 | 0.7 |
| neon_term_choice26 | 44 | L1 | n/a | | | | | | | | |
| neon_annotate | 98 | raw | 0.643 | 0.238 | 0.573 | 0.020 | 0.020 |  | 35 | 98 | 0.1 |
| neon_annotate | 98 | L0 | 0.622 | 0.101 | 0.473 | 0.020 | 0.020 | 0.163 | 35 | 196 | 0.2 |
| neon_annotate | 98 | L1 (LOCO) | 0.647 | 0.353 | 0.457 | 0.118 | 0.118 |  | 5 | 196 | 0.1 |
| neon_aspect | 60 | raw | 0.200 | 0.216 | 0.964 | 0.017 | 0.017 | 0.917 | 0 | 120 | 0.3 |
| neon_aspect | 60 | L0 | 0.117 | 0.156 | 0.878 | 0.000 | 0.000 | 0.000 | 0 | 861 | 1.8 |
| neon_aspect | 60 | L1 (LOCO) | 0.167 | 0.164 | 0.890 | 0.000 | 0.000 |  | 0 | 3074 | 6.3 |
| neon_ontology_fits | 190 | raw | 0.400 | 0.480 | 0.938 | 0.000 | 0.000 |  | 114 | 190 | 0.1 |
| neon_ontology_fits | 190 | L0 | 0.432 | 0.167 | 0.549 | 0.000 | 0.000 | 0.189 | 114 | 380 | 0.2 |
| neon_ontology_fits | 190 | L1 (LOCO) | 0.432 | 0.123 | 0.501 | 0.000 | 0.000 |  | 114 | 2660 | 1.0 |
| neon_keep_avu | 313 | raw | 0.965 | 0.085 | 0.083 | 1.000 | 1.000 |  | 11 | 313 | 0.2 |
| neon_keep_avu | 313 | L0 | 0.911 | 0.318 | 0.352 | 0.888 | 1.000 | 0.173 | 11 | 626 | 0.3 |
| neon_keep_avu | 313 | L1 | n/a | | | | | | | | |
| neon_value_kind | 278 | raw | 0.478 | 0.130 | 0.691 | 0.007 | 0.007 | 1.000 | 0 | 556 | 0.3 |
| neon_value_kind | 278 | L0 | 0.216 | 0.112 | 0.757 | 0.004 | 0.004 | 0.007 | 0 | 2195 | 0.8 |
| neon_value_kind | 278 | L1 (LOCO) | 0.227 | 0.107 | 0.750 | 0.004 | 0.004 |  | 0 | 7661 | 2.4 |

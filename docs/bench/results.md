---
title: "Bench results"
description: "Measured raw, L0, L1 and L2 numbers per question on local Qwen3-8B weights, the CARC gateway and the synthetic backend, every cell naming its results JSON."
type: Reference
tags:
  - bench
  - results
  - calibration
generated:
  by: "claude/fable-5.1"
  at: "2026-09-25T00:00:00Z"
sources:
  - id: hf-results
    resource: "https://github.com/idss-mesa/mesa-anyjev/blob/main/bench/results/2026-09-25/Qwen__Qwen3-8B.hf.json"
    title: "bench results 2026-09-25, Qwen/Qwen3-8B bf16 on the GB10 workstation"
    author: "team:idss-mesa"
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

## Qwen/Qwen3-8B bf16 on the GB10 workstation (local weights), 2026-09-25

The same labels through AnyJev's `HFBackend` on this host: the full vocabulary is read (no
top-20 cap, so no erased labels), and L2 heads read the hidden state of one block.
`L1 (LOCO)` and `L2 (LOCO)` cells are pooled held-out decisions over seven leave-one-card-out
folds. The `prompts` column reads 0 because the local backend had no request counter in this
run (added afterwards; the next run reports prefills).

| task | n | level | acc | ece | brier | cov@5% | cov@10% | flip | n_neg | prompts | ms/dec |
| neon_annotate | 98 | L0 | 0.439 | 0.474 | 0.970 | 0.000 | 0.000 | 0.143 | 35 | 0 | 817.1 |
| neon_annotate | 98 | L1 (LOCO) | 0.294 | 0.529 | 0.569 | 0.000 | 0.000 |  | 5 | 0 | 369.2 |
| neon_annotate | 98 | L2 (LOCO) | 0.765 | 0.257 | 0.354 | 0.353 | 0.353 |  | 5 | 0 | 196.5 |
| neon_annotate | 98 | raw | 0.439 | 0.512 | 1.022 | 0.000 | 0.000 |  | 35 | 0 | 398.8 |
| neon_aspect | 60 | L0 | 0.533 | 0.456 | 0.922 | 0.000 | 0.000 | 0.150 | 0 | 0 | 1917.5 |
| neon_aspect | 60 | L1 (LOCO) | 0.517 | 0.162 | 0.633 | 0.000 | 0.000 |  | 0 | 0 | 8175.3 |
| neon_aspect | 60 | L2 (LOCO) | 0.567 | 0.175 | 0.552 | 0.383 | 0.417 |  | 0 | 0 | 2983.5 |
| neon_aspect | 60 | raw | 0.550 | 0.420 | 0.841 | 0.117 | 0.117 | 0.250 | 0 | 0 | 447.3 |
| neon_keep_avu | 313 | L0 | 0.949 | 0.031 | 0.085 | 0.994 | 1.000 | 0.022 | 11 | 0 | 845.2 |
| neon_keep_avu | 313 | L1 | n/a | | | | | | | | |
| neon_keep_avu | 313 | L2 | n/a | | | | | | | | |
| neon_keep_avu | 313 | raw | 0.962 | 0.036 | 0.074 | 1.000 | 1.000 |  | 11 | 0 | 744.1 |
| neon_ontology_fits | 190 | L0 | 0.426 | 0.560 | 1.111 | 0.089 | 0.116 | 0.084 | 114 | 0 | 621.4 |
| neon_ontology_fits | 190 | L1 (LOCO) | 0.426 | 0.233 | 0.537 | 0.089 | 0.111 |  | 114 | 0 | 5381.1 |
| neon_ontology_fits | 190 | L2 (LOCO) | 0.837 | 0.078 | 0.231 | 0.547 | 0.768 |  | 114 | 0 | 1906.4 |
| neon_ontology_fits | 190 | raw | 0.421 | 0.566 | 1.137 | 0.089 | 0.105 |  | 114 | 0 | 216.5 |
| neon_ontology_for_column | 59 | L0 | 0.441 | 0.508 | 1.027 | 0.169 | 0.271 | 0.186 | 0 | 0 | 3085.4 |
| neon_ontology_for_column | 59 | L1 (LOCO) | 0.441 | 0.141 | 0.732 | 0.102 | 0.102 |  | 0 | 0 | 6622.9 |
| neon_ontology_for_column | 59 | L2 (LOCO) | 0.712 | 0.154 | 0.391 | 0.458 | 0.678 |  | 0 | 0 | 1679.8 |
| neon_ontology_for_column | 59 | raw | 0.492 | 0.443 | 0.896 | 0.051 | 0.051 | 0.136 | 0 | 0 | 568.5 |
| neon_term_choice26 | 44 | L0 | 0.114 | 0.833 | 1.648 | 0.000 | 0.000 | 0.236 | 28 | 0 | 578.2 |
| neon_term_choice26 | 44 | L1 | n/a | | | | | | | | |
| neon_term_choice26 | 44 | L2 | n/a | | | | | | | | |
| neon_term_choice26 | 44 | raw | 0.114 | 0.789 | 1.621 | 0.000 | 0.000 |  | 28 | 0 | 191.4 |
| neon_term_fits | 285 | L0 | 0.663 | 0.293 | 0.616 | 0.077 | 0.182 | 0.119 | 199 | 0 | 392.3 |
| neon_term_fits | 285 | L1 (LOCO) | 0.656 | 0.079 | 0.422 | 0.049 | 0.077 |  | 199 | 0 | 2765.1 |
| neon_term_fits | 285 | L2 (LOCO) | 0.765 | 0.058 | 0.327 | 0.088 | 0.396 |  | 199 | 0 | 1374.7 |
| neon_term_fits | 285 | raw | 0.642 | 0.316 | 0.656 | 0.074 | 0.112 |  | 199 | 0 | 199.1 |
| neon_value_kind | 278 | L0 | 0.421 | 0.477 | 1.022 | 0.014 | 0.014 | 0.385 | 0 | 0 | 1046.0 |
| neon_value_kind | 278 | L1 (LOCO) | 0.428 | 0.088 | 0.669 | 0.007 | 0.007 |  | 0 | 0 | 1709.4 |
| neon_value_kind | 278 | L2 (LOCO) | 0.576 | 0.069 | 0.565 | 0.029 | 0.040 |  | 0 | 0 | 1275.0 |
| neon_value_kind | 278 | raw | 0.406 | 0.536 | 1.127 | 0.014 | 0.014 | 0.500 | 0 | 0 | 370.4 |

How to read it (`bench/results/2026-09-25/Qwen__Qwen3-8B.hf.json`):

* **term.fits** (285 states, 199 negatives): accuracy 0.642 raw, 0.663 L0, 0.656 L1, 0.765
  L2; ECE 0.316, 0.293, 0.079, 0.058; coverage at 5% risk 0.074, 0.077, 0.049, 0.088 and at
  10% risk 0.112, 0.182, 0.077, 0.396. L2 meets the 0.10 ECE bar with the best accuracy,
  but only 25 of 285 items can be answered at 5% risk, so `term.fits` `auto` stays null.
* **column.ontology_fits** (190, 114 negatives) is the one question where L2 changes the
  picture: accuracy 0.837 and ECE 0.078 with coverage 0.547 at 5% risk and 0.768 at 10%,
  against 0.426 / 0.233 / 0.089 at L1. The `policy_suggestions` block cites this cell.
  Before any threshold is adopted the number needs a second label source: these labels are
  the agentic models' ontology choices, so the head has learnt their habits, not the truth.
* **column.annotate** (98, 5 negatives above the weight cut at L1/L2): L2 accuracy 0.765
  against 0.294 at L1, but with five negatives the coverage figure (0.353) rests on almost
  nothing and every fold barely passes the class guard.
* **column.aspect** (60) and **avu.value_kind** (278) gain accuracy at L2 (0.567, 0.576)
  and calibrate (ECE 0.175, 0.069) but stay near zero coverage at 5% risk.
* **neon_term_choice26** at raw/L0 scores 0.114 locally against 0.227 / 0.205 on the
  gateway: reading the full vocabulary does not rescue a 26-way choice with a new key per
  item; the noul-per-candidate design is the right one for this model.
* **avu.keep**: 11 negatives, no fold may fit, unchanged.
* Against the gateway at L0 the local bf16 checkpoint scores term.fits 0.663 vs 0.611 and
  annotate 0.439 vs 0.398, with flip rates 0.119 vs 0.393 on term.fits: the NVFP4 quant and
  the bf16 weights are not interchangeable, which is why bundles are keyed per model (D5).

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
* In M2 the bench and `learn fit` disagreed on term.fits ECE (0.072 here, 0.231 in the
  bundle's manifest). The cause was not the fitting: `learn fit` averaged per-fold ECE and
  coverage, which are not linear in the items, while the bench pooled the held-out
  decisions. Since M3 both share one fitting path and both pool (DESIGN D18).

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

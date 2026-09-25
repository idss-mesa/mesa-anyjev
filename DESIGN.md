# DESIGN.md — decisions register

Numbered, append-only. A change of mind is a new entry that cites the one it amends. Facts the
decisions rest on live in `RESEARCH.md`; the full plan that produced this repository is
summarised at the end.

| # | Decision | Status |
|---|---|---|
| D0 | Depend on AnyJev, mesa-mcp and mesa-ducklake by pinned git commit; the local monorepo checkouts are not the source of truth | accepted |
| D1 | Fixed-key questions: option lists are frozen, variable content lives in the state; a lock file pins every `Question.key` | accepted |
| D2 | The reasoning model plans, never decides and never writes; gateway planner by default, Claude from M2, Static as fallback | accepted |
| D3 | `level` is the AnyJev level or `none`; `calibration` is separate; `probs` NULL iff calibration `none`; `none` never auto-writes | accepted |
| D4 | Provenance is a mesa-anyjev-owned sidecar keyed by decision_id with snapshot_id nullable; never a DuckLake column or a table in schema `mesa` | accepted |
| D5 | Artifact slug = the served repo id until the doctor records tokenizer parity with the bf16 checkpoint; `fitted_on` recorded; strict load | accepted |
| D6 | Write thresholds come only from leave-one-card-out bench cells and cite them; positives-only cells are refused | accepted |
| D7 | Registry, aspect and value-kind vocabularies are frozen; an edit rotates the key, needs the lock updated and labels migrated | accepted |
| D8 | MRTR requestState carries ids only; labels, probabilities and levels on resume come from the sidecar row, never from the client blob | accepted |
| D9 | Tool prefix `mesa_decide_`, env prefix `MESA_ANYJEV_`, source tag `mesa-anyjev:<tool>:<outcome>`; "anyjev" only in the package name | accepted |
| D10 | Two phases: `annotate` (decide, propose, sidecar) and `apply` (iRODS write, one DuckLake snapshot per run and path) | accepted |
| D11 | Capability gating: gateway `logprobs=20`, no `allowed_token_ids`, `max_choice_k=8` with noul twins; a missing label in any shift caps the outcome at rule/abstain; L2 only from local weights | accepted |
| D12 | AVU value kind is a learnable choice with deterministic pre-rules (site environment -> site code; unit -> label) | accepted |
| D13 | One snapshot per (run, path); `record_changes` is never called with an empty list | accepted |
| D14 | Integrate as a plugin into mesa-mcp's registry (entry point + surface meta), not a separate MCP server; the mesa-nmdid separate-server route is the recorded alternative | accepted |
| D15 | Serving processes never fit: labels are stored, fitting happens through `learn fit` + `learn promote` (LOCO gate) | accepted |
| D16 | Hosted Jev (MotherDuck `prompt_jev`) is a third provider for the same questions and lock: level `none`, calibration `typesafe`, egress-gated per project, off by default; it may re-score effective AVUs from the DuckLake history as new sidecar runs | accepted |
| D17 | Claude-as-teacher labels are not part of 0.1.0 (accuracy ceiling, spend); `label_source 'teacher'` stays allowed in the schema | accepted |

## D0. Pinned git dependencies

PyPI `anyjev` is 0.0.2 and lacks L2, `observe()`, `route()` and vLLM hidden states; the code
this repo was designed against is commit `795a4970b47218b7c0686cd579d691fc2cf8df2f`. mesa-mcp
and mesa-ducklake are not on PyPI; the live mains at design time were `8fbaedf850301146b38b997774ab7aed1d0cb179`
and `7bc143fefb093f0a2ace748a7036598147533ca0`. The checkouts under `~/github/idss-mesa/` were
behind those mains (RESEARCH.md), so pins point at GitHub, not at sibling paths.

## D1. Fixed-key questions

`Question.key` is `sha256(kind, text, options)` and every batch prior, L1 temperature and L2 head
is keyed by it. A dynamic option list (OLS hits) is therefore a new question every time and can
never learn. Two learnable shapes: a frozen registry asked as a `choice` with inapplicable
options masked and renormalised after the decision (AnyJev's 2048 demo), and one fixed `noul`
per candidate with the candidate in the state, ranked by `p_true` (the Minesweeper demo). The
second removes the 26-option cap and shares one key across column, site, unit and taxon scopes.
`questions.py` is the only place production questions are built; `questions.lock.json` pins
their keys and `tests/test_questions_lock.py` fails on drift.

## D3. Honest levels

AnyJev's `LEVEL_ORDER` knows raw, L0, L1, L2. Records from rules, planners, Claude structured
outputs and hosted Jev carry `level='none'`; `calibration` says where the probabilities come
from (`anyjev`, `typesafe`, `none`). The invariant is `probs IS NULL iff calibration='none'`,
so a Claude answer never gets a one-hot vector and a hosted Jev answer keeps its distribution.
The provider resolves the level per exact key itself because `Decider.route()` would serve any
noul question from any fitted noul head (same kind, same Yes/No options) at "L2" (amendment A1
in the plan); a record served through routing is stored at L0 with `routed_refused=true`. For
noul the policy reads `p_true`, never `confidence = max(p, 1-p)`.

## D4. Provenance sidecar

`mesa_ducklake.AvuChange` is `extra='forbid'`, the triple is a hard contract, the Parquet reader
unpacks exactly 13 columns, and mesa-ducklake's own rules ask any new column to justify itself
against the triple. Decision provenance (candidate sets, distributions, thresholds, overrides)
does not fit a flat column, so it lives in schema `mesa_anyjev` (or a DuckDB file) with its own
migration runner, joined to DuckLake by `(project_id, snapshot_id, irods_path, attribute,
value, unit)`. `snapshot_id` is obtained by calling `DuckLakeClient.record_changes` directly
until mesa-mcp returns the Snapshot from its mirror (PR 2). Per-row `source` tags
(`mesa-anyjev:apply:auto|human|proposed`) make AnyJev-written rows visible in `get_history`
without a join.

## D6. Thresholds cite evidence

`policy_defaults.yaml` ships `auto: null` everywhere. A numeric `auto` threshold must cite
`bench/results/<date>/<file>.json#cov@5%` from a leave-one-card-out run on the same masked
distribution the pipeline applies; `tests/test_policy_citations.py` parses the cited JSON and
refuses a cell with fewer than 30 negatives. Day-0 `term.fits` labels are positives-only at
weight 0.6 (92 Yes, 0 No), which is why `min_weight` is per question and consensus negatives
(0.5) enter for `term.fits`.

## D11. Capability gating

The CARC gateway drops `allowed_token_ids` and caps `logprobs` at 20 over the full vocabulary
(RESEARCH.md). AnyJev's L0 combines cyclic shifts by log-mean, so one missing label (-30) in
any shift erases that option. Hence: `logprobs=20` always, `missing_labels` counted per
`decide_batch`, any batch with a missing label capped at outcome `rule`/`abstain`, `max_choice_k`
8 until the doctor measures zero erasure at a higher K, and every K>8 fixed choice has a noul
twin under its own key. L2 needs hidden states, which only local weights provide.

## D14. Plugin, not a separate server

mesa-mcp registers tools by import side effect; a guarded entry-point loader (PR 1) lets an
installed mesa-anyjev register `mesa_decide_*` before `MesaServer()` copies the registry, with
the surface declared by the plugin through `register_tool(..., meta=...)`. Until PR 1 lands,
`import mesa_anyjev.mcp_tools` before construction lists the tools under surface `core`.
mesa-nmdid's separate-server pattern (copied registry, own prefix) is the recorded alternative;
it was not chosen because the decision tools must call mesa-mcp's OLS and AVU code in-process
and share its auth context and DuckLake client.

## D16. Hosted Jev

MotherDuck's `prompt_jev(text, instructions, choice := [...] | noul := TRUE | score := [...])`
returns `STRUCT(choice, probabilities STRUCT(value, probability)[], confidence)` (RESEARCH.md).
It is the same three primitives, so the same `Question` objects and the same lock apply: the
question text and option list are the SQL constants. Records: provider `motherduck`, method
`hosted:prompt_jev`, level `none`, calibration `typesafe`, probabilities re-ordered into the
frozen option order by value match. Hosted decisions may reach `proposed` but never `auto` until
a cited bench row exists (`allow_hosted_write`, off). Because the state leaves the host,
`policy.hosted_providers` is `off` by default and `allowlist` requires the project root to carry
`mesa.hosted_inference=allow` or to be allow-listed; neon-avu-eval cards are public (CC BY 4.0).
A SQL batch path (`mesa-anyjev hosted score`) builds states in SQL in the same key order as
`states.py`, can read effective AVUs from the DuckLake Parquet history, and writes results as
ordinary sidecar runs with `data_left_host=true`. MotherDuck-managed DuckLake (it can attach
`ducklake:` catalogs) is a possible future mesa-ducklake catalog backend, out of scope here.

## Plan (summary; the full plan with facts and amendments is the design record)

Milestones: M0 scaffold and contracts (this) -> M1 one card end to end with StaticPlanner and
GatewayPlanner, fake backend then carc-fast L0, local apply, mesa-mcp PR 2 -> M2 labels
(class-balanced), bench (masked, LOCO), L1 on the gateway, ClaudePlanner -> M3 local HF L2 on
the GB10, composite backend, promotion -> M4 mesa-mcp tools, chooser e2e, Postgres sidecar,
PRs 1/3/4/5, installer -> M4b hosted Jev on MotherDuck -> M5 Claude structured-output provider,
specificity pass, DataCite questions, docs site, 0.1.0.

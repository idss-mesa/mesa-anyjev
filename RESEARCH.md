# RESEARCH.md — verified facts (2026-09-25)

Every fact names where it was verified. Re-check anything marked `stale_after`.

## Hosts and the gateway

- This workstation (`sparky-1`): NVIDIA GB10, aarch64, 121 GB unified memory, 20 cores, driver
  580.173.02 / CUDA 13.0, Docker 29 (`nvidia-smi`, `free -g`, `nproc`). `uv pip install --dry-run`
  resolves torch 2.14.0 (CUDA 13 wheels) + transformers 5.17.0 for aarch64 py3.11. Installed
  in M3 (venv on the system CPython 3.12, `uv sync --all-extras`): `torch 2.14.0+cu130`, CUDA
  13.0 visible, a bf16 4096x4096 matmul runs; `Qwen/Qwen3-8B` bf16 loads through AnyJev's
  `HFBackend` in about two minutes from the safetensors cache (16.4 GB resident; `doctor
  --backend hf` 2026-09-25: `n_layers=36 hidden_size=4096`, labels single-token, probe
  answer mass 1.000, block loop available). `accelerate` is required by transformers 5 for
  `device_map`; it is in the `hf` extra. torch 2.14's `torch._native` routes the Qwen3 rotary
  outer product through a Triton kernel whose driver shim is compiled against `Python.h` on
  first use; the system Python has no headers (`sudo` needs a password), so the factory
  deregisters the Triton DSL overrides (`torch._native.registry.deregister_op_overrides(
  disable_dsl_names="triton")`, DESIGN D19) and the forward pass runs on aten kernels.
- CARC gateway: LiteLLM (`ghcr.io/berriai/litellm:main-latest`, `drop_params: true`,
  `request_timeout: 600`, no DB, master key only) in front of vLLM 0.21.0 backends; owned by
  account `tredfear` (mesa-nmdid DECISIONS #D28). Reached only over loopback tunnels:
  `127.0.0.1:8000` (`carc-litellm-tunnel.service` user unit) and `127.0.0.1:18000`
  (mesa-mcp `scripts/llm_tunnel.sh`, present only on that repo's `live-llm-tests` branch).
  Both answered 200 on 2026-09-25 (`/health/liveliness`).
- Served models (`GET /v1/models`, `~/github/carc-agents/models/*.yaml`): `carc-fast` =
  `RedHatAI/Qwen3-8B-NVFP4` rev `e391349c110709b87bfc2ad2fde3f50dc5839fd8` (40k ctx,
  `--max-num-seqs 8`, prefix caching on, thinking-off honoured, no `--max-logprobs`);
  `carc-tools` = `unsloth/Qwen3.8-27B-NVFP4` (~5 tok/s, 4 seqs); `ab-moe` = Qwen3.6-35B-A3B
  NVFP4 (no thinking-off, hung 2026-09-24); `carc-embed` = `Qwen/Qwen3-Embedding-0.6B`
  (`--runner=pooling`, normalized; a different model, not usable for L2); `zammad-assist`.
- Gateway probes on `carc-fast` (2026-09-25, `/v1/completions`, `max_tokens=1`): `logprobs=N`
  passes through (`top_logprobs` over the full vocabulary); `logprobs=20` works, `logprobs=50`
  -> HTTP 400 "greater than max allowed: 20"; `allowed_token_ids` is silently ignored both
  top-level and inside `extra_body`; `/v1/chat/completions` with `logprobs=true,
  top_logprobs=5` also works; `chat_template_kwargs.enable_thinking=false` passes through.
  Consequence: a label outside the top-20 floors at -30 in AnyJev's `VLLMBackend`, and the L0
  log-mean over cyclic shifts then erases the option. `stale_after: 2026-12-31`.
- First engine run (2026-09-25, `bet_sorting`, carc-fast, L0, static planner, OLS fixtures
  auto): 82 decisions from 124 prefills in 5.1 s; `column.annotate` mean p_true 0.22 over 10
  columns with a phrasing-flip rate of 0.30 at L0; `column.ontology_fits` said Yes to nearly
  every ontology (mean 0.98) so the twin is not discriminative before calibration; 4 of 124
  prefills had a label outside the top-20 and their batches were capped at abstain (D11).
  Numbers from the run's sidecar rows, not a bench file; the bench arrives in M2.
- First L1 fit through the gateway (2026-09-25, `term.fits`, carc-fast, 285 labelled states
  from the neon-avu-eval silver at min_weight 0.5: 86 consensus positives, 199 single-model
  negatives; leave-one-card-out over 7 cards, no fold skipped): pooled held-out accuracy
  0.618, ECE 0.231, Brier 0.459, NLL 0.650, coverage at 5% risk 0.077, at 10% risk 0.140;
  every fold froze its prior. Per fold accuracy 0.52 to 0.64 and cov@5% 0.02 to 0.18.
  Source: the promoted bundle's manifest, `~/.mesa/anyjev/artifacts/RedHatAI__Qwen3-8B-NVFP4/
  0190586d/v1/manifest.json` (local; bundles are release assets, not git). Against the
  pre-registered criteria (DESIGN.md § Plan): coverage exists but ECE misses the 0.10 bar, so
  every `auto` threshold stays null (proposed-only). Read as agreement with the four agentic
  models, not correctness.
- First gateway bench (2026-09-25, `bench/results/2026-09-25/RedHatAI__Qwen3-8B-NVFP4.gateway.json`,
  carc-fast, raw/L0/L1 with leave-one-card-out): `term.fits` acc 0.502 / 0.611 / 0.621, ECE
  0.372 / 0.262 / 0.072, cov@5% 0.004 / 0.007 / 0.007 (n 285, n_neg 199); `column.aspect`
  (K=8) lost 19 labels to the top-20 readout at L0 (7 at raw); the choice26 control at L0
  scored 0.205 on 44 items; `avu.keep` fits nothing (11 negatives). The bench's L1 and `learn fit`'s
  L1 disagreed on ECE (0.072 vs 0.231) on the same labels; M3 traced it to `learn fit`
  averaging per-fold ECE and coverage instead of pooling the held-out decisions (the per-fold
  values agree exactly), fixed in `learn/fit.py` (D18). Thresholds stay proposed-only.
- First local bench (2026-09-25, `bench/results/2026-09-25/Qwen__Qwen3-8B.hf.json`,
  `Qwen/Qwen3-8B` bf16 on the GB10, raw/L0/L1/L2 with leave-one-card-out, 7 tasks, 1 h 50 min
  while sharing the GPU with the fit and two annotate runs): `term.fits` acc 0.642 / 0.663 /
  0.656 / 0.765, ECE 0.316 / 0.293 / 0.079 / 0.058, cov@5% 0.074 / 0.077 / 0.049 / 0.088,
  cov@10% at L2 0.396 (n 285, n_neg 199); `column.ontology_fits` at L2 acc 0.837, ECE 0.078,
  cov@5% 0.547 (n 190, n_neg 114) against 0.426 / 0.233 / 0.089 at L1; `column.annotate` at
  L2 acc 0.765 on 5 negatives; choice26 control 0.114 at raw and L0; `avu.keep` fits nothing.
  Local L0 vs gateway L0 on the same items: term.fits 0.663 vs 0.611 with phrasing-flip 0.119
  vs 0.393. The `prompts` column is 0 (the local backend had no counter; added after the run).
  The results file was written under the gateway slug by a CLI bug fixed in the same
  milestone and renamed by hand; its `environment.model` is `Qwen/Qwen3-8B`.
- First L2 fit on local weights (2026-09-25, `term.fits`, `Qwen/Qwen3-8B` bf16 on the GB10,
  the same 285 labelled states as the gateway L1 fit, leave-one-card-out over 7 cards, no
  fold skipped): LDA head on block 25 of 36; pooled held-out accuracy 0.765, ECE 0.058,
  Brier 0.327, NLL 0.499, coverage at 5% risk 0.088, at 10% risk 0.396; per fold accuracy
  0.71 to 0.85 and ECE 0.15 to 0.26. Source: `.local/artifacts/Qwen__Qwen3-8B/0190586d/v2/
  manifest.json` (promoted as CURRENT; local), identical to the bench's `neon_term_fits.L2`
  cell as D18 requires. Against the gateway L1 (acc 0.618, ECE 0.231 fold-averaged, 0.072
  pooled; cov@5% 0.077) the head is more accurate and meets the 0.10 ECE bar, but coverage
  at 5% risk is 25 items, so `auto` thresholds stay null. The fit ran three times: a CLI bug
  keyed the first bundle under the gateway slug (deleted), and the second (v1) reported
  fold-averaged ECE 0.193 and cov@5% 0.389 before the pooling fix; v2 supersedes it.
- First composite run (2026-09-25, `bet_sorting`, static planner, OLS fixtures auto, level
  auto, bundle above loaded): gateway logprobs for the L0 questions and the local head for
  `term.fits`; the sidecar records `backend_kind=composite`, `canonical_model=Qwen/Qwen3-8B`,
  `served_model=carc-fast`, 26 `term.fits` decisions at L2 and every other question at L0
  (`column.annotate` 20 rule rows at level none); 34 gateway prompts, 0 missing labels,
  13.1 s. Numbers from `.local/prov-m3c.duckdb`, not a bench file.
- Any gateway change (`--max-logprobs`, a pooling instance of the decision model, a
  text-completion alias, direct vLLM ports) is a request to `tredfear`.

## AnyJev (commit 795a497)

- `Question.choice` (2..26 options), `.noul` (Yes/No), `.score` (2..10 levels);
  `q.key = sha256(kind, text, options, scale, centers)[:16]`, name excluded.
- `Decider(backend, level='L0'|'L1'|'L2'|'auto', prior='batch', min_prior_n=8,
  adaptive_shifts=False, adapt='routed', adapt_min_n=30)`; `decide`, `decide_batch`,
  `calibrate` (L1 artifact), `fit_head` (L2; needs `hidden_states`; minimum `max(8, 2K)`
  labels), `observe(fit_at=30, refit_factor=2.0)`, `route`, `export_artifacts` /
  `load_artifacts` (refuse a model-name mismatch; `save_artifacts` drops observations).
- `route()` matches the exact key, else any head with the same kind and option texts (no text
  check): every noul routes to any fitted noul head. Level `auto` picks L2 whenever `route()`
  is truthy.
- `calibrate()` with `adaptive_shifts=True` copies `decs[0].diagnostics['prior']`, which is
  None for the first states of an isolated run, so the artifact freezes no prior.
- The batch prior is a running per-process accumulator (`_running`), applied once 8 same-key
  states have been seen; decisions are order-dependent until then. `Decider` is stateful and
  not thread-safe. `Decision.confidence = max(probs)`; for noul `p_true = probs[0]`.
- `Decision.diagnostics` carries numpy arrays and numpy scalars.
- `VLLMBackend(base_url, model, tokenizer_name=None, api_key='EMPTY', workers=16,
  timeout=120)`: appends `/v1/completions` itself (pass the base without `/v1`), loads the
  tokenizer without `revision=`, sends `logprobs=len(ids)` + `allowed_token_ids`, floors
  missing labels at -30, and `hidden_states()` POSTs `/v1/embeddings`. `name = model`.
- `bench/` and `demo/` are not in the wheel; `bench/metrics.py` (sha256
  `df0af62c248f27cc806c90266ae62c28b47df8dd8aa9645de815dedc50d7bbbe`) is vendored byte-identical
  as `mesa_anyjev/bench/_metrics.py`.
- Readout prompt: system "You are a decision function..." then `State:\n...\n\nQuestion:
  ...\nOptions:\nA. ...\nAnswer with the letter only.` rendered through the tokenizer chat
  template with `enable_thinking=False` (Qwen3 emits `<think>\n\n</think>`); the gateway probe
  reproduced that rendering.
- Research numbers used in this design: L2 needs 100-300 labels in practice (20 labels ~ L0,
  50 -> 0.707, 100 -> 0.740, 300 -> 0.772 on Qwen3-8B typed-decisions); K>8 heads keep the
  canonical listing order; teacher labels cap a student at the teacher's accuracy.

## mesa-mcp (main 8fbaedf) and mesa-ducklake (main 7bc143f)

- Tools register by import side effect (`server.py` lines ~244-248); `register_tool(name,
  description, *, input_model, output_model)`; `ToolSpec.meta` exists but `_tool_definitions`
  always calls `_tool_surface(name)`; `MesaServer(config).call(name, args)` is the in-process
  dispatch; `InputRequired(message, schema, state, key)` is the MRTR elicitation; requestState
  is unsigned, client-controlled, capped at 16 KiB.
- OLS term dict: `{label, iri, curie, description, ontologyId, isRoot, hasChildren,
  synonyms[:5]}`; no parents/ancestors API; `search_terms(ontology_id=...)` leaks imported
  terms (GO/CL/PR/UBERON under pato); `search_term_descendants` raises `requests.HTTPError`.
- `handle_avu_from_term` is `async def`; with iri+label and no curie the unit comes back
  empty, with curie alone it fails; `_label_to_snake` + `ontology_annotations_to_avus` are the
  sync pieces that produce `attribute='<ontology>.<snake_label>', unit=CURIE`.
- `record_avu_change(s)` return `None` (the Snapshot is discarded); `_resolve_mirror_target` is
  private and needs a session.
- mesa-ducklake: `DuckLakeClient(catalog_dsn='duckdb:///...'|'postgresql://...', cache_dir=,
  irods_session=None)` runs local-only; `record_changes(project_id, actor, changes, note=,
  session=) -> Snapshot`; empty `changes` raises `ValueError`; `AvuChange` is `extra='forbid'`
  with `@field_validator('actor', 'source')` rejecting blanks; the DuckDB catalog is
  single-writer (file lock); `tests/llm_e2e/harness/mcp_server.py` provides `ElicitationBroker`
  with `Chooser = Callable[[message, requestedSchema], Awaitable[content | None]]`.
- 50 registered tools on main (irods 35, ontology 8, datacite 4, policy 2, history 1).
- DataCite enums (`mesa_mcp.datacite.schema`): ResourceTypeGeneral 28, ContributorType 21,
  RelationType 23, RelatedIdentifierType 15, DateType 10, DescriptionType 6, NameType 2.

## mesa-mcp integration facts (main 8fbaedf, read 2026-09-25 for M4)

- `register_tool(name, description, *, input_model, output_model)` (server.py); handlers
  are `async def h(args: Model, auth_value=None, elicited=None) -> dict`, the two keywords
  injected only when declared. `ToolSpec.meta` existed but was never populated; `_meta` on
  the wire was only `{"io.mesa/surface": _tool_surface(name)}` (prefix rule; anything else
  is `core`). No entry-point loading anywhere: tools register through side-effect imports of
  the four subsystem packages. PR #6 (feat/plugin-entry-points) adds `register_tool(meta=)`,
  `load_plugins()` over the `mesa_mcp.tools` group and `server.strict_plugins`.
- MRTR: `InputRequired(message, schema, state, key)`; the server encodes `state` as compact
  JSON in urlsafe base64, unsigned, at most 16 KiB (32 KiB on decode); on resume the handler
  sees `elicited={"responses": {key: {"action", "content"}}, "state": {...}}`. mesa-mcp's own
  picker (`mesa_avu_apply_term`) offers at most 8 candidates as an `iri` enum with
  `enumNames`, message `Which ENVO term describes 'value'?`, and accepts only an IRI it
  offered; decline or cancel raises `invalid_argument`.
- `record_avu_change(s)` still return `None` on main (mesa-mcp PR #5 open). The mirror uses a
  process-wide `DuckLakeClient` singleton (`get_default_client()`), so `mesa_decide_apply`
  passes that client to its own `record_changes` and keeps the `mesa-anyjev:` source tag.
- Conformance (`tests/test_spec_conformance_2026_07_28.py`) asserts per tool: a 2020-12
  `$schema` on the input (and any output) schema that passes `check_schema`, a non-empty
  `_meta` with `io.mesa/surface`. mesa-anyjev's five tools pass it in mesa-mcp's own suite
  when installed beside it (468 passed with the loader branch).
- Postgres sidecar verified 2026-09-25 against `postgres:16` in Docker (round trip, CHECK
  constraint, ON CONFLICT label dedup) with psycopg 3. No iRODS credentials exist on this
  host, so the `e2e` chooser tier is written and gated but not yet run.

## neon-avu-eval (labels and states)

- 7 dataset cards (DP1.10003.001 brd_countdata, brd_perpoint; DP1.10022.001 bet_*), 42 runs
  (carc-tools 7, claude-haiku-4-5 7, claude-opus-5-5 14, claude-sonnet-5 14).
- `results/validated.json`: 572 valid AVUs, 493 column-linked, 303 unique valid (card, CURIE)
  candidate states, 92 proposed by >= 2 models, 211 by one model. Value kinds: 264 term label,
  128 site code, 44 column name, 136 other.
- Resolving CURIE prefixes (valid, not obsolete): ENVO 171, NCBITaxon 94, OBI 82, PATO 61,
  UO 51, GAZ 31, BCO 27, PCO 16, IAO 15, TAXRANK 7, GENEPIO 5, GO 3, EUPATH 2, AGRO 2,
  APOLLO_SV 1, OBCS 1, NCIT 1, GECKO 1, CHEBI 1, RO 0.
- Agreement (`results/metrics.json`): cross-model Jaccard 0.11-0.23 (soft-F1 0.86-0.90 against a
  0.81 floor); self-consistency Opus 0.50, Sonnet 0.39. NEON data are CC BY 4.0.

## Claude API (claude-api skill, 2026-09-25)

- No logprobs, no `n`, no public tokenizer, no prefill on 4.6+ models: Claude cannot implement
  the AnyJev Backend protocol at any level.
- Forced `tool_choice` (`any`/`tool`) returns 400 on Claude Fable 5.1 and Opus 5.5; structured
  outputs (`output_config.format` json_schema, or `client.messages.parse(output_format=...)`)
  work with thinking and Batches. Default model `claude-opus-5`, adaptive thinking,
  `output_config.effort`. `temperature` is rejected on Opus 5 / 5.5 / Sonnet 5 / Fable.
- Credentials resolve through the SDK: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an
  `ant auth login` profile (an `idss-mesa` OAuth profile exists on this host).

## MotherDuck `prompt_jev` (docs read 2026-09-25; preview, `stale_after: 2026-12-31`)

- MotherDuck SQL only (not local DuckDB), Lite or Business plan, us-east-1 / us-west-2.
- `prompt_jev(text_input, instructions, choice := [labels|{label, description}])`,
  `score := [ordered levels]`, `noul := TRUE`, or `questions := {name: {type, instructions}}`;
  `instructions` and option lists must be SQL constants; optional `batch_size` (32 rows per
  request for single questions). Metered on input tokens (blog benchmark: $0.50 per 100k rows,
  ~2,484 rows/s). NULL input or a failed request yields NULL and the query continues.
- Returns `STRUCT(choice VARCHAR, probabilities STRUCT(value VARCHAR, probability DOUBLE)[],
  confidence DOUBLE)` for choice, a weighted position for score, a DOUBLE for noul. Guidance:
  confidence under ~0.6 signals overlapping or missing options; keep criteria disjoint and
  phrased as questions.
- MotherDuck can attach `ducklake:` catalogs (managed DuckLake in preview). The `motherduck`
  DuckDB extension is not installed in `~/.mesa/.venv`'s duckdb 1.5.5 today.

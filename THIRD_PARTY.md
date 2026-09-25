# Third-party code and data

| Component | Source | License | Use here |
|---|---|---|---|
| AnyJev | https://github.com/nokia-applied-research/AnyJev @ 795a4970b47218b7c0686cd579d691fc2cf8df2f | Apache-2.0 | Runtime dependency (typed decisions, calibration, backends). Not affiliated with TypeSafe AI or Jev. |
| AnyJev `bench/metrics.py` | same commit; upstream sha256 `df0af62c248f27cc806c90266ae62c28b47df8dd8aa9645de815dedc50d7bbbe` | Apache-2.0 | Vendored byte-identical as `src/mesa_anyjev/bench/_metrics.py` (`bench/` is not in the AnyJev wheel). Apache-2.0 §4: the copy is unmodified; attribution is given here rather than in a header the upstream file does not carry. |
| AnyJev `bench/tasks/base.py` pattern | same commit | Apache-2.0 | `Task` / `register` / `get_task` reimplemented (about a dozen lines) in `src/mesa_anyjev/bench/tasks/base.py`. |
| neon-mcp scripts | https://github.com/idss-mesa/neon-mcp (`scripts/okf_validate.py`, `gen_llms_txt.py`, `postbuild_agent_surface.py`) | MIT (Regents of UNM) | Documentation tooling, adapted (site name and description strings). |
| mesa-mcp, mesa-ducklake | https://github.com/idss-mesa/mesa-mcp, https://github.com/idss-mesa/mesa-ducklake | MIT (Regents of UNM) | Runtime dependencies (OLS client and AVU transform; DuckLake client). |
| NEON data (DP1.10003.001, DP1.10022.001, RELEASE-2026) via neon-avu-eval dataset cards and validated results | https://data.neonscience.org | CC BY 4.0 | Bench states and silver labels; referenced by path (`MESA_ANYJEV_EVAL_ROOT`), not vendored. |
| EMBL-EBI OLS4 responses | https://www.ebi.ac.uk/ols4 | OBO Foundry ontologies, each under its own license (mostly CC BY 3.0/4.0, CC0) | Recorded fixtures under `tests/fixtures/ols/` for hermetic tests. |
| MotherDuck `prompt_jev` | https://motherduck.com/docs/key-tasks/ai-and-motherduck/classify-text-with-prompt-jev/ | MotherDuck terms of service (preview feature) | Optional hosted provider (M4b); nothing vendored. |

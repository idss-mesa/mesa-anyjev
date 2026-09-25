# Agent guide — mesa-anyjev

This repository holds **mesa-anyjev** (package `mesa_anyjev`, script `mesa-anyjev`, tests
under `tests/`) and its documentation site, built with [Zensical](https://zensical.org) from
`docs/`, an **Open Knowledge Format (OKF) v0.2** bundle: every content page carries YAML
frontmatter (`type`, `title`, `description`, `tags`, `generated`, `sources`, optional
`status`/`stale_after`); section `index.md` files carry none; `docs/log.md` is the dated log.

## Ground rules (adopted from AnyJev's AGENTS.md, plus ours)

1. **No fabricated numbers, ever.** Bench numbers come from committed `bench/results/<date>/`
   JSON; every number in prose names its JSON. If a run did not happen, the cell is empty.
2. **No hidden generation in decision mode.** Any code path that samples tokens to answer a
   question is a bug. Planners generate; deciders read logits or hosted probabilities.
3. **Level is mandatory.** Every decision record carries `level` (raw / L0 / L1 / L2 / none)
   and `calibration` (anyjev / typesafe / none). `auto` is a request, never a stored level.
4. **Backends are thin.** A backend exposes a tokenizer, a name and `next_token_logprobs`;
   debiasing, calibration and policy live above it.
5. **Licenses are checked** before any dataset or third-party code is used, and recorded in
   `THIRD_PARTY.md`.
6. **Thresholds cite their evidence.** A numeric write threshold names the LOCO bench cell it
   came from; a test refuses uncited numbers.
7. **Data stays on the host unless a project opted in.** Hosted providers are off by default
   and refused for projects that are not allow-listed; nothing from a DUA-gated project ever
   reaches a hosted model.
8. **Git is shared.** Commit on feature branches, Conventional Commits, PRs; never amend a
   shared branch.

## Definition of done

Code + tests + docstring + a `CHANGELOG.md` line; a `DESIGN.md` entry for any new decision;
`docs/log.md` entry for any docs change; `questions.lock.json` updated (with a key-rotation
note) whenever a question's wording or options change; `ruff`, `mypy --strict`, `pytest -q`
green.

## Documentation commands

```bash
python scripts/okf_validate.py docs           # OKF conformance (CI-enforced; 0 errors)
python scripts/gen_llms_txt.py                # regenerate docs/llms.txt + docs/llms-full.txt
zensical build --clean --strict               # static site -> site/
python scripts/postbuild_agent_surface.py site
```

Editing rules are those of neon-mcp's AGENTS.md: frontmatter on every content page,
`index.md` listings without frontmatter, a dated `docs/log.md` entry per change, generated
files regenerated and committed, new pages added to `zensical.toml`'s `nav`, never add
`verified:` yourself.

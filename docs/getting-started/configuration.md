---
title: "Configuration"
description: "Configure mesa-anyjev with a YAML file, MESA_ANYJEV_ environment variables or flags; precedence and the mesa-mcp fallbacks."
type: Reference
tags:
  - getting-started
  - configuration
  - environment
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

# Configuration

Precedence, highest first: command-line flag, environment variable, YAML file
(`--config FILE`), built-in default. Environment variables use the `MESA_ANYJEV_` prefix and
`__` to descend into a section: `MESA_ANYJEV_BACKEND__KIND=gateway` sets `backend.kind`.

Two mesa-mcp names are honoured as fallbacks so an existing `mesa-mcp/.env` keeps working:
`MESA_LLM_BASE_URL` (a trailing `/v1` is stripped) and `MESA_LLM_API_KEY`. Never `source` that
file in a shell: it contains a hyphenated key; use
`export MESA_LLM_API_KEY=$(grep '^MESA_LLM_API_KEY=' ../mesa-mcp/.env | cut -d= -f2-)`.

The sections and every field with its default are in `config.yaml.example` and
`.env.example` in the repository; the model is `mesa_anyjev.config.Config`.

## Sections

| Section | What it configures |
|---|---|
| `backend` | The AnyJev backend: `fake`, `gateway` (carc-fast, always `logprobs=20`, `max_choice_k=8`), `hf` (local weights), `composite`. |
| `decider` | Level request (`auto`), prior, adaptive shifts. |
| `planner` | The reasoning model: `gateway` (carc-tools, default), `claude`, `static` (fallback). |
| `policy` | Profile (`prod` needs L1 for auto-writes), thresholds file, hosted-provider switch. |
| `artifacts`, `provenance`, `ducklake`, `ols`, `motherduck` | Bundle directory, sidecar DSN, local DuckLake catalog, OLS base URL and fixtures, hosted Jev settings. |
| `eval_root` | The neon-avu-eval checkout for silver labels and the bench (no default). |

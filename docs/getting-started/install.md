---
title: "Install"
description: "Install mesa-anyjev with uv, choose the extras for the gateway, local weights, Claude or Postgres, and run the doctor."
type: Guide
tags:
  - getting-started
  - install
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

# Install

```bash
git clone https://github.com/idss-mesa/mesa-anyjev
cd mesa-anyjev
uv sync --all-extras --no-extra hf     # everything except local CUDA weights
uv run mesa-anyjev doctor
```

Python 3.11 or newer. `anyjev`, `mesa-mcp` and `mesa-ducklake` come from pinned git commits
(`pyproject.toml`, `[tool.uv.sources]`); none of them is on PyPI at a usable version.

## Extras

| Extra | Adds | When |
|---|---|---|
| `gateway` | `transformers` (tokenizer only) | Deciding through the CARC LiteLLM gateway (L0, L1). |
| `hf` | `torch`, `transformers` | Local hidden states for L2 on a CUDA host. |
| `claude` | `anthropic` | The Claude planner and structured-output provider (M2+). |
| `pg` | `psycopg` | The Postgres provenance sidecar (production). |
| `e2e` | `mcp` | The end-to-end test over MCP stdio (M4). |

The gateway is reached only over a loopback SSH tunnel; see `RESEARCH.md` in the repository
for the hosts, the served models and what the gateway does and does not pass through.

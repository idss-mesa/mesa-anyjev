---
title: "Gateway"
description: "Decide with carc-fast on the CARC LiteLLM gateway at L0: the tunnel, the key, what the doctor checks, and what the gateway cannot do."
type: Guide
tags:
  - getting-started
  - gateway
  - carc
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

# Gateway

The gateway is LiteLLM in front of vLLM on the CARC hosts (RESEARCH.md). It is reached only
over a loopback SSH tunnel: `127.0.0.1:8000` (the `carc-litellm-tunnel` user unit) or
`127.0.0.1:18000` (mesa-mcp's `scripts/llm_tunnel.sh`). Never `source` mesa-mcp's `.env`;
take the key out of it:

```bash
export MESA_LLM_API_KEY=$(grep '^MESA_LLM_API_KEY=' ../mesa-mcp/.env | cut -d= -f2-)
uv run mesa-anyjev doctor --backend gateway
```

The doctor checks liveness, that the served tokenizer loads at the pinned revision, that
the answer labels are single tokens, that a two-option probe rendered through the chat
template comes back with both labels in the top-20, and that `logprobs=26` is refused (the
cap is 20). Then (`auto` replays recorded OLS responses and records the pairs a real model
searches for the first time):

```bash
MESA_ANYJEV_BACKEND__KIND=gateway MESA_ANYJEV_OLS__FIXTURES=auto uv run mesa-anyjev annotate \
    --level L0 --card tests/fixtures/cards/DP1.10022.001.bet_sorting.md \
    --provenance duckdb:///$PWD/.local/prov.duckdb --out .local/gateway_L0.json
```

What the gateway cannot do (DESIGN D11): it drops `allowed_token_ids`, so every request asks
for the top-20 logprobs and a label outside them is floored and counted as missing; a batch
with a missing label never auto-writes. It serves no hidden states, so L2 needs local
weights (milestone M3). Choices with more than eight options use their yes/no twin.

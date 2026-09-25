---
title: "Local weights"
description: "Run Qwen/Qwen3-8B on a CUDA host (the GB10 workstation) for hidden states and L2 heads, the composite backend that pairs it with the gateway, and the torch 2.14 Triton switch."
type: Guide
tags:
  - getting-started
  - hf
  - l2
  - gb10
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

# Local weights

The gateway reads a top-20 logprob list and has no hidden states, so it stops at L1. L2
(a closed-form head on a hidden state, cheaper than L0 because the forward pass stops at the
head's block) needs the weights on the host. The `hf` extra installs torch, transformers and
accelerate; on the GB10 workstation that resolves to torch 2.14 with CUDA 13 wheels for
aarch64 (RESEARCH.md § Hosts).

```bash
uv sync --all-extras                 # includes hf
uv run mesa-anyjev doctor --backend hf
```

The doctor loads `backend.hf_model` (default `Qwen/Qwen3-8B`, bf16, 16.4 GB; about two minutes
from the local safetensors cache, longer on the first download), reports `n_layers` and
`hidden_size`, checks that the answer labels are single tokens, runs a two-state probe and
requires the label tokens to carry the answer mass, and confirms the block loop for early
stop is available. Then, with the neon-avu-eval labels ingested (see [Learning](../concepts/learning.md)):

```bash
uv run mesa-anyjev bench run --backend hf --levels raw,L0,L1,L2
uv run mesa-anyjev learn fit --backend hf --question term.fits --level L2
uv run mesa-anyjev learn promote --backend hf --version <N> --question term.fits
uv run mesa-anyjev annotate --backend hf --level auto --card <card.md>
```

`annotate` and `bench` load the promoted bundle for the configured model before deciding, so
`--level auto` resolves `term.fits` at L2 when a head is promoted and every other question at
L0. A bundle fitted on local weights is refused by the gateway backend and the other way
round (DESIGN D5): the NVFP4 checkpoint the gateway serves and the bf16 checkpoint here are
recorded as different models until the bench shows they agree.

## Composite: gateway logprobs plus local hidden states

`--backend composite` builds both backends under one name. Logprobs come from the gateway
(carc-fast) and hidden states from the local model, so a single process can serve L0 and L1
from the gateway's artifacts and L2 from a local head. The constructor asserts that the two
tokenizers agree on every label token id (A..Z, Yes, No, 1..9) and refuses otherwise; the
composite still reads the gateway's top-20 list, so `max_choice_k` stays at 8 for L0 and L1
while L2 heads read the hidden state and ignore the cap. It needs the tunnel and the key like
the [gateway](gateway.md) page describes.

## torch 2.14 and Triton

torch 2.14 routes some eager aten ops (the rotary-embedding outer product in Qwen3) through
Triton kernels that are compiled against `Python.h` on first use. A host whose Python has no
development headers (the GB10 with the system CPython 3.12) fails inside the forward pass with
`fatal error: Python.h: No such file or directory`. `backend.hf_native_triton` is therefore
`false` by default: the factory deregisters torch's Triton DSL overrides before loading, and
the stock aten kernels run instead. Set it to `true` on a host with headers if you want the
Triton path.

## What the host can do

| Backend | Levels | Prompts per decision | Where |
|---|---|---|---|
| `gateway` | raw, L0, L1 | one prefill per cyclic shift | sparky-2 over the tunnel |
| `hf` | raw, L0, L1, L2 | one prefill, stopped at the head's block for L2 | this host, CUDA |
| `composite` | raw, L0, L1 (gateway) and L2 (local) | as above | both |
| `fake` | all (synthetic hidden states) | none | CI |

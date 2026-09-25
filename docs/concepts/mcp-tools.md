---
title: "MCP tools"
description: "The mesa_decide_* tools inside mesa-mcp: annotate, apply with one question per candidate group, explain, feedback and health; how they register, and what the elicitation state carries."
type: Guide
tags:
  - concepts
  - mcp
  - tools
  - elicitation
generated:
  by: "claude/fable-5.1"
  at: "2026-09-25T00:00:00Z"
sources:
  - id: design
    resource: "https://github.com/idss-mesa/mesa-anyjev/blob/main/DESIGN.md"
    title: "mesa-anyjev decisions register (DESIGN.md)"
    author: "team:idss-mesa"
  - id: mesa-mcp-plugins
    resource: "https://github.com/idss-mesa/mesa-mcp/pull/6"
    title: "mesa-mcp: load third-party tools from mesa_mcp.tools entry points"
    author: "team:idss-mesa"
status: draft
---

# MCP tools

mesa-anyjev is a plugin inside mesa-mcp's registry, not a second server (DESIGN D14). The
package declares a `mesa_mcp.tools` entry point (`decide = "mesa_anyjev.mcp_tools"`); once
mesa-mcp's loader is merged, installing mesa-anyjev next to mesa-mcp adds five tools with the
surface tag `decision` in their `_meta`. Until then, `import mesa_anyjev.mcp_tools` before
`MesaServer()` registers them under the surface `core`.

| Tool | Phase | What it does |
|---|---|---|
| `mesa_decide_annotate` | decide | Runs the decision graph on a dataset card (`card_text` or `card_path`), records every decision and proposal in the sidecar, writes nothing. |
| `mesa_decide_apply` | write | Writes accepted AVUs to an iRODS path and mirrors them into the DuckLake (one snapshot per run and path). `accept="proposed"` asks the user to pick for each open candidate group first. `dry_run=True` by default. |
| `mesa_decide_explain` | read | Every decision of a run with its level, calibration and probability, the AVU links with write status, the open groups; or the decisions behind the AVUs on a path. |
| `mesa_decide_feedback` | curate | A pick, a reject or a decline on a candidate group; authoritative, stored as an override and as labels. |
| `mesa_decide_health` | ops | Questions lock, provenance store, backend and promoted bundle (local weights are not loaded here). |

Handlers receive the iRODS `auth_value` mesa-mcp injects; the authenticated user is the
actor, and the write goes through mesa-mcp's own `assert_allowed` and AVU helpers with the
session from its pool. Without an authenticated user the tool runs in local mode and writes
only to the sidecar.

## One question per round trip

`mesa_decide_apply(accept="proposed")` uses mesa-mcp's Multi Round-Trip Requests: for the first
open group it raises an elicitation with key `term_choice:<group_id>`, a form whose options are
the group's candidate decision ids and whose names read `label (CURIE) p(fits)=0.83 L1` (an
unranked group says so instead of showing a number). The request state carries ids only
(`tool`, `run_id`, `asked`), never a path, a label or a probability: on resume every label,
probability and level is read back from the sidecar (`candidates_for_group`), and an answer
whose id was not offered, or a state that belongs to another run, is refused (DESIGN D8). A
decline leaves the group unwritten. When no group is open the call writes what is accepted.

## What a pick does

A pick is authoritative (outcome `human`): the chosen candidate's link becomes `accepted`
(a candidate that was not the winner gets a fresh link built from the same value rule), the
other proposed links are `rejected`, the group's winner is updated, an override row records
who chose what from which offered list, and labels are written for the next `learn fit`:
the pick as a curator positive (weight 1.0), the other offered candidates as implicit
negatives (0.7), and an explicit "none of these" as curator negatives (1.0). Serving never
fits (D15).

## The chooser for mesa-mcp's own picker

`mesa_avu_apply_term` has an eight-candidate picker of its own. `ElicitationChooser` answers
it over the real protocol with the fixed-key question `term.fits.chooser` over the only state
the picker carries (ontology, value, candidate), and returns the best candidate when p(fits)
clears the `propose` threshold, otherwise a decline. It never reuses `term.fits` artifacts
(plan amendment B13). `tests/e2e/test_chooser_harness.py` runs it through mesa-ducklake's
llm_e2e broker against a live iRODS zone.

---
title: "DataCite decisions"
description: "The frozen DataCite vocabularies as questions: resource type as a yes/no per member, the others as a choice with a yes/no twin, and the record fragment for mesa_avu_apply_datacite."
type: Guide
tags:
  - concepts
  - datacite
  - questions
generated:
  by: "claude/fable-5.1"
  at: "2026-09-25T00:00:00Z"
sources:
  - id: design
    resource: "https://github.com/idss-mesa/mesa-anyjev/blob/main/DESIGN.md"
    title: "mesa-anyjev decisions register (DESIGN.md)"
    author: "team:idss-mesa"
  - id: datacite
    resource: "https://schema.datacite.org/"
    title: "DataCite Metadata Schema"
    author: "DataCite"
status: draft
---

# DataCite decisions

mesa-mcp writes DataCite records as AVUs (`mesa_avu_apply_datacite`) and validates them
against the DataCite 4.x controlled vocabularies. Choosing a value from those vocabularies
is a closed choice, so M5 freezes five of them in `registry.py` (a test asserts parity with
mesa-mcp's enums) and asks them like any other question:

| Vocabulary | Members | Question | Shape |
|---|---|---|---|
| ResourceTypeGeneral | 28 | `datacite.resource_type_fits` | yes/no per member (28 exceeds the 26-option cap) |
| ContributorType | 21 | `datacite.contributor_type` (+ `_fits` twin) | choice under a wide backend, twin under the gateway's cap of 8 |
| RelationType | 23 | `datacite.relation_type` (+ twin) | same |
| DateType | 10 | `datacite.date_type` (+ twin) | same |
| DescriptionType | 6 | `datacite.description_type` | choice |

The state is the text being classified (a description, a contributor line, a related
identifier, a date), the vocabulary name and, for the twins, the candidate value, with the
card header when a card is at hand. Levels, calibration and thresholds work as for every
other question; there are no labels yet, so every threshold is `auto: null` and the answers
are proposals.

```bash
uv run mesa-anyjev datacite --card tests/fixtures/cards/DP1.10022.001.bet_sorting.md
uv run mesa-anyjev datacite --vocabulary DescriptionType --text "We sampled beetles with pitfall traps."
```

`datacite --card` answers what a dataset card can answer on its own (the general resource
type of the product and the description type of its description) and prints the
`record` fragment in the shape `mesa_avu_apply_datacite` takes, with the ranked
alternatives beside it.

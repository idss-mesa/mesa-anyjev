---
okf_version: "0.2"
title: mesa-anyjev
description: "mesa-anyjev makes MESA's ontology and schema choices declarative and calibrated: AnyJev typed decisions over mesa-mcp's OBO/OLS tools, with provenance next to the mesa-ducklake AVU history."
---

# mesa-anyjev

**mesa-anyjev** turns every choice on the way from a dataset to an ontology-grounded AVU into
a typed question with a probability and an explicit calibration level, using
[AnyJev](https://github.com/nokia-applied-research/AnyJev){target=_blank} over an open model.
A reasoning model only *plans* (which ontologies are in play, what to search); deterministic
code drives [mesa-mcp](https://github.com/idss-mesa/mesa-mcp){target=_blank}'s OBO/OLS tools;
AnyJev answers the multiple-choice steps; every decision is recorded in a provenance sidecar
next to the AVU history kept by
[mesa-ducklake](https://github.com/idss-mesa/mesa-ducklake){target=_blank}. Hosted Jev through
MotherDuck's `prompt_jev` is a third, policy-gated provider for the same questions.

It is developed by [idss-mesa](https://github.com/idss-mesa){target=_blank} at the University
of New Mexico and released under the MIT license. Status: pre-alpha (milestone M0).

## Documentation

This site is an [Open Knowledge Format (OKF) v0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md){target=_blank}
knowledge bundle; the corpus is available to agents at [`llms.txt`](llms.txt) and
[`llms-full.txt`](llms-full.txt).

* [Getting started](getting-started/index.md) - Install mesa-anyjev, reach the CARC gateway and configure it.
* [Concepts](concepts/index.md) - The decision graph, calibration levels and the write policy, provenance, hosted providers.
* [Develop](develop/index.md) - Architecture and the hermetic test suite.
* [About](about/index.md) - How agents should consume this site, licenses, and the change log.

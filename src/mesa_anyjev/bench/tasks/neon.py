"""Bench tasks over the neon-avu-eval labels (NEON data CC BY 4.0; eval outputs MIT).

Every task reads its items from the provenance ``labels`` table (``mesa-anyjev learn ingest``
first) so the bench and the fits see the same states. ``neon_term_choice26`` is the control:
one ``Question.choice`` per (card, column) over the recorded candidates with a new key per
item, so it can only ever run raw/L0 and, on the gateway, shows the top-20 degradation.
"""

from __future__ import annotations

import collections
from typing import Any

from anyjev import Question

from mesa_anyjev.bench.tasks.base import Task
from mesa_anyjev.learn.labels import labelled_states
from mesa_anyjev.provenance.store import ProvenanceStore
from mesa_anyjev.questions import QUESTIONS

LICENSE = (
    "NEON data CC BY 4.0 (DP1.10003.001, DP1.10022.001, RELEASE-2026); "
    "labels are agreement between four agentic models"
)
SOURCE = "neon-avu-eval/results/validated.json"


def _task(
    store: ProvenanceStore, name: str, question_id: str, *, min_weight: float = 0.0, notes: str = ""
) -> Task:
    q = QUESTIONS[question_id].question
    ls = labelled_states(store, q, min_weight=min_weight)
    return Task(
        name,
        q,
        list(zip(ls.states, ls.labels, strict=True)),
        LICENSE,
        SOURCE,
        notes=notes or f"{len(ls)} labelled states from the eval; class counts {ls.class_counts()}",
        cards=ls.cards,
        weights=ls.weights,
        meta={
            "question_id": question_id,
            "min_weight": min_weight,
            "class_counts": ls.class_counts(),
        },
    )


def tasks_from_store(store: ProvenanceStore) -> dict[str, Task]:
    out = {
        "neon_term_fits": _task(store, "neon_term_fits", "term.fits", min_weight=0.5),
        "neon_annotate": _task(store, "neon_annotate", "column.annotate", min_weight=0.5),
        "neon_aspect": _task(store, "neon_aspect", "column.aspect", min_weight=0.6),
        "neon_ontology_for_column": _task(
            store, "neon_ontology_for_column", "column.ontology", min_weight=0.6
        ),
        "neon_ontology_fits": _task(
            store, "neon_ontology_fits", "column.ontology_fits", min_weight=0.5
        ),
        "neon_keep_avu": _task(store, "neon_keep_avu", "avu.keep", min_weight=0.5),
        "neon_value_kind": _task(store, "neon_value_kind", "avu.value_kind", min_weight=0.5),
    }
    out["neon_term_choice26"] = term_choice26(store)
    return {k: v for k, v in out.items() if v.items}


def term_choice26(store: ProvenanceStore) -> Task:
    """The control: per (card, column), one choice over that column's labelled candidates
    (plus "none of these"), gold = the consensus candidate. One key per item."""
    q = QUESTIONS["term.fits"].question
    ls = labelled_states(store, q, min_weight=0.5)
    groups: dict[tuple[str, str], list[tuple[dict[str, Any], int]]] = collections.defaultdict(list)
    for state, label in zip(ls.states, ls.labels, strict=True):
        col = (state.get("column") or {}).get("name") or state.get("scope", "")
        groups[(state["card"]["dataset"], str(col))].append((state, label))
    items: list[tuple[Any, int]] = []
    cards: list[str] = []
    for (card, col), members in sorted(groups.items()):
        positives = [m for m in members if m[1] == 0]
        if not positives or len(members) < 2:
            continue
        members = members[:25]
        options = [
            f"{m[0]['candidate']['label']} ({m[0]['candidate']['curie']})" for m in members
        ] + ["none of these"]
        if len(set(options)) != len(options):
            continue
        gold = next(i for i, m in enumerate(members) if m[1] == 0)
        base = dict(members[0][0])
        base.pop("candidate", None)
        base.pop("n_candidates", None)
        item_q = Question.choice(
            "Which candidate term is the correct annotation for the described column, site or dataset?",
            options,
            name=f"choice26:{card}:{col}",
        )
        items.append(({"question": item_q, "state": base}, gold))
        cards.append(card)
    return Task(
        "neon_term_choice26",
        q,
        items,
        LICENSE,
        SOURCE,
        notes="control: one choice per (card, column); a new key per item, raw/L0 only",
        levels_supported=("raw", "L0"),
        cards=cards,
        meta={"question_id": "term.fits", "per_item_question": True},
    )

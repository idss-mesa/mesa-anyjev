from __future__ import annotations

import json

from mesa_anyjev.cards import DatasetCard
from mesa_anyjev.states import (
    avu_state,
    candidate_state,
    card_header,
    chooser_state,
    column_state,
    state_sha256,
    value_kind_state,
)


def _json_safe(obj: object) -> None:
    json.dumps(obj)  # raises on numpy, datetime, sets


def test_states_are_json_and_share_the_card_header(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    cand = {
        "label": "distance",
        "curie": "PATO:0000040",
        "ontology_id": "pato",
        "description": "x" * 500,
        "synonyms": ["a", "b", "c", "d", "e", "f"],
        "has_children": True,
    }
    states = [
        column_state(card, col),
        candidate_state(card, "column", col, "measurement", cand, 7),
        candidate_state(card, "site", card.sites[0], "environment", cand, 3),
        value_kind_state(card, col, cand, "measurement"),
        avu_state(
            card,
            {"attribute": "pato.distance", "value": "distance", "unit": "PATO:0000040"},
            cand,
            "column",
            col.name,
            [],
        ),
        chooser_state("pato", "distance", cand, 8),
    ]
    for s in states:
        _json_safe(s)
        assert len(state_sha256(s)) == 64
    header = card_header(card)
    for s in states[:5]:
        assert s["card"] == header
    assert len(states[1]["candidate"]["description"]) == 300
    assert len(states[1]["candidate"]["synonyms"]) == 5
    assert "site" in states[2] and "column" not in states[2]


def test_state_hash_is_key_order_independent(card: DatasetCard) -> None:
    a = {"card": card_header(card), "x": 1}
    b = {"x": 1, "card": card_header(card)}
    assert state_sha256(a) == state_sha256(b)

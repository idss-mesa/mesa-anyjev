from __future__ import annotations

from mesa_anyjev.cards import DatasetCard
from mesa_anyjev.planner.base import Plan
from mesa_anyjev.planner.gateway_planner import _coerce, parse_json
from mesa_anyjev.planner.static_planner import (
    StaticPlanner,
    split_camel,
)


def test_static_plan(card: DatasetCard) -> None:
    result = StaticPlanner().plan(card)
    plan = result.plan
    assert result.planner == "static" and not result.fallback
    assert len(plan.ontologies) == 12
    assert plan.columns["uid"].annotate is False
    assert plan.columns["observerDistance"].annotate is None
    assert plan.columns["observerDistance"].queries[0] == "observer distance"
    assert "meter" in plan.columns["observerDistance"].queries
    assert plan.sites["HARV"].environment_queries[0].startswith("temperate deciduous")
    assert plan.sites["SRER"].environment_queries[:2] == ["desert biome", "xeric shrubland biome"]
    assert plan.taxon_queries == ["Aves"]


def test_helpers() -> None:
    assert split_camel("scientificName") == "scientific name"
    assert split_camel("taxonID") == "taxon id"


def test_plan_schema_drops_unknown_enums() -> None:
    plan = Plan.model_validate(
        {
            "ontologies": ["ENVO", "bogus", "envo"],
            "columns": {
                "x": {"aspect": "nope", "ontology": "NOPE", "queries": ["a", "b", "c", "d"]}
            },
        }
    )
    assert plan.ontologies == ["envo"]
    assert plan.columns["x"].aspect is None and plan.columns["x"].ontology is None
    assert plan.columns["x"].queries == ["a", "b", "c"]


def test_parse_json_and_coerce() -> None:
    raw = (
        '<think>hmm</think>```json\n{"ontologies": ["envo"], '
        '"columns": [{"name": "c", "queries": "q"}], "sites": 3}\n```'
    )
    parsed = parse_json(raw)
    assert parsed is not None
    plan = Plan.model_validate(_coerce(parsed))
    assert plan.columns["c"].queries == ["q"] and plan.sites == {}
    assert parse_json("no json here") is None

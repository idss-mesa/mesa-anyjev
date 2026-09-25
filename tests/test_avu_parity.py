"""build_avu equals mesa-mcp's mesa_avu_from_term byte for byte (amendment B1)."""

from __future__ import annotations

import asyncio

from mesa_mcp.ols.tools.avu_from_term import AvuFromTermInput, handle_avu_from_term

from mesa_anyjev.avu import build_avu, pre_rule_value_kind, top_profile_value, value_for
from mesa_anyjev.ols import Candidate, iri_for


def test_parity_with_the_tool() -> None:
    cases = [
        ("envo", "temperate deciduous broadleaf forest", "ENVO:01000385", "HARV"),
        ("ncbitaxon", "Aves", "NCBITaxon:8782", "Aves"),
        ("uo", "degree Celsius", "UO:0000027", "degree Celsius"),
        ("pato", "has number of", "PATO:0001555", "clusterSize"),
    ]
    for ont, label, curie, value in cases:
        cand = Candidate(label=label, curie=curie, iri=iri_for(curie), ontology_id=ont)
        ours = build_avu(cand, value)
        theirs = asyncio.run(
            handle_avu_from_term(
                AvuFromTermInput(
                    ontology_id=ont, value=value, iri=cand.iri, curie=curie, label=label
                )
            )
        )["avu"]
        assert ours == theirs, (ours, theirs)
        assert ours["unit"] == curie and ours["attribute"].startswith(ont + ".")


def test_value_rules() -> None:
    assert pre_rule_value_kind("environment", "site") == "the site code"
    assert pre_rule_value_kind("unit", "column") == "the term label"
    assert pre_rule_value_kind("measurement", "column") is None
    assert (
        value_for(
            "the site code", term_label="x", column_name="c", site_code="HARV", top_value=None
        )
        == "HARV"
    )
    assert (
        value_for(
            "the column name", term_label="x", column_name="c", site_code=None, top_value=None
        )
        == "c"
    )
    assert (
        value_for(
            "the most frequent data value",
            term_label="x",
            column_name=None,
            site_code=None,
            top_value="singing",
        )
        == "singing"
    )
    assert (
        value_for("the site code", term_label="x", column_name=None, site_code=None, top_value=None)
        == "x"
    )
    assert top_profile_value("2 distinct; top: SRER (9416), HARV (6068)") == "SRER"
    assert top_profile_value("numeric, n=3") is None

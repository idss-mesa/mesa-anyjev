from __future__ import annotations

from mesa_anyjev.registry import (
    ASPECT_OPTIONS,
    ASPECTS,
    ONTOLOGY_OPTIONS,
    ONTOLOGY_REGISTRY,
    VALUE_KINDS,
    allowed_for_aspect,
    mask_for_aspect,
    prefix_of,
)


def test_registry_rule_d7() -> None:
    ids = [e.id for e in ONTOLOGY_REGISTRY]
    assert ids == [
        "envo",
        "ncbitaxon",
        "pato",
        "uo",
        "obi",
        "iao",
        "pco",
        "bco",
        "gaz",
        "ro",
        "taxrank",
        "genepio",
    ]
    assert len(ONTOLOGY_OPTIONS) == len(set(ONTOLOGY_OPTIONS)) == 12
    assert all(
        opt.split(":")[0] == e.curie_prefix
        for opt, e in zip(ONTOLOGY_OPTIONS, ONTOLOGY_REGISTRY, strict=True)
    )


def test_aspects_and_value_kinds_frozen() -> None:
    assert len(ASPECTS) == len(ASPECT_OPTIONS) == 8
    assert all(opt.startswith(a + ":") for a, opt in zip(ASPECTS, ASPECT_OPTIONS, strict=True))
    assert len(VALUE_KINDS) == 4


def test_prefix_of_canonical_case() -> None:
    assert prefix_of("ncbitaxon:8782") == "NCBITaxon"
    assert prefix_of("envo:01000178") == "ENVO"
    assert prefix_of("GO:0007601") == "GO"
    assert prefix_of("nocolon") == ""


def test_masks() -> None:
    assert allowed_for_aspect("unit") == {"uo"}
    assert allowed_for_aspect("other") == {e.id for e in ONTOLOGY_REGISTRY}
    m = mask_for_aspect("taxon", frozenset({"ncbitaxon", "envo"}))
    assert m.sum() == 1 and m[1]
    assert mask_for_aspect("environment").sum() == 2

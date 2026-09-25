"""The ElicitationChooser answers mesa-mcp's term_choice form from its own fixed-key question."""

from __future__ import annotations

import asyncio

from mesa_anyjev.backends.factory import make_backend
from mesa_anyjev.chooser import ElicitationChooser, parse_picker
from mesa_anyjev.config import load_config
from mesa_anyjev.policy import load_policy
from mesa_anyjev.providers.anyjev_provider import AnyJevProvider
from mesa_anyjev.questions import Q_TERM_FITS, Q_TERM_FITS_CHOOSER

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "iri": {
            "type": "string",
            "title": "Ontology term",
            "enum": [
                "http://purl.obolibrary.org/obo/ENVO_01000228",
                "http://purl.obolibrary.org/obo/ENVO_00000446",
            ],
            "enumNames": [
                "tropical moist broadleaf forest biome (ENVO:01000228)",
                "terrestrial biome (ENVO:00000446)",
            ],
        }
    },
    "required": ["iri"],
}
MESSAGE = "Which ENVO term describes 'tropical moist broadleaf forest'?"


def test_parse_picker() -> None:
    ont, value, cands = parse_picker(MESSAGE, SCHEMA)
    assert ont == "envo" and value == "tropical moist broadleaf forest"
    assert cands[0] == {
        "iri": "http://purl.obolibrary.org/obo/ENVO_01000228",
        "label": "tropical moist broadleaf forest biome",
        "curie": "ENVO:01000228",
    }


def test_chooser_answers_or_declines_and_uses_its_own_key() -> None:
    cfg = load_config(env={})
    provider = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    chooser = ElicitationChooser(provider, load_policy())
    content = asyncio.run(chooser(MESSAGE, SCHEMA))
    entry = chooser.log[-1]
    assert entry["level"] == "L0" and len(entry["ranked"]) == 2
    assert (content is None) == (entry["p_true"] < entry["threshold"])
    if content is not None:
        assert content["iri"] in SCHEMA["properties"]["iri"]["enum"]
    assert Q_TERM_FITS_CHOOSER.key != Q_TERM_FITS.key  # B13: never reuses term.fits artifacts
    assert asyncio.run(chooser("Which X term describes 'y'?", {"properties": {}})) is None

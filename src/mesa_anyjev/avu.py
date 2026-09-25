"""Canonical AVU construction and value rules (DESIGN D12; amendment B1).

The AVU shape is mesa-mcp's contract: ``attribute = '<ontology>.<snake_case_label>'``,
``unit = CURIE``. ``mesa_mcp.ols.tools.avu_from_term.handle_avu_from_term`` is ``async``, and
with curie+iri+label supplied it never touches OLS, so ``build_avu`` calls the two sync pieces
it wraps; ``tests/test_avu_parity.py`` asserts byte-for-byte parity with the tool.
"""

from __future__ import annotations

from mesa_mcp.ols.client import _label_to_snake
from mesa_mcp.ols.transform import ontology_annotations_to_avus

from mesa_anyjev.ols import Candidate
from mesa_anyjev.registry import VALUE_KINDS

Avu = dict[str, str]


def build_avu(candidate: Candidate, value: str) -> Avu:
    key = _label_to_snake(candidate.label)
    if not key:
        raise ValueError(f"label {candidate.label!r} does not snake-case to a key")
    avus = ontology_annotations_to_avus(
        candidate.ontology_id, [{"key": key, "value": value, "curie": candidate.curie}]
    )
    if not avus:
        raise ValueError(f"could not build an AVU for {candidate.curie} with value {value!r}")
    avu = avus[0]
    return {
        "attribute": str(avu["attribute"]),
        "value": str(avu["value"]),
        "unit": str(avu["unit"]),
    }


def pre_rule_value_kind(aspect: str, scope: str) -> str | None:
    """Deterministic pre-rules before the value-kind question is asked (D12)."""
    if scope == "site" or (aspect in ("environment", "location") and scope != "column"):
        return "the site code"
    if aspect == "unit":
        return "the term label"
    if scope == "dataset":
        return "the term label"
    return None


def value_for(
    kind: str,
    *,
    term_label: str,
    column_name: str | None,
    site_code: str | None,
    top_value: str | None,
) -> str:
    if kind not in VALUE_KINDS:
        raise ValueError(f"unknown value kind {kind!r}")
    if kind == "the site code" and site_code:
        return site_code
    if kind == "the column name" and column_name:
        return column_name
    if kind == "the most frequent data value" and top_value:
        return top_value
    return term_label


def top_profile_value(profile: str) -> str | None:
    """The most frequent value from a card profile like ``'2 distinct; top: SRER (9416), HARV (6068)'``."""
    marker = "top:"
    if marker not in profile:
        return None
    first = profile.split(marker, 1)[1].split(",", 1)[0].strip()
    return first.rsplit(" (", 1)[0].strip() or None


def triple(avu: Avu) -> tuple[str, str, str]:
    return (avu["attribute"], avu["value"], avu["unit"])

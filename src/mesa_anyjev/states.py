"""JSON-serialisable state builders (plain dicts of str/int/list only: no numpy, no datetime,
so ``observe()`` can store them and the sidecar can hash them).

The shared card header comes first in every state so the rendered prompt prefix is identical
across the questions asked of one card: vLLM prefix caching and the Decider's shared-prefix
scoring both key on it. Key order is fixed and documented because the SQL batch path
(``hosted score``) rebuilds the same dicts with ``struct_pack`` and must produce the same
``state_sha256`` (DESIGN D16).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from mesa_anyjev.cards import ColumnInfo, DatasetCard, SiteInfo

MAX_PROFILE = 300
MAX_DESCRIPTION = 300
MAX_SIBLINGS = 25


def state_sha256(state: dict[str, Any]) -> str:
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def card_header(card: DatasetCard) -> dict[str, Any]:
    return {
        "dataset": card.name,
        "product_title": card.product_title,
        "product_description": card.product_description,
        "sites": [
            {"code": s.code, "name": s.name, "domain": s.domain, "habitat": s.habitat}
            for s in card.sites
        ],
        "rows": card.rows,
        "months": f"{card.months_from} to {card.months_to}",
        "columns": [c.name for c in card.columns],
    }


def _column(col: ColumnInfo) -> dict[str, Any]:
    return {
        "name": col.name,
        "description": col.description,
        "dtype": col.dtype,
        "unit": col.unit,
        "profile": col.profile[:MAX_PROFILE],
    }


def column_state(card: DatasetCard, col: ColumnInfo) -> dict[str, Any]:
    return {"card": card_header(card), "column": _column(col)}


def ontology_state(
    card: DatasetCard,
    col: ColumnInfo,
    aspect: str,
    ontology_id: str,
    option_text: str,
    aspects: list[str],
) -> dict[str, Any]:
    return {
        "card": card_header(card),
        "column": _column(col),
        "aspect": aspect,
        "ontology": {"id": ontology_id, "description": option_text, "aspects": sorted(aspects)},
    }


def candidate_state(
    card: DatasetCard,
    scope: str,
    target: ColumnInfo | SiteInfo | None,
    aspect: str,
    candidate: dict[str, Any],
    n_candidates: int,
) -> dict[str, Any]:
    state: dict[str, Any] = {"card": card_header(card), "scope": scope}
    if isinstance(target, ColumnInfo):
        state["column"] = _column(target)
    elif isinstance(target, SiteInfo):
        state["site"] = {
            "code": target.code,
            "name": target.name,
            "domain": target.domain,
            "habitat": target.habitat,
        }
    state["aspect"] = aspect
    state["candidate"] = {
        "label": str(candidate.get("label", "")),
        "curie": str(candidate.get("curie", "")),
        "ontology_id": str(candidate.get("ontology_id", "")),
        "description": str(candidate.get("description", ""))[:MAX_DESCRIPTION],
        "synonyms": [str(s) for s in list(candidate.get("synonyms") or [])[:5]],
        "has_children": bool(candidate.get("has_children", False)),
    }
    state["n_candidates"] = int(n_candidates)
    return state


def chooser_state(
    ontology_id: str, value: str, candidate: dict[str, Any], n_candidates: int
) -> dict[str, Any]:
    """The state for ``term.fits.chooser``: only what mesa-mcp's picker carries."""
    return {
        "ontology_id": ontology_id,
        "value": value,
        "candidate": {
            "label": str(candidate.get("label", "")),
            "curie": str(candidate.get("curie", "")),
            "description": str(candidate.get("description", ""))[:MAX_DESCRIPTION],
        },
        "n_candidates": int(n_candidates),
    }


def value_kind_state(
    card: DatasetCard, col: ColumnInfo, term: dict[str, Any], aspect: str
) -> dict[str, Any]:
    return {
        "card": card_header(card),
        "column": _column(col),
        "aspect": aspect,
        "term": {"label": str(term.get("label", "")), "curie": str(term.get("curie", ""))},
    }


def avu_state(
    card: DatasetCard,
    avu: dict[str, str],
    term: dict[str, Any],
    scope: str,
    target_name: str | None,
    siblings: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "card": card_header(card),
        "avu": {"attribute": avu["attribute"], "value": avu["value"], "unit": avu["unit"]},
        "term": {
            "label": str(term.get("label", "")),
            "description": str(term.get("description", ""))[:MAX_DESCRIPTION],
        },
        "scope": scope,
        "target": target_name or "",
        "siblings": [
            {"attribute": s["attribute"], "value": s["value"], "unit": s["unit"]}
            for s in siblings[:MAX_SIBLINGS]
        ],
    }

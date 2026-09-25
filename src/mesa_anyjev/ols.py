"""Deterministic candidate generation over mesa-mcp's OLS client.

Everything a model would otherwise do by tool calls is code here: search, prefix filtering
(``search_terms(ontology_id=...)`` also returns imported terms, GO/CL/PR/UBERON under pato),
dropping obsolete or root terms, dedup by CURIE, capping. ``RecordingOLS`` records and replays
responses so tests and the bench never touch the network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol

import requests
from pydantic import BaseModel, ConfigDict, Field

from mesa_anyjev.registry import prefix_of

BIOME_IRI = "http://purl.obolibrary.org/obo/ENVO_00000428"

UNIT_TABLE: dict[str, dict[str, str]] = {
    "meter": {"label": "meter", "curie": "UO:0000008"},
    "centimeter": {"label": "centimeter", "curie": "UO:0000015"},
    "millimeter": {"label": "millimeter", "curie": "UO:0000016"},
    "kilometer": {"label": "kilometer", "curie": "UO:0000009"},
    "gram": {"label": "gram", "curie": "UO:0000021"},
    "kilogram": {"label": "kilogram", "curie": "UO:0000009"},
    "milligram": {"label": "milligram", "curie": "UO:0000022"},
    "second": {"label": "second", "curie": "UO:0000010"},
    "minute": {"label": "minute", "curie": "UO:0000031"},
    "hour": {"label": "hour", "curie": "UO:0000032"},
    "day": {"label": "day", "curie": "UO:0000033"},
    "celsius": {"label": "degree Celsius", "curie": "UO:0000027"},
    "percent": {"label": "percent", "curie": "UO:0000187"},
    "kilometer per hour": {"label": "kilometer per hour", "curie": "UO:0010008"},
    "number": {"label": "count unit", "curie": "UO:0000189"},
    "count": {"label": "count unit", "curie": "UO:0000189"},
    "degree": {"label": "degree", "curie": "UO:0000185"},
    "milliliter": {"label": "milliliter", "curie": "UO:0000098"},
    "liter": {"label": "liter", "curie": "UO:0000099"},
}


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    curie: str
    iri: str
    ontology_id: str
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    has_children: bool = False
    rank: int = 0
    query: str = ""

    def as_state(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "curie": self.curie,
            "ontology_id": self.ontology_id,
            "description": self.description[:300],
            "synonyms": self.synonyms[:5],
            "has_children": self.has_children,
        }


def iri_for(curie: str) -> str:
    prefix, local = curie.split(":", 1)
    return f"http://purl.obolibrary.org/obo/{prefix}_{local}"


class OLSLike(Protocol):
    def search_terms(
        self, query: str, ontology_id: str | None = None, size: int = 15
    ) -> list[dict[str, Any]]: ...
    def search_term_descendants(
        self, query: str, ontology_id: str, parent_iri: str, size: int = 20
    ) -> list[dict[str, Any]]: ...
    def get_term_children(
        self, ontology_id: str, iri: str, size: int = 50
    ) -> list[dict[str, Any]]: ...
    def get_term(self, ontology_id: str, iri: str) -> dict[str, Any] | None: ...


FixtureMode = Literal["off", "record", "replay", "auto"]


class RecordingOLS:
    """Record/replay wrapper: ``<dir>/<sha256(method|args)>.json``. In ``replay`` a miss is an
    error (the fixture set is the contract for hermetic tests); in ``record`` and ``auto`` a
    miss calls the real client and records the response (``auto`` is for engine runs, whose
    decisions differ from the fake backend's and therefore search pairs never seen before)."""

    def __init__(self, inner: OLSLike | None, fixture_dir: str | Path, mode: FixtureMode) -> None:
        self.inner = inner
        self.dir = Path(fixture_dir)
        self.mode = mode
        self.hits = 0
        self.misses = 0
        if mode in ("record", "auto"):
            self.dir.mkdir(parents=True, exist_ok=True)

    def _call(self, method: str, args: dict[str, Any]) -> Any:
        key = hashlib.sha256(
            json.dumps({"method": method, **args}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        path = self.dir / f"{key}.json"
        if self.mode != "off" and path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        if self.mode == "replay":
            raise FileNotFoundError(
                f"no OLS fixture for {method} {args} (record it with fixtures=record)"
            )
        if self.inner is None:
            raise RuntimeError("RecordingOLS has no inner client and no fixture")
        self.misses += 1
        response = getattr(self.inner, method)(**args)
        if self.mode in ("record", "auto"):
            path.write_text(
                json.dumps(
                    {"method": method, "args": args, "response": response},
                    indent=1,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return response

    def search_terms(
        self, query: str, ontology_id: str | None = None, size: int = 15
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self._call(
            "search_terms", {"query": query, "ontology_id": ontology_id, "size": size}
        )
        return out

    def search_term_descendants(
        self, query: str, ontology_id: str, parent_iri: str, size: int = 20
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self._call(
            "search_term_descendants",
            {"query": query, "ontology_id": ontology_id, "parent_iri": parent_iri, "size": size},
        )
        return out

    def get_term_children(self, ontology_id: str, iri: str, size: int = 50) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self._call(
            "get_term_children", {"ontology_id": ontology_id, "iri": iri, "size": size}
        )
        return out

    def get_term(self, ontology_id: str, iri: str) -> dict[str, Any] | None:
        out: dict[str, Any] | None = self._call(
            "get_term", {"ontology_id": ontology_id, "iri": iri}
        )
        return out


def to_candidates(hits: list[dict[str, Any]], ontology_id: str, query: str) -> list[Candidate]:
    """Prefix filter (the CURIE must belong to the ontology searched), drop obsolete, root and
    incomplete hits, dedup by CURIE keeping the first rank."""
    want = ontology_id.lower()
    out: list[Candidate] = []
    seen: set[str] = set()
    for rank, hit in enumerate(hits):
        curie = str(hit.get("curie") or "")
        iri = str(hit.get("iri") or "")
        label = str(hit.get("label") or "")
        if not curie or not iri or not label:
            continue
        if prefix_of(curie).lower() != want:
            continue
        if hit.get("isRoot") or label.lower().startswith("obsolete"):
            continue
        if curie in seen:
            continue
        seen.add(curie)
        out.append(
            Candidate(
                label=label,
                curie=curie,
                iri=iri,
                ontology_id=want,
                description=str(hit.get("description") or ""),
                synonyms=[str(s) for s in (hit.get("synonyms") or [])][:5],
                has_children=bool(hit.get("hasChildren", False)),
                rank=rank,
                query=query,
            )
        )
    return out


class OLSLayer:
    def __init__(self, client: OLSLike, *, max_candidates: int = 12, search_size: int = 20) -> None:
        self.client = client
        self.max_candidates = max_candidates
        self.search_size = search_size
        self.calls = 0

    def search_candidates(
        self, queries: list[str], ontology_id: str
    ) -> tuple[list[Candidate], dict[str, Any]]:
        """Candidates for an ontology over up to three queries, with the search log for the
        decision group (queries, raw hit counts, CURIEs filtered out)."""
        merged: dict[str, Candidate] = {}
        log: dict[str, Any] = {
            "ontology_id": ontology_id,
            "queries": [],
            "raw_hits": {},
            "filtered_out": [],
        }
        for query in [q for q in queries if q][:3]:
            self.calls += 1
            try:
                hits = self.client.search_terms(
                    query=query, ontology_id=ontology_id, size=self.search_size
                )
            except (requests.RequestException, RuntimeError) as exc:
                log.setdefault("errors", []).append(f"{query}: {exc}")
                continue
            log["queries"].append(query)
            log["raw_hits"][query] = len(hits)
            kept = to_candidates(hits, ontology_id, query)
            log["filtered_out"].extend(
                sorted({str(h.get("curie")) for h in hits} - {c.curie for c in kept})
            )
            for c in kept:
                merged.setdefault(c.curie, c)
        cands = sorted(merged.values(), key=lambda c: (c.rank, c.curie))[: self.max_candidates]
        log["n_candidates"] = len(cands)
        return cands, log

    def biome_candidates(self, queries: list[str]) -> tuple[list[Candidate], dict[str, Any]]:
        """The closed ENVO biome set under ENVO:00000428 for a site's habitat queries; parents
        are kept so specificity is the model's decision."""
        merged: dict[str, Candidate] = {}
        log: dict[str, Any] = {
            "ontology_id": "envo",
            "parent": BIOME_IRI,
            "queries": [],
            "raw_hits": {},
            "filtered_out": [],
        }
        for query in [q for q in queries if q][:3]:
            self.calls += 1
            try:
                hits = self.client.search_term_descendants(
                    query=query, ontology_id="envo", parent_iri=BIOME_IRI, size=self.search_size
                )
            except (
                requests.RequestException,
                RuntimeError,
            ) as exc:  # raises HTTPError, not OLSAPIError
                log.setdefault("errors", []).append(f"{query}: {exc}")
                continue
            log["queries"].append(query)
            log["raw_hits"][query] = len(hits)
            for c in to_candidates(hits, "envo", query):
                merged.setdefault(c.curie, c)
        cands = sorted(merged.values(), key=lambda c: (c.rank, c.curie))[: self.max_candidates]
        log["n_candidates"] = len(cands)
        return cands, log

    def children(self, ontology_id: str, iri: str, size: int = 10) -> list[Candidate]:
        self.calls += 1
        hits = self.client.get_term_children(ontology_id=ontology_id, iri=iri, size=size)
        return to_candidates(hits, ontology_id, f"children:{iri}")

    def unit_candidate(self, unit_str: str) -> Candidate | None:
        key = unit_str.strip().lower()
        for name, spec in UNIT_TABLE.items():
            if key == name or key.endswith(name) or (name == "celsius" and "celsius" in key):
                return Candidate(
                    label=spec["label"],
                    curie=spec["curie"],
                    iri=iri_for(spec["curie"]),
                    ontology_id="uo",
                    query=unit_str,
                )
        return None

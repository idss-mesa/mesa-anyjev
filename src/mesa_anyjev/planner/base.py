from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mesa_anyjev.cards import DatasetCard
from mesa_anyjev.registry import ASPECTS, ONTOLOGY_REGISTRY

_ONTOLOGY_IDS = {e.id for e in ONTOLOGY_REGISTRY}
MAX_QUERIES = 3


class ColumnHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotate: bool | None = None
    aspect: str | None = None
    ontology: str | None = None
    queries: list[str] = Field(default_factory=list)

    @field_validator("aspect")
    @classmethod
    def _aspect(cls, v: str | None) -> str | None:
        return v if v is None or v in ASPECTS else None

    @field_validator("ontology")
    @classmethod
    def _ontology(cls, v: str | None) -> str | None:
        return v.lower() if v and v.lower() in _ONTOLOGY_IDS else None

    @field_validator("queries")
    @classmethod
    def _queries(cls, v: list[str]) -> list[str]:
        return [q.strip() for q in v if q and q.strip()][:MAX_QUERIES]


class SiteHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment_queries: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """What the reasoning model proposes; closed enums keep it a plan, not an answer."""

    model_config = ConfigDict(extra="forbid")

    ontologies: list[str] = Field(default_factory=list)
    columns: dict[str, ColumnHint] = Field(default_factory=dict)
    sites: dict[str, SiteHint] = Field(default_factory=dict)
    taxon_queries: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("ontologies")
    @classmethod
    def _ontologies(cls, v: list[str]) -> list[str]:
        seen: list[str] = []
        for o in v:
            o = o.lower()
            if o in _ONTOLOGY_IDS and o not in seen:
                seen.append(o)
        return seen


class PlanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: Plan
    planner: str
    model: str | None = None
    prompt_sha256: str | None = None
    raw_text: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    fallback: bool = False


class Planner(Protocol):
    name: str
    model: str | None

    def plan(self, card: DatasetCard) -> PlanResult: ...

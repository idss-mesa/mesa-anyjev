"""Pydantic rows mirroring the sidecar DDL (``migrations/0001_mesa_anyjev.sql``)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

RunStatus = Literal["decided", "applied", "partial", "failed", "dry_run"]
WriteStatus = Literal["proposed", "rejected", "accepted", "written", "mirror_failed", "dry_run"]
LabelSource = Literal[
    "curator",
    "curator_implicit",
    "accepted_avu",
    "consensus_all",
    "consensus_majority",
    "consensus_negative",
    "teacher",
    "hosted_jev",
    "gold",
]
OverrideAction = Literal["pick", "accept", "reject", "decline", "restore"]
Outcome = Literal[
    "auto", "proposed", "human", "escalated", "abstain", "rejected", "rule", "decider_unavailable"
]


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _non_blank(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("must be a non-empty string (provenance is mandatory)")
    return value


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunRow(_Row):
    run_id: UUID = Field(default_factory=uuid4)
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    status: RunStatus = "decided"
    actor: str
    irods_path: str | None = None
    project_id: UUID | None = None
    card_name: str
    card_sha256: str
    planner: str
    planner_model: str | None = None
    planner_prompt_sha256: str | None = None
    plan_json: dict[str, Any] = Field(default_factory=dict)
    planner_fallback: bool = False
    backend_kind: str
    canonical_model: str | None = None
    served_model: str | None = None
    served_revision: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    anyjev_commit: str
    mesa_anyjev_version: str
    questions_lock_sha: str
    artifacts_sha256: str | None = None
    policy_profile: str
    config_sha256: str
    write_mode: Literal["local", "irods", "none"] = "none"
    hosted_provider_used: bool = False
    data_left_host: bool = False
    egress_region: str | None = None
    missing_labels: int = 0
    n_prompts: int | None = None
    seconds: float | None = None

    _actor = field_validator("actor")(_non_blank)


class DecisionRow(_Row):
    decision_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    seq: int
    group_id: UUID | None = None
    parent_decision_id: UUID | None = None
    question_id: str
    question_key: str
    kind: Literal["choice", "noul", "score"]
    k: int
    scope: Literal["dataset", "column", "site", "avu"]
    column_name: str | None = None
    site_code: str | None = None
    state_sha256: str
    state_json: dict[str, Any]
    prompt_sha256: str | None = None
    provider: str
    method: str
    model: str
    served_model: str | None = None
    level: Literal["raw", "L0", "L1", "L2", "none"]
    calibration: Literal["anyjev", "typesafe", "none"]
    probs: list[float] | None = None
    answer_index: int
    answer: str
    confidence: float | None = None
    p_true: float | None = None
    margin: float | None = None
    score_value: float | None = None
    answer_mass: float | None = None
    prior_method: str | None = None
    order_flip_l0: float | None = None
    permutations: int | None = None
    temperature: float | None = None
    n_calib: int | None = None
    routed_from: str | None = None
    adapted: bool | None = None
    blocks_executed: int | None = None
    masked_mass: float | None = None
    threshold_auto: float | None = None
    threshold_propose: float | None = None
    outcome: Outcome
    ts: datetime = Field(default_factory=_now)

    _provider = field_validator("provider", "method", "model")(_non_blank)

    @field_validator("probs")
    @classmethod
    def _probs_shape(cls, value: list[float] | None) -> list[float] | None:
        return value


class DecisionOptionRow(_Row):
    decision_id: UUID
    option_index: int
    option_text: str
    curie: str | None = None
    iri: str | None = None
    ontology_id: str | None = None
    prob: float | None = None
    rank: int | None = None


class DecisionGroupRow(_Row):
    group_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    question_id: str
    scope: Literal["dataset", "column", "site", "avu"]
    column_name: str | None = None
    site_code: str | None = None
    aspect: str | None = None
    ontology_id: str | None = None
    search_json: dict[str, Any] = Field(default_factory=dict)
    n_candidates: int = 0
    winner_decision_id: UUID | None = None
    top_p: float | None = None
    group_margin: float | None = None
    level: str | None = None
    outcome: Outcome
    missing_labels: int = 0
    escalated_from: UUID | None = None
    ts: datetime = Field(default_factory=_now)


class AvuLinkRow(_Row):
    link_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    group_id: UUID | None = None
    decision_id: UUID | None = None
    project_id: UUID | None = None
    snapshot_id: int | None = None
    irods_path: str | None = None
    target_type: str = "data_object"
    attribute: str
    value: str
    unit: str = ""
    op: Literal["add", "delete"] = "add"
    term_curie: str | None = None
    term_iri: str | None = None
    term_label: str | None = None
    ontology_id: str | None = None
    aspect: str | None = None
    column_name: str | None = None
    site_code: str | None = None
    value_kind: str | None = None
    source: str | None = None
    write_status: WriteStatus = "proposed"
    duplicate_of: UUID | None = None
    written_at: datetime | None = None

    _attr = field_validator("attribute", "value")(_non_blank)


class HumanOverrideRow(_Row):
    override_id: UUID = Field(default_factory=uuid4)
    group_id: UUID | None = None
    decision_id: UUID | None = None
    link_id: UUID | None = None
    actor: str
    action: OverrideAction
    chosen_index: int | None = None
    chosen_curie: str | None = None
    elicitation_key: str | None = None
    offered: list[dict[str, Any]] = Field(default_factory=list)
    ts: datetime = Field(default_factory=_now)

    _actor = field_validator("actor")(_non_blank)


class LabelRow(_Row):
    label_id: UUID = Field(default_factory=uuid4)
    question_id: str
    question_key: str
    state_sha256: str
    state_json: dict[str, Any]
    label_index: int
    label_source: LabelSource
    weight: float
    source_ref: str | None = None
    card: str | None = None
    decision_id: UUID | None = None
    batch_id: str | None = None
    consumed_in_bundle: str | None = None
    ts: datetime = Field(default_factory=_now)

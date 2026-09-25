"""The provider contract above AnyJev (DESIGN D3).

``DecisionRecord`` is what every provider returns and the sidecar stores. Two fields say how
much to trust the numbers: ``level`` is AnyJev's level (``raw``/``L0``/``L1``/``L2``) or
``none`` for everything that is not an AnyJev decision (a rule, a planner hint, a Claude
structured answer, a hosted Jev answer); ``calibration`` says where the probabilities come
from (``anyjev``, ``typesafe`` for hosted Jev, ``none``). Invariant: ``probs`` is ``None`` iff
``calibration`` is ``none``. Nothing is ever one-hot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from anyjev import Question
from pydantic import BaseModel, ConfigDict, Field, model_validator

Level = Literal["raw", "L0", "L1", "L2", "none"]
Calibration = Literal["anyjev", "typesafe", "none"]
Method = Literal[
    "anyjev",
    "fake",
    "rule",
    "planner",
    "claude:structured_output",
    "hosted:prompt_jev",
    "unavailable",
]

LEVEL_RANK: dict[str, int] = {"none": -1, "raw": 0, "L0": 1, "L1": 2, "L2": 3}

Scalar = float | int | str | bool | None


class DecisionRecord(BaseModel):
    """One answered question, ready for the sidecar."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_key: str
    kind: Literal["choice", "noul", "score"]
    options: list[str]
    state: dict[str, Any]
    state_sha256: str
    prompt_sha256: str | None = None
    provider: str
    method: Method
    model: str
    served_model: str | None = None
    level: Level
    calibration: Calibration
    probs: list[float] | None = None
    answer_index: int  # -1 = abstain / no answer
    answer: str
    confidence: float | None = None
    p_true: float | None = None  # noul only: probs[0]
    margin: float | None = None  # top1 - top2 within this decision
    score_value: float | None = None  # score only
    diagnostics: dict[str, Scalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _honest(self) -> DecisionRecord:
        if (self.probs is None) != (self.calibration == "none"):
            raise ValueError("probs must be None exactly when calibration is 'none'")
        if self.calibration == "anyjev" and self.level == "none":
            raise ValueError("an AnyJev-calibrated record must carry an AnyJev level")
        if self.calibration != "anyjev" and self.level != "none":
            raise ValueError("only AnyJev decisions carry an AnyJev level")
        if self.probs is not None:
            if len(self.probs) != len(self.options):
                raise ValueError("probs must have one entry per option")
            if any(p < 0 for p in self.probs) or abs(sum(self.probs) - 1.0) > 1e-3:
                raise ValueError("probs must be a distribution over the options")
            if self.kind == "noul" and self.p_true is None:
                raise ValueError("a noul record with probs must carry p_true")
        if self.answer_index >= len(self.options):
            raise ValueError("answer_index out of range")
        return self


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can honestly read (DESIGN D11)."""

    max_choice_k: int
    hidden_states: bool
    logprob_top_k: int | None


class LevelUnavailable(RuntimeError):
    """The requested AnyJev level cannot be served honestly by this provider."""


class CapabilityError(ValueError):
    """The question shape exceeds what the backend can read (ask the noul twin instead)."""


class DecisionProvider(Protocol):
    name: str
    model: str
    capabilities: BackendCapabilities

    def decide_batch(
        self, states: list[dict[str, Any]], question: Question, *, level: str | None = None
    ) -> list[DecisionRecord]: ...

    def supports_level(self, question: Question, level: str) -> bool: ...


def meets_level(record: DecisionRecord, required: Level) -> bool:
    return LEVEL_RANK[record.level] >= LEVEL_RANK[required]


def rank_probs(probs: list[float]) -> tuple[int, float, float]:
    """(argmax, confidence, margin) for a distribution."""
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top = probs[order[0]]
    second = probs[order[1]] if len(order) > 1 else 0.0
    return order[0], top, top - second

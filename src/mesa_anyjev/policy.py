"""Write policy: thresholds and outcomes in one place (DESIGN D6, D11; amendments A2, A4, A7).

``policy_defaults.yaml`` holds per-question thresholds. A numeric ``auto`` threshold must cite
the leave-one-card-out bench cell it came from; ``tests/test_policy_citations.py`` enforces the
citation and the negatives count. ``outcome()`` never turns a record with calibration ``none``
into ``auto``, and a batch in which any label was missing from the gateway's top-20 is capped
at ``rule``/``abstain``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from mesa_anyjev.providers.base import LEVEL_RANK, DecisionRecord, Level, rank_probs

Outcome = Literal[
    "auto", "proposed", "human", "escalated", "abstain", "rejected", "rule", "decider_unavailable"
]
Stat = Literal["p_true", "confidence"]

DEFAULTS_PATH = Path(__file__).resolve().parents[2] / "policy_defaults.yaml"


@dataclass(frozen=True)
class Thresholds:
    stat: Stat
    auto: float | None
    propose: float
    margin: float
    min_level: Level
    min_weight: float
    cite: str


@dataclass(frozen=True)
class Profile:
    name: str
    min_level_write: Level
    auto_calibrations: frozenset[str]
    allow_hosted_write: bool


@dataclass(frozen=True)
class Policy:
    questions: dict[str, Thresholds]
    profiles: dict[str, Profile]

    def thresholds(self, question_id: str) -> Thresholds:
        return self.questions[question_id]

    def profile(self, name: str) -> Profile:
        return self.profiles[name]


def load_policy(path: str | Path = DEFAULTS_PATH) -> Policy:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    questions = {
        qid: Thresholds(
            stat=t["stat"],
            auto=None if t.get("auto") is None else float(t["auto"]),
            propose=float(t["propose"]),
            margin=float(t.get("margin", 0.0)),
            min_level=t.get("min_level", "L1"),
            min_weight=float(t.get("min_weight", 0.6)),
            cite=str(t.get("cite", "")),
        )
        for qid, t in (raw.get("questions") or {}).items()
    }
    profiles = {
        name: Profile(
            name=name,
            min_level_write=p.get("min_level_write", "L1"),
            auto_calibrations=frozenset(p.get("auto_calibrations", ["anyjev"])),
            allow_hosted_write=bool(p.get("allow_hosted_write", False)),
        )
        for name, p in (raw.get("profiles") or {}).items()
    }
    return Policy(questions=questions, profiles=profiles)


def statistic(record: DecisionRecord, stat: Stat) -> float | None:
    return record.p_true if stat == "p_true" else record.confidence


def outcome(
    record: DecisionRecord,
    thresholds: Thresholds,
    profile: Profile,
    *,
    missing_labels: int = 0,
) -> Outcome:
    """The outcome of one decision under the thresholds and the profile.

    ``auto`` needs: an AnyJev level at or above both floors, a calibration the profile allows,
    a cited numeric threshold, the statistic at or above it and the margin at or above the
    margin floor, and no missing label in the batch (D11). ``proposed`` needs the statistic at
    or above ``propose``. Everything else abstains.
    """
    if record.answer_index < 0 or record.probs is None:
        return "abstain"
    if missing_labels > 0:
        return "abstain"
    value = statistic(record, thresholds.stat)
    if value is None:
        return "abstain"
    margin_ok = record.margin is None or record.margin >= thresholds.margin
    level_ok = LEVEL_RANK[record.level] >= max(
        LEVEL_RANK[thresholds.min_level], LEVEL_RANK[profile.min_level_write]
    )
    calibration_ok = (
        record.calibration in profile.auto_calibrations and record.calibration != "none"
    )
    if (
        thresholds.auto is not None
        and level_ok
        and calibration_ok
        and value >= thresholds.auto
        and margin_ok
    ):
        return "auto"
    if value >= thresholds.propose:
        return "proposed"
    return "abstain"


def masked(record: DecisionRecord, mask: np.ndarray) -> DecisionRecord:
    """Zero the probability of options that are out of play, renormalise, and log the mass
    that was masked away (AnyJev's 2048 demo pattern). A fully masked distribution abstains."""
    if record.probs is None:
        return record
    probs = np.asarray(record.probs, dtype=float)
    keep = np.asarray(mask, dtype=bool)
    if keep.shape != probs.shape:
        raise ValueError("mask must have one entry per option")
    kept = probs * keep
    total = float(kept.sum())
    diagnostics: dict[str, Any] = dict(record.diagnostics)
    diagnostics["masked_mass"] = round(1.0 - total, 6)
    if total <= 0.0:
        diagnostics["mask_empty"] = True
        return record.model_copy(
            update={"answer_index": -1, "answer": "", "diagnostics": diagnostics}
        )
    new = (kept / total).tolist()
    idx, conf, margin = rank_probs(new)
    return record.model_copy(
        update={
            "probs": new,
            "answer_index": idx,
            "answer": record.options[idx],
            "confidence": conf,
            "margin": margin,
            "diagnostics": diagnostics,
        }
    )

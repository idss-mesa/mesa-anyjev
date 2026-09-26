"""DataCite questions (M5): classify free text into the frozen DataCite vocabularies with the
same providers and levels as every other question, and shape the answers for
``mesa_avu_apply_datacite`` (mesa-mcp). ResourceTypeGeneral has 28 members, above the
26-option cap, so it is asked as one yes/no per member (the Minesweeper pattern); the other
vocabularies are asked as a choice when the backend can read that many options and as the
yes/no twin otherwise."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from anyjev import Question

from mesa_anyjev.cards import DatasetCard
from mesa_anyjev.policy import Policy, Profile, outcome
from mesa_anyjev.providers.base import CapabilityError, DecisionRecord
from mesa_anyjev.questions import QUESTIONS
from mesa_anyjev.registry import DATACITE_VOCABULARIES
from mesa_anyjev.states import datacite_state

CHOICE_QUESTION: dict[str, str] = {
    "ContributorType": "datacite.contributor_type",
    "RelationType": "datacite.relation_type",
    "DateType": "datacite.date_type",
    "DescriptionType": "datacite.description_type",
}
TWIN_QUESTION: dict[str, str] = {
    "ResourceTypeGeneral": "datacite.resource_type_fits",
    "ContributorType": "datacite.contributor_type_fits",
    "RelationType": "datacite.relation_type_fits",
    "DateType": "datacite.date_type_fits",
}


@dataclass
class Classification:
    vocabulary: str
    text: str
    value: str | None
    p: float | None
    level: str
    outcome: str
    question_id: str
    records: list[DecisionRecord] = field(default_factory=list)
    ranked: list[tuple[str, float | None]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "vocabulary": self.vocabulary,
            "value": self.value,
            "p": self.p,
            "level": self.level,
            "outcome": self.outcome,
            "question_id": self.question_id,
            "ranked": self.ranked[:5],
        }


def classify(
    provider: Any,
    vocabulary: str,
    text: str,
    *,
    policy: Policy,
    profile: Profile,
    card: DatasetCard | None = None,
) -> Classification:
    """One DataCite value for ``text``: a choice when the backend can read the whole list,
    else one yes/no per member; the outcome follows the question's thresholds."""
    if vocabulary not in DATACITE_VOCABULARIES:
        raise ValueError(f"unknown DataCite vocabulary {vocabulary!r}")
    members = DATACITE_VOCABULARIES[vocabulary]
    qid = CHOICE_QUESTION.get(vocabulary)
    if qid is not None and len(members) <= provider.capabilities.max_choice_k:
        q: Question = QUESTIONS[qid].question
        recs = provider.decide_batch([datacite_state(card, vocabulary, text)], q)
        rec = recs[0]
        out = outcome(
            rec, policy.thresholds(qid), profile, missing_labels=provider.last_missing_labels
        )
        ranked = sorted(
            zip(members, rec.probs or [None] * len(members), strict=True),
            key=lambda x: -(x[1] or 0.0),
        )
        return Classification(
            vocabulary,
            text,
            rec.answer or None if rec.answer_index >= 0 else None,
            rec.confidence,
            rec.level,
            out,
            qid,
            recs,
            ranked,
        )
    twin = TWIN_QUESTION.get(vocabulary)
    if twin is None:
        raise CapabilityError(
            f"{vocabulary}: {len(members)} options exceed "
            f"max_choice_k={provider.capabilities.max_choice_k} and no twin exists"
        )
    q = QUESTIONS[twin].question
    states = [datacite_state(card, vocabulary, text, value=m) for m in members]
    recs = provider.decide_batch(states, q)
    pairs = sorted(zip(members, recs, strict=True), key=lambda x: -(x[1].p_true or 0.0))
    best_value, best = pairs[0]
    out = outcome(
        best, policy.thresholds(twin), profile, missing_labels=provider.last_missing_labels
    )
    return Classification(
        vocabulary,
        text,
        best_value if best.answer_index >= 0 else None,
        best.p_true,
        best.level,
        out,
        twin,
        recs,
        [(m, r.p_true) for m, r in pairs],
    )


def card_record(
    provider: Any, card: DatasetCard, *, policy: Policy, profile: Profile
) -> dict[str, Any]:
    """The DataCite fragments a dataset card can answer on its own: the general resource type
    of the product and the description type of its product description. Returned in the
    ``record`` shape ``mesa_avu_apply_datacite`` takes, plus the classifications."""
    text = f"{card.product_title}. {card.product_description}"
    rtype = classify(
        provider, "ResourceTypeGeneral", text, policy=policy, profile=profile, card=card
    )
    dtype = classify(
        provider,
        "DescriptionType",
        card.product_description,
        policy=policy,
        profile=profile,
        card=card,
    )
    record: dict[str, Any] = {"titles": [{"title": card.product_title}]}
    if rtype.value and rtype.outcome in ("auto", "proposed"):
        record["types"] = {"resourceTypeGeneral": rtype.value}
    if dtype.value and dtype.outcome in ("auto", "proposed"):
        record["descriptions"] = [
            {"description": card.product_description, "descriptionType": dtype.value}
        ]
    return {"record": record, "classifications": [rtype.as_dict(), dtype.as_dict()]}

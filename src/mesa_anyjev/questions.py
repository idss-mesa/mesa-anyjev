"""The production questions: the only place ``Question`` objects are built (DESIGN D1).

Every question's wording and option list are frozen here and pinned in ``questions.lock.json``.
The variable content of a decision (the column, the site, the candidate term, the sibling
AVUs) lives in the *state*, so one key accumulates a batch prior, an L1 temperature and an L2
head across every column, site and dataset.

Two shapes (DESIGN D1): fixed registries as ``choice`` (inapplicable options masked after the
decision) and open candidate sets as one ``noul`` per candidate ranked by ``p_true``. Every
``choice`` with more options than the gateway can read (``max_choice_k``) has a ``noul``
*twin* under its own key; the pipeline picks the form from the backend's capabilities (D11).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from anyjev import Question

from mesa_anyjev.registry import (
    ASPECT_OPTIONS,
    DATACITE_CONTRIBUTOR_TYPES,
    DATACITE_DATE_TYPES,
    DATACITE_DESCRIPTION_TYPES,
    DATACITE_RELATION_TYPES,
    ONTOLOGY_OPTIONS,
    VALUE_KINDS,
)

Scope = Literal["dataset", "column", "site", "avu"]

Q_COLUMN_ANNOTATE: Final = Question.noul(
    "Should this column receive an ontology-grounded annotation? Answer No for record "
    "identifiers, internal codes, timestamps and bookkeeping fields.",
    name="column.annotate",
)

Q_COLUMN_ASPECT: Final = Question.choice(
    "Which aspect of the dataset does this column describe?",
    ASPECT_OPTIONS,
    name="column.aspect",
)

Q_COLUMN_ONTOLOGY: Final = Question.choice(
    "Which ontology should be searched for a term that annotates this column?",
    ONTOLOGY_OPTIONS,
    name="column.ontology",
)

# The noul twin of Q_COLUMN_ONTOLOGY: one state per registry entry (K=12 exceeds the gateway's
# readable option count, DESIGN D11).
Q_COLUMN_ONTOLOGY_FITS: Final = Question.noul(
    "Does the ontology described in the state contain the right kind of term for annotating "
    "this column?",
    name="column.ontology_fits",
)

Q_TERM_FITS: Final = Question.noul(
    "Is the candidate ontology term a correct annotation for the described column, site or "
    "dataset: the right concept at an appropriate specificity, not merely related?",
    name="term.fits",
)

# The chooser answers mesa-mcp's own term picker, which carries only (ontology, value,
# candidates): a different state shape, hence its own key (plan amendment B13).
Q_TERM_FITS_CHOOSER: Final = Question.noul(
    "Is the candidate ontology term the correct term for the value the user wants to "
    "annotate with: the right concept at an appropriate specificity, not merely related?",
    name="term.fits.chooser",
)

Q_VALUE_KIND: Final = Question.choice(
    "What should the AVU value be?",
    VALUE_KINDS,
    name="avu.value_kind",
)

Q_KEEP_AVU: Final = Question.noul(
    "Given the annotations already selected for this dataset, does this annotation add "
    "correct, non-redundant information worth keeping?",
    name="avu.keep",
)


# -- M5: the planner audit and the DataCite vocabularies ------------------------------------------
Q_DATASET_ONTOLOGY_APPLIES: Final = Question.noul(
    "Does this ontology apply to the dataset as a whole: would at least one of its terms "
    "correctly describe what the dataset measures, observes, samples or records?",
    name="dataset.ontology_applies",
)

Q_DATACITE_RESOURCE_TYPE_FITS: Final = Question.noul(
    "Is the DataCite ResourceTypeGeneral value the right general type for this resource?",
    name="datacite.resource_type_fits",
)
Q_DATACITE_CONTRIBUTOR_TYPE: Final = Question.choice(
    "Which DataCite ContributorType best describes this contributor's role?",
    DATACITE_CONTRIBUTOR_TYPES,
    name="datacite.contributor_type",
)
Q_DATACITE_CONTRIBUTOR_TYPE_FITS: Final = Question.noul(
    "Is the DataCite ContributorType value the right role for this contributor?",
    name="datacite.contributor_type_fits",
)
Q_DATACITE_RELATION_TYPE: Final = Question.choice(
    "Which DataCite RelationType describes how this resource relates to the related identifier?",
    DATACITE_RELATION_TYPES,
    name="datacite.relation_type",
)
Q_DATACITE_RELATION_TYPE_FITS: Final = Question.noul(
    "Is the DataCite RelationType value the right relation from this resource to the related "
    "identifier?",
    name="datacite.relation_type_fits",
)
Q_DATACITE_DATE_TYPE: Final = Question.choice(
    "Which DataCite DateType describes this date?",
    DATACITE_DATE_TYPES,
    name="datacite.date_type",
)
Q_DATACITE_DATE_TYPE_FITS: Final = Question.noul(
    "Is the DataCite DateType value the right type for this date?",
    name="datacite.date_type_fits",
)
Q_DATACITE_DESCRIPTION_TYPE: Final = Question.choice(
    "Which DataCite DescriptionType describes this description text?",
    DATACITE_DESCRIPTION_TYPES,
    name="datacite.description_type",
)


@dataclass(frozen=True)
class QuestionSpec:
    question: Question
    scope: Scope
    twin_id: str | None = None  # the noul twin asked when a choice exceeds max_choice_k


QUESTIONS: Final[dict[str, QuestionSpec]] = {
    "column.annotate": QuestionSpec(Q_COLUMN_ANNOTATE, "column"),
    "column.aspect": QuestionSpec(Q_COLUMN_ASPECT, "column"),
    "column.ontology": QuestionSpec(Q_COLUMN_ONTOLOGY, "column", twin_id="column.ontology_fits"),
    "column.ontology_fits": QuestionSpec(Q_COLUMN_ONTOLOGY_FITS, "column"),
    "term.fits": QuestionSpec(Q_TERM_FITS, "column"),
    "term.fits.chooser": QuestionSpec(Q_TERM_FITS_CHOOSER, "column"),
    "avu.value_kind": QuestionSpec(Q_VALUE_KIND, "avu"),
    "avu.keep": QuestionSpec(Q_KEEP_AVU, "avu"),
    "dataset.ontology_applies": QuestionSpec(Q_DATASET_ONTOLOGY_APPLIES, "dataset"),
    "datacite.resource_type_fits": QuestionSpec(Q_DATACITE_RESOURCE_TYPE_FITS, "dataset"),
    "datacite.contributor_type": QuestionSpec(
        Q_DATACITE_CONTRIBUTOR_TYPE, "dataset", twin_id="datacite.contributor_type_fits"
    ),
    "datacite.contributor_type_fits": QuestionSpec(Q_DATACITE_CONTRIBUTOR_TYPE_FITS, "dataset"),
    "datacite.relation_type": QuestionSpec(
        Q_DATACITE_RELATION_TYPE, "dataset", twin_id="datacite.relation_type_fits"
    ),
    "datacite.relation_type_fits": QuestionSpec(Q_DATACITE_RELATION_TYPE_FITS, "dataset"),
    "datacite.date_type": QuestionSpec(
        Q_DATACITE_DATE_TYPE, "dataset", twin_id="datacite.date_type_fits"
    ),
    "datacite.date_type_fits": QuestionSpec(Q_DATACITE_DATE_TYPE_FITS, "dataset"),
    "datacite.description_type": QuestionSpec(Q_DATACITE_DESCRIPTION_TYPE, "dataset"),
}

LOCK_PATH: Final = Path(__file__).resolve().parents[2] / "questions.lock.json"


def question_by_id(question_id: str) -> Question:
    return QUESTIONS[question_id].question


def twin_for(question_id: str) -> Question | None:
    twin = QUESTIONS[question_id].twin_id
    return QUESTIONS[twin].question if twin else None


def lock_payload() -> dict[str, Any]:
    """What ``questions.lock.json`` pins: kind, text, options, key and twin per question."""
    entries = {
        qid: {
            "kind": spec.question.kind,
            "text": spec.question.text,
            "options": list(spec.question.options),
            "key": spec.question.key,
            "twin": spec.twin_id,
        }
        for qid, spec in QUESTIONS.items()
    }
    from mesa_anyjev.providers.motherduck_provider import sql_for

    # The hosted SQL literal per question is pinned beside the key (M4b) but does not enter
    # the sha: rotating artifacts when only the hosted rendering changes would be wrong.
    hosted_sql = {qid: sql_for(spec.question) for qid, spec in QUESTIONS.items()}
    return {"questions": entries, "hosted_sql": hosted_sql, "lock_sha": _sha(entries)}


def _sha(entries: dict[str, Any]) -> str:
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def lock_sha() -> str:
    return str(lock_payload()["lock_sha"])


def read_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def write_lock(path: Path = LOCK_PATH) -> str:
    payload = lock_payload()
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(payload["lock_sha"])


def lock_drift(path: Path = LOCK_PATH) -> list[str]:
    """Human-readable differences between the code and the lock file (empty means in sync)."""
    current = lock_payload()["questions"]
    try:
        locked = read_lock(path)["questions"]
    except FileNotFoundError:
        return ["questions.lock.json is missing; run `mesa-anyjev questions --update-lock`"]
    problems: list[str] = []
    for qid in sorted(set(current) | set(locked)):
        if qid not in locked:
            problems.append(f"{qid}: new question, not in the lock")
        elif qid not in current:
            problems.append(f"{qid}: in the lock but no longer defined")
        elif current[qid]["key"] != locked[qid]["key"]:
            problems.append(
                f"{qid}: key {locked[qid]['key']} -> {current[qid]['key']} "
                "(wording or options changed; labels and artifacts need migration, DESIGN D7)"
            )
        elif current[qid]["twin"] != locked[qid]["twin"]:
            problems.append(f"{qid}: twin changed")
    return problems

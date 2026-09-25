"""Label ingestion into ``mesa_anyjev.labels`` (DESIGN D6; amendments A2, A3).

Every label row carries the exact state the question is asked on, so a fit reads
``(state, label_index, weight)`` triples per question key. Sources and weights, highest wins
when one state has several: ``curator`` 1.0 (an explicit pick or "none of these"),
``curator_implicit`` 0.7 (the other offered candidates of a pick), ``accepted_avu`` 0.8 (an
auto-written AVU later deleted through mesa-mcp), ``consensus_all`` 0.8 and
``consensus_majority`` 0.6 (neon-avu-eval agreement), ``consensus_negative`` 0.5,
``teacher`` / ``hosted_jev`` 0.5, ``gold`` 1.0.

The neon-avu-eval silver is *agreement between four agentic models*, not truth, and it is
circular with the agentic baselines the bench compares against; the docs say so wherever a
number derived from it appears.
"""

from __future__ import annotations

import collections
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anyjev import Question

from mesa_anyjev.cards import is_identifier, load_card
from mesa_anyjev.ols import Candidate, OLSLike, iri_for, to_candidates
from mesa_anyjev.provenance.models import LabelRow
from mesa_anyjev.provenance.store import ProvenanceStore
from mesa_anyjev.questions import QUESTIONS
from mesa_anyjev.registry import (
    ASPECTS,
    ONTOLOGY_REGISTRY,
    VALUE_KINDS,
    allowed_for_aspect,
    prefix_of,
)
from mesa_anyjev.states import (
    avu_state,
    candidate_state,
    column_state,
    ontology_state,
    state_sha256,
    value_kind_state,
)

WEIGHTS = {
    "curator": 1.0,
    "curator_implicit": 0.7,
    "accepted_avu": 0.8,
    "consensus_all": 0.8,
    "consensus_majority": 0.6,
    "consensus_negative": 0.5,
    "teacher": 0.5,
    "hosted_jev": 0.5,
    "gold": 1.0,
}

_REGISTRY_IDS = [e.id for e in ONTOLOGY_REGISTRY]


@dataclass
class IngestReport:
    inserted: int = 0
    skipped_existing: int = 0
    per_question: dict[str, collections.Counter[str]] = field(default_factory=dict)
    terms_resolved: int = 0
    terms_missing: list[str] = field(default_factory=list)
    source_ref: str = ""

    def count(self, question_id: str, source: str, n: int = 1) -> None:
        self.per_question.setdefault(question_id, collections.Counter())[source] += n

    def summary(self) -> dict[str, Any]:
        return {
            "inserted": self.inserted,
            "skipped_existing": self.skipped_existing,
            "terms_resolved": self.terms_resolved,
            "terms_missing": len(self.terms_missing),
            "per_question": {q: dict(c) for q, c in self.per_question.items()},
            "source_ref": self.source_ref,
        }


def _row(
    question_id: str,
    state: dict[str, Any],
    label_index: int,
    source: str,
    card: str,
    source_ref: str,
) -> LabelRow:
    q = QUESTIONS[question_id].question
    return LabelRow(
        question_id=question_id,
        question_key=q.key,
        state_sha256=state_sha256(state),
        state_json=state,
        label_index=label_index,
        label_source=source,  # type: ignore[arg-type]
        weight=WEIGHTS[source],
        source_ref=source_ref,
        card=card,
    )


def _value_kind(value: str, ols_label: str, column: str | None, site_codes: set[str]) -> int:
    v = value.strip()
    if v.lower() == ols_label.strip().lower():
        return VALUE_KINDS.index("the term label")
    if v in site_codes:
        return VALUE_KINDS.index("the site code")
    if column and v == column:
        return VALUE_KINDS.index("the column name")
    return VALUE_KINDS.index("the most frequent data value")


class TermResolver:
    """``get_term`` through the OLS layer (recorded fixtures), memoised per CURIE."""

    def __init__(self, client: OLSLike) -> None:
        self.client = client
        self.cache: dict[str, Candidate | None] = {}

    def resolve(self, curie: str) -> Candidate | None:
        if curie in self.cache:
            return self.cache[curie]
        prefix = prefix_of(curie)
        ontology_id = prefix.lower()
        try:
            hit = self.client.get_term(ontology_id, iri_for(curie))
        except Exception:
            hit = None
        cand = None
        if hit:
            cands = to_candidates([hit], ontology_id, f"get_term:{curie}")
            cand = cands[0] if cands else None
        self.cache[curie] = cand
        return cand


def ingest_neon_eval(
    store: ProvenanceStore, eval_root: str | Path, resolver: TermResolver
) -> IngestReport:
    root = Path(eval_root).expanduser()
    validated_path = root / "results" / "validated.json"
    runs: list[dict[str, Any]] = json.loads(validated_path.read_text(encoding="utf-8"))
    ref = f"neon-avu-eval/results/validated.json@{hashlib.sha256(validated_path.read_bytes()).hexdigest()[:12]}"
    report = IngestReport(source_ref=ref)
    cards = {c.stem: load_card(c) for c in sorted((root / "cards").glob("*.md"))}
    n_models = len({r["model"] for r in runs})
    rows: list[LabelRow] = []

    # ---- gather: valid AVUs per run, models per (card, curie), columns per (card, curie) ----
    valid: list[tuple[dict[str, Any], dict[str, Any]]] = []
    models_by_pair: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    columns_by_pair: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    aspects_by_pair: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for run in runs:
        for avu in run["avus"]:
            if not (
                avu.get("resolves")
                and not avu.get("obsolete")
                and avu.get("canonical")
                and avu.get("curie")
            ):
                continue
            valid.append((run, avu))
            pair = (run["card"], avu["curie"])
            models_by_pair[pair].add(run["model"])
            if avu.get("column"):
                columns_by_pair[pair][avu["column"]] += 1
            if avu.get("aspect") in ASPECTS:
                aspects_by_pair[pair][avu["aspect"]] += 1

    def consensus_source(pair: tuple[str, str]) -> str:
        n = len(models_by_pair[pair])
        if n >= n_models:
            return "consensus_all"
        if n >= 2:
            return "consensus_majority"
        return "consensus_negative"

    # ---- term.fits: one state per unique (card, curie) ----------------------------------------
    for pair in sorted(models_by_pair):
        card_name, curie = pair
        card = cards.get(card_name)
        if card is None:
            continue
        cand = resolver.resolve(curie)
        if cand is None:
            report.terms_missing.append(curie)
            continue
        report.terms_resolved += 1
        column: str | None = (
            columns_by_pair[pair].most_common(1)[0][0] if columns_by_pair[pair] else None
        )
        aspect = aspects_by_pair[pair].most_common(1)[0][0] if aspects_by_pair[pair] else "other"
        target: Any = None
        scope = "dataset"
        if column and any(c.name == column for c in card.columns):
            target, scope = card.column(column), "column"
        elif aspect in ("environment", "location") and len(card.sites) == 1:
            target, scope = card.sites[0], "site"
        state = candidate_state(card, scope, target, aspect, cand.as_state(), 0)
        source = consensus_source(pair)
        rows.append(
            _row(
                "term.fits",
                state,
                0 if source != "consensus_negative" else 1,
                source,
                card_name,
                ref,
            )
        )
        report.count("term.fits", source)

    # ---- column-level questions --------------------------------------------------------------
    col_models: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    col_aspects: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    col_prefixes: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for run, avu in valid:
        if avu.get("column"):
            key = (run["card"], avu["column"])
            col_models[key].add(run["model"])
            if avu.get("aspect") in ASPECTS:
                col_aspects[key][avu["aspect"]] += 1
            col_prefixes[key][prefix_of(avu["curie"]).lower()] += 1
    for card_name, card in cards.items():
        for col in card.columns:
            key = (card_name, col.name)
            n = len(col_models.get(key, set()))
            st = column_state(card, col)
            if n >= 2:
                rows.append(_row("column.annotate", st, 0, "consensus_majority", card_name, ref))
                report.count("column.annotate", "consensus_majority")
            elif n == 0 and not is_identifier(col):
                rows.append(_row("column.annotate", st, 1, "consensus_negative", card_name, ref))
                report.count("column.annotate", "consensus_negative")
            if n >= 2 and col_aspects.get(key):
                aspect, votes = col_aspects[key].most_common(1)[0]
                if votes >= 2:
                    rows.append(
                        _row(
                            "column.aspect",
                            st,
                            ASPECTS.index(aspect),
                            "consensus_majority",
                            card_name,
                            ref,
                        )
                    )
                    report.count("column.aspect", "consensus_majority")
                    used = {p for p in col_prefixes[key] if p in _REGISTRY_IDS}
                    if used:
                        top = col_prefixes[key].most_common(1)[0][0]
                        if top in _REGISTRY_IDS:
                            rows.append(
                                _row(
                                    "column.ontology",
                                    {**st, "aspect": aspect},
                                    _REGISTRY_IDS.index(top),
                                    "consensus_majority",
                                    card_name,
                                    ref,
                                )
                            )
                            report.count("column.ontology", "consensus_majority")
                        for e in ONTOLOGY_REGISTRY:
                            if e.id not in allowed_for_aspect(aspect):
                                continue
                            ost = ontology_state(
                                card, col, aspect, e.id, e.option_text, sorted(e.aspects)
                            )
                            if e.id in used:
                                rows.append(
                                    _row(
                                        "column.ontology_fits",
                                        ost,
                                        0,
                                        "consensus_majority",
                                        card_name,
                                        ref,
                                    )
                                )
                                report.count("column.ontology_fits", "consensus_majority")
                            else:
                                rows.append(
                                    _row(
                                        "column.ontology_fits",
                                        ost,
                                        1,
                                        "consensus_negative",
                                        card_name,
                                        ref,
                                    )
                                )
                                report.count("column.ontology_fits", "consensus_negative")

    # ---- avu.value_kind and avu.keep per valid AVU -----------------------------------------------
    for run, avu in valid:
        card = cards.get(run["card"])
        if card is None:
            continue
        pair = (run["card"], avu["curie"])
        cand = resolver.resolve(avu["curie"])
        if cand is None:
            continue
        canonical = avu["canonical"]
        column = None
        if avu.get("column") and any(c.name == avu.get("column") for c in card.columns):
            column = str(avu["column"])
        aspect = str(avu.get("aspect")) if avu.get("aspect") in ASPECTS else "other"
        site_codes = {s.code for s in card.sites}
        if column:
            kind = _value_kind(
                str(avu.get("value") or ""), str(avu.get("ols_label") or ""), column, site_codes
            )
            vst = value_kind_state(card, card.column(column), cand.as_state(), aspect)
            source = (
                "consensus_majority"
                if kind != VALUE_KINDS.index("the most frequent data value")
                else "consensus_negative"
            )
            rows.append(_row("avu.value_kind", vst, kind, source, run["card"], ref))
            report.count("avu.value_kind", source)
        siblings = [a["canonical"] for a in run["avus"] if a.get("canonical") and a is not avu]
        scope = (
            "column" if column else ("site" if str(avu.get("value")) in site_codes else "dataset")
        )
        kst = avu_state(
            card,
            canonical,
            cand.as_state(),
            scope,
            column or (str(avu.get("value")) if scope == "site" else None),
            siblings,
        )
        source = consensus_source(pair)
        keep = source != "consensus_negative" or bool(avu.get("label_match"))
        if source == "consensus_negative" and keep:
            continue  # a single-model but well-formed AVU is ambiguous: no label
        rows.append(_row("avu.keep", kst, 0 if keep else 1, source, run["card"], ref))
        report.count("avu.keep", source)

    inserted = store.insert_labels(rows)
    report.inserted = inserted
    report.skipped_existing = len(rows) - inserted
    return report


@dataclass
class LabelledSet:
    states: list[dict[str, Any]]
    labels: list[int]
    weights: list[float]
    cards: list[str]

    def class_counts(self) -> dict[int, int]:
        return dict(collections.Counter(self.labels))

    def __len__(self) -> int:
        return len(self.labels)


def labelled_states(
    store: ProvenanceStore,
    question: Question,
    *,
    min_weight: float = 0.0,
    exclude_cards: Sequence[str] = (),
) -> LabelledSet:
    """One (state, label, weight, card) per state, the highest-weight source winning."""
    best: dict[str, dict[str, Any]] = {}
    for row in store.labels_for(question.key, min_weight=min_weight, exclude_cards=exclude_cards):
        cur = best.get(row["state_sha256"])
        if cur is None or float(row["weight"]) > float(cur["weight"]):
            best[row["state_sha256"]] = row
    rows = sorted(best.values(), key=lambda r: r["state_sha256"])
    return LabelledSet(
        states=[r["state_json"] for r in rows],
        labels=[int(r["label_index"]) for r in rows],
        weights=[float(r["weight"]) for r in rows],
        cards=[str(r["card"] or "") for r in rows],
    )

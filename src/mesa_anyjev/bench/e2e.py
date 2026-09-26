"""``bench e2e``: the whole graph on the fixture cards, several repetitions per planner, scored
as *agreement* (plan amendment A6 measures rep-to-rep agreement instead of asserting
determinism): the Jaccard overlap of proposed AVU triples between repetitions, and the recall
of the neon-avu-eval consensus terms (proposed by every model, or by at least two) for each
card. Consensus is computed from ``validated.json`` itself, never retyped."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mesa_anyjev.cards import DatasetCard, load_card
from mesa_anyjev.pipeline import AnnotationRun, Annotator


def consensus_from_validated(path: Path) -> dict[str, dict[str, set[str]]]:
    """card -> {"all": CURIEs every model proposed, "majority": CURIEs at least two models
    proposed}, from neon-avu-eval's ``validated.json`` (one entry per model, rep and card,
    each with its validated AVUs; only resolving, non-obsolete CURIEs count)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("results", [])
    models_per_card: dict[str, set[str]] = {}
    by_card: dict[str, dict[str, set[str]]] = {}
    for entry in entries:
        card = str(entry.get("card") or "")
        model = str(entry.get("model") or "")
        if not card or not model:
            continue
        models_per_card.setdefault(card, set()).add(model)
        for avu in entry.get("avus") or []:
            curie = avu.get("curie")
            if not curie or avu.get("resolves") is False or avu.get("obsolete"):
                continue
            by_card.setdefault(card, {}).setdefault(str(curie), set()).add(model)
    per_card: dict[str, dict[str, set[str]]] = {}
    for card, curies_ in by_card.items():
        n_models = len(models_per_card.get(card, set())) or 1
        per_card[card] = {
            "all": {c for c, ms in curies_.items() if len(ms) >= n_models},
            "majority": {c for c, ms in curies_.items() if len(ms) >= 2},
        }
    return per_card


def triples(run: AnnotationRun) -> set[tuple[str, str, str]]:
    return {(p.avu["attribute"], p.avu["value"], p.avu["unit"]) for p in run.proposals if p.avu}


def curies(run: AnnotationRun) -> set[str]:
    return {p.candidate.curie for p in run.proposals if p.candidate.curie}


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_e2e(
    make_annotator: Any,
    cards: list[Path],
    *,
    planners: list[str],
    reps: int = 2,
    consensus: dict[str, dict[str, set[str]]] | None = None,
) -> dict[str, Any]:
    """``make_annotator(planner_kind) -> Annotator``; returns per planner and card the rep
    agreement and consensus recall, plus pooled means."""
    out: dict[str, Any] = {"reps": reps, "planners": {}}
    for planner in planners:
        annot: Annotator = make_annotator(planner)
        per_card: dict[str, Any] = {}
        for path in cards:
            card: DatasetCard = load_card(path)
            runs = [annot.annotate(card) for _ in range(reps)]
            sets = [triples(r) for r in runs]
            pair = [jaccard(sets[i], sets[j]) for i in range(reps) for j in range(i + 1, reps)]
            entry: dict[str, Any] = {
                "n_proposals": [len(s) for s in sets],
                "rep_agreement": sum(pair) / len(pair) if pair else 1.0,
                "planner_fallback": runs[0].plan.fallback,
                "audit": runs[0].audit,
            }
            if consensus and card.name in consensus:
                c = consensus[card.name]
                got = curies(runs[0])
                entry["consensus_all_recall"] = (
                    len(got & c["all"]) / len(c["all"]) if c["all"] else None
                )
                entry["consensus_majority_recall"] = (
                    len(got & c["majority"]) / len(c["majority"]) if c["majority"] else None
                )
            per_card[card.name] = entry
        vals = [e["rep_agreement"] for e in per_card.values()]
        rec_all = [
            e["consensus_all_recall"]
            for e in per_card.values()
            if e.get("consensus_all_recall") is not None
        ]
        rec_maj = [
            e["consensus_majority_recall"]
            for e in per_card.values()
            if e.get("consensus_majority_recall") is not None
        ]
        out["planners"][planner] = {
            "cards": per_card,
            "rep_agreement_mean": sum(vals) / len(vals) if vals else None,
            "consensus_all_recall_mean": sum(rec_all) / len(rec_all) if rec_all else None,
            "consensus_majority_recall_mean": sum(rec_maj) / len(rec_maj) if rec_maj else None,
        }
    return out

"""Offline fitting and promotion (DESIGN D6, D15; amendments A2, A3, A5).

``fit_question`` fits an L1 temperature (M2) for one question: leave-one-card-out first (the
only split that may promote), refusing a fold whose classes are too small, then a final fit
on every card, saved as a new immutable artifact version whose manifest carries the LOCO
numbers. ``promote`` moves ``CURRENT`` only when held-out accuracy and ECE do not regress.
Fitting runs on a Decider with ``adaptive_shifts=False`` so the artifact freezes its prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from anyjev import Decider, Question

from mesa_anyjev.artifacts import ArtifactStore
from mesa_anyjev.bench import _metrics
from mesa_anyjev.bench.run import MIN_HELDOUT_PER_CLASS, MIN_TRAIN_PER_CLASS
from mesa_anyjev.learn.labels import LabelledSet, labelled_states
from mesa_anyjev.policy import Policy
from mesa_anyjev.provenance.store import ProvenanceStore
from mesa_anyjev.questions import QUESTIONS, lock_sha

MIN_LABELS_L1 = 100


@dataclass
class FitReport:
    question_id: str
    level: str
    status: str  # fitted | insufficient_labels | insufficient_negatives | no_folds
    n_labels: int = 0
    class_counts: dict[int, int] = field(default_factory=dict)
    folds: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped_folds: dict[str, str] = field(default_factory=dict)
    loco: dict[str, Any] = field(default_factory=dict)
    version: int | None = None
    artifact_path: str | None = None
    detail: str = ""


def _fit_decider(backend: Any, cfg: Any) -> Decider:
    return Decider(
        backend,
        level="L0",
        prior=cfg.prior,
        adaptive_shifts=False,
        max_permutations=cfg.max_permutations,
    )


def _eval(
    decider: Decider, q: Question, states: list[Any], labels: list[int], level: str
) -> dict[str, Any]:
    decs = decider.decide_batch(states, q, level=level)
    import numpy as np

    probs = np.asarray([d.probs for d in decs], dtype=float)
    out: dict[str, Any] = _metrics.summarize(probs, labels)
    out["cov@10%"] = float(_metrics.coverage_at_risk(probs, labels, target=0.10))
    counts: dict[int, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    out["class_counts"] = counts
    out["n_neg"] = sum(v for k, v in counts.items() if k != 0) if q.kind == "noul" else None
    out["levels"] = sorted({d.level for d in decs})
    out["prior_frozen"] = all(
        str(d.diagnostics.get("prior_method", "")).startswith("frozen:") for d in decs
    )
    return out


def fit_question(
    provider: Any,
    store: ProvenanceStore,
    artifacts: ArtifactStore,
    question_id: str,
    *,
    policy: Policy,
    level: str = "L1",
    loco: bool = True,
    backend_kind: str,
    fitted_on: dict[str, Any] | None = None,
) -> FitReport:
    q = QUESTIONS[question_id].question
    thresholds = policy.thresholds(question_id)
    ls: LabelledSet = labelled_states(store, q, min_weight=thresholds.min_weight)
    report = FitReport(
        question_id, level, "fitted", n_labels=len(ls), class_counts=ls.class_counts()
    )
    if level != "L1":
        report.status = "unsupported"
        report.detail = "L2 fitting arrives with milestone M3"
        return report
    if len(ls) < MIN_LABELS_L1:
        report.status = "insufficient_labels"
        report.detail = f"{len(ls)} < {MIN_LABELS_L1} labels at min_weight {thresholds.min_weight}"
        return report
    if q.kind == "noul" and (
        len(ls.class_counts()) < 2 or min(ls.class_counts().values()) < MIN_TRAIN_PER_CLASS
    ):
        report.status = "insufficient_negatives"
        report.detail = (
            f"class counts {ls.class_counts()}; need >= {MIN_TRAIN_PER_CLASS} per class (A2)"
        )
        return report
    backend = provider.backend
    with provider.lock:
        if loco:
            for card in sorted(set(ls.cards)):
                train = [
                    (s, lbl)
                    for s, lbl, c in zip(ls.states, ls.labels, ls.cards, strict=True)
                    if c != card
                ]
                test = [
                    (s, lbl)
                    for s, lbl, c in zip(ls.states, ls.labels, ls.cards, strict=True)
                    if c == card
                ]
                tr = _counts(train)
                te = _counts(test)
                if q.kind == "noul" and (len(tr) < 2 or min(tr.values()) < MIN_TRAIN_PER_CLASS):
                    report.skipped_folds[card] = f"insufficient_train_per_class {tr}"
                    continue
                if q.kind == "noul" and (len(te) < 2 or min(te.values()) < MIN_HELDOUT_PER_CLASS):
                    report.skipped_folds[card] = f"insufficient_heldout_per_class {te}"
                    continue
                dec = _fit_decider(backend, provider.cfg)
                dec.calibrate(q, [s for s, _ in train], [lbl for _, lbl in train])
                report.folds[card] = _eval(
                    dec, q, [s for s, _ in test], [lbl for _, lbl in test], level
                )
            if not report.folds:
                report.status = "no_folds"
                report.detail = "every leave-one-card-out fold was skipped"
                return report
            report.loco = _pool(report.folds)
        final = _fit_decider(backend, provider.cfg)
        final.calibrate(q, ls.states, ls.labels)
    manifest = {
        "fitted_on": fitted_on
        or {"backend_kind": backend_kind, "served_model": getattr(provider, "served_model", None)},
        "per_question": {
            question_id: {
                "key": q.key,
                "level": level,
                "n_calib": len(ls),
                "class_counts": ls.class_counts(),
                "min_weight": thresholds.min_weight,
            }
        },
        "validation": {
            question_id: {
                "loco": report.loco,
                "folds": report.folds,
                "skipped_folds": report.skipped_folds,
            }
        },
        "questions_lock_sha": lock_sha(),
    }
    version = artifacts.save(final, manifest)
    report.version = version.version
    report.artifact_path = str(version.path)
    return report


def _counts(items: list[tuple[Any, int]]) -> dict[int, int]:
    out: dict[int, int] = {}
    for _, lbl in items:
        out[lbl] = out.get(lbl, 0) + 1
    return out


def _pool(folds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = sum(int(f["n"]) for f in folds.values())
    if not n:
        return {}
    w = {k: int(f["n"]) / n for k, f in folds.items()}
    keys = ("acc", "ece", "brier", "nll", "cov@5%", "cov@10%")
    pooled = {k: float(sum(w[c] * float(folds[c][k]) for c in folds)) for k in keys}
    pooled["n"] = n
    pooled["n_neg"] = sum(int(f.get("n_neg") or 0) for f in folds.values())
    pooled["n_folds"] = len(folds)
    return pooled


def promote(artifacts: ArtifactStore, version: int, question_id: str) -> tuple[bool, str]:
    """Move CURRENT to ``version`` unless its LOCO acc/ece regress against CURRENT (D6)."""
    candidates = {v.version: v for v in artifacts.versions()}
    if version not in candidates:
        return False, f"no version v{version}"
    new = candidates[version].manifest.get("validation", {}).get(question_id, {}).get("loco", {})
    current = artifacts.current()
    if current is not None and current.version != version:
        old = current.manifest.get("validation", {}).get(question_id, {}).get("loco", {})
        regresses = bool(old and new) and (
            new.get("acc", 0.0) < old.get("acc", 0.0) - 1e-9
            or new.get("ece", 1.0) > old.get("ece", 1.0) + 1e-9
        )
        if regresses:
            return (
                False,
                f"v{version} regresses against v{current.version}: "
                f"acc {new.get('acc'):.3f} vs {old.get('acc'):.3f}, ece {new.get('ece'):.3f} vs {old.get('ece'):.3f}",
            )
    artifacts.promote(version)
    return True, f"CURRENT -> v{version}"

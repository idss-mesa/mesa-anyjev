"""``mesa-anyjev hosted score``: send a batch of stored states to MotherDuck ``prompt_jev`` and
record the answers as a normal sidecar run (``backend_kind='motherduck'``, ``data_left_host``
true), then score them against the labels the way the bench does and write their own results
file (``motherduck__prompt_jev.motherduck.json``), never mixed with gateway or local rows.

Sources: ``labels`` (the labelled states of one question, exact ``state_json`` so the
``state_sha256`` matches the local runs) and ``run`` (re-score the decisions of a sidecar run).
Re-scoring effective AVUs straight from the DuckLake history (``ducklake``) waits for states
that carry the dataset card; the history rows do not (D16 keeps the door open)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from anyjev import Question

from mesa_anyjev import __version__
from mesa_anyjev.bench.run import _cell, environment, write_results
from mesa_anyjev.config import Config, config_sha256
from mesa_anyjev.learn.labels import labelled_states
from mesa_anyjev.pipeline import ANYJEV_COMMIT
from mesa_anyjev.provenance.models import DecisionRow, RunRow
from mesa_anyjev.provenance.store import ProvenanceStore
from mesa_anyjev.providers.base import DecisionRecord
from mesa_anyjev.providers.motherduck_provider import HostedJevProvider, sql_for
from mesa_anyjev.questions import QUESTIONS, lock_sha
from mesa_anyjev.states import state_sha256

SLUG = "motherduck__prompt_jev"


@dataclass
class HostedBatch:
    question_id: str
    states: list[dict[str, Any]]
    labels: list[int | None]
    cards: list[str | None]
    source: str
    chars: int = 0

    def __post_init__(self) -> None:
        from anyjev.state import render_state

        self.chars = sum(len(render_state(s)) for s in self.states)

    def estimate(self, batch_size: int) -> dict[str, Any]:
        rows = len(self.states)
        return {
            "question_id": self.question_id,
            "rows": rows,
            "requests": -(-rows // max(1, batch_size)) if rows else 0,
            "input_chars": self.chars,
            "input_tokens_approx": self.chars // 4,
            "sql": sql_for(QUESTIONS[self.question_id].question),
        }


@dataclass
class HostedReport:
    run_id: UUID | None
    question_id: str
    n: int
    n_null: int
    n_abstain: int
    cell: dict[str, Any] = field(default_factory=dict)
    results_path: str | None = None
    dry_run: bool = False
    estimate: dict[str, Any] = field(default_factory=dict)


def batch_from_labels(
    store: ProvenanceStore, question_id: str, *, min_weight: float, limit: int | None = None
) -> HostedBatch:
    q: Question = QUESTIONS[question_id].question
    ls = labelled_states(store, q, min_weight=min_weight)
    n = len(ls.states) if limit is None else min(limit, len(ls.states))
    return HostedBatch(
        question_id,
        list(ls.states[:n]),
        [int(x) for x in ls.labels[:n]],
        [str(c) for c in ls.cards[:n]],
        "labels",
    )


def batch_from_run(
    store: ProvenanceStore, run_id: UUID, question_id: str, *, limit: int | None = None
) -> HostedBatch:
    rows = [d for d in store.decisions(run_id) if d["question_id"] == question_id]
    if limit is not None:
        rows = rows[:limit]
    run = store.run(run_id) or {}
    return HostedBatch(
        question_id,
        [dict(d["state_json"] or {}) for d in rows],
        [None] * len(rows),
        [str(run.get("card_name") or "")] * len(rows),
        f"run:{run_id}",
    )


def score_batch(
    provider: HostedJevProvider,
    batch: HostedBatch,
    *,
    store: ProvenanceStore,
    cfg: Config,
    actor: str,
    out_dir: str | Path | None = None,
    date: str | None = None,
    dry_run: bool = False,
) -> HostedReport:
    q = QUESTIONS[batch.question_id].question
    if dry_run:
        return HostedReport(
            None,
            batch.question_id,
            len(batch.states),
            0,
            0,
            dry_run=True,
            estimate=batch.estimate(cfg.motherduck.batch_size),
        )
    started = time.monotonic()
    run = RunRow(
        actor=actor,
        card_name=f"hosted:{batch.source}",
        card_sha256=state_sha256({"source": batch.source, "question": batch.question_id}),
        planner="none",
        plan_json={},
        backend_kind="motherduck",
        canonical_model=provider.model,
        served_model=None,
        anyjev_commit=ANYJEV_COMMIT,
        mesa_anyjev_version=__version__,
        questions_lock_sha=lock_sha(),
        policy_profile=cfg.policy.profile,
        config_sha256=config_sha256(cfg),
        hosted_provider_used=True,
        data_left_host=True,
        egress_region=cfg.motherduck.region,
    )
    store.begin_run(run)
    recs = provider.decide_batch(batch.states, q)
    rows = [
        _decision_row(run.run_id, i, rec, QUESTIONS[batch.question_id].scope)
        for i, rec in enumerate(recs, start=1)
    ]
    store.insert_decisions(rows)
    n_null = sum(1 for r in recs if r.diagnostics.get("hosted_null"))
    n_abstain = sum(1 for r in recs if r.answer_index < 0)
    store.finish_run(
        run.run_id,
        "decided",
        finished_at=datetime.now(tz=UTC),
        n_prompts=provider.rows_sent,
        seconds=round(time.monotonic() - started, 3),
    )
    report = HostedReport(run.run_id, batch.question_id, len(recs), n_null, n_abstain)
    labelled = [
        (r, lbl)
        for r, lbl in zip(recs, batch.labels, strict=True)
        if lbl is not None and r.probs is not None
    ]
    if labelled:
        positive = 0 if q.kind == "noul" else None
        cell = _cell([r for r, _ in labelled], [lbl for _, lbl in labelled], positive=positive)
        cell["n_scored"] = len(recs)
        cell["n_null"] = n_null
        cell["n_abstain"] = n_abstain
        cell["seconds"] = round(time.monotonic() - started, 2)
        cell["loco"] = False
        cell["masked"] = False
        cell["hosted"] = True
        report.cell = cell
        if out_dir is not None:
            result = {
                "task": f"hosted_{batch.question_id.replace('.', '_')}",
                "question_id": batch.question_id,
                "kind": q.kind,
                "n": len(labelled),
                "n_items": len(labelled),
                "license": "CC BY 4.0 (NEON metadata via neon-avu-eval)",
                "source": batch.source,
                "notes": "MotherDuck prompt_jev; level none, calibration typesafe; no LOCO (nothing is fitted)",
                "loco": False,
                "cells": {"hosted": cell},
                "run_id": str(run.run_id),
            }
            env = environment(provider, "motherduck")
            env["egress_region"] = cfg.motherduck.region
            report.results_path = str(
                write_results(
                    [result],
                    out_dir,
                    SLUG,
                    "motherduck",
                    env,
                    date=date or datetime.now(tz=UTC).strftime("%Y-%m-%d"),
                )
            )
    return report


def _decision_row(run_id: UUID, seq: int, rec: DecisionRecord, scope: str) -> DecisionRow:
    scalars = {k: v for k, v in rec.diagnostics.items() if isinstance(v, int | float | str | bool)}
    return DecisionRow(
        run_id=run_id,
        seq=seq,
        question_id=rec.question_id,
        question_key=rec.question_key,
        kind=rec.kind,
        k=len(rec.options),
        scope=scope,  # type: ignore[arg-type]
        column_name=(rec.state.get("column") or {}).get("name"),
        site_code=(rec.state.get("site") or {}).get("code"),
        state_sha256=state_sha256(rec.state),
        state_json=rec.state,
        prompt_sha256=str(scalars.get("prompt_sha256") or ""),
        provider=rec.provider,
        method=rec.method,
        model=rec.model,
        served_model=None,
        level="none",
        calibration=rec.calibration,
        probs=rec.probs,
        answer_index=rec.answer_index,
        answer=rec.answer,
        confidence=rec.confidence,
        p_true=rec.p_true,
        margin=rec.margin,
        score_value=rec.score_value,
        outcome="decider_unavailable"
        if scalars.get("hosted_null")
        else ("abstain" if rec.answer_index < 0 else "proposed"),
    )


def summarize(reports: Sequence[HostedReport]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": r.question_id,
            "run_id": str(r.run_id) if r.run_id else None,
            "n": r.n,
            "n_null": r.n_null,
            "n_abstain": r.n_abstain,
            **({k: r.cell[k] for k in ("acc", "ece", "cov@5%", "cov@10%", "n_neg") if k in r.cell}),
            **({"estimate": r.estimate} if r.dry_run else {}),
            "results": r.results_path,
        }
        for r in reports
    ]

"""The bench runner (AnyJev ground rule 1; DESIGN D6; amendments A2, A9).

For each task and level: raw and L0 are decided once over every item (running prior reset
first); L1 is fitted and evaluated per leave-one-card-out fold (the only split that may set
a threshold or promote an artifact). Every cell records acc, macro_f1, brier, nll, ece,
cov@5%, cov@10%, aurc, the flip rate under option reversal (choice) or phrasing (noul), the
class counts, missing labels, prompts and milliseconds per decision. ``environment()`` is
stored with every results file. Unsupported levels get an explicit ``not applicable`` cell.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from anyjev import Question

from mesa_anyjev import __version__
from mesa_anyjev.bench import _metrics
from mesa_anyjev.bench.tasks.base import Item, Task
from mesa_anyjev.providers.base import DecisionRecord, LevelUnavailable
from mesa_anyjev.questions import lock_sha

LEVELS = ("raw", "L0", "L1", "L2")
MIN_TRAIN_PER_CLASS = 30
MIN_HELDOUT_PER_CLASS = 5


def environment(provider: Any, backend_kind: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "mesa_anyjev": __version__,
        "anyjev_commit": "795a4970b47218b7c0686cd579d691fc2cf8df2f",
        "questions_lock_sha": lock_sha(),
        "backend_kind": backend_kind,
        "model": getattr(provider, "model", None),
        "served_model": getattr(provider, "served_model", None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
    }
    smi = shutil.which("nvidia-smi")
    env["nvidia_smi"] = None
    if smi:
        try:
            env["nvidia_smi"] = subprocess.run(  # noqa: S603
                [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            env["nvidia_smi"] = None
    return env


def _probs(recs: Sequence[DecisionRecord]) -> np.ndarray:
    """Stack distributions; per-item questions (the choice26 control) have different K per
    item, so shorter rows are zero-padded (argmax, brier, nll and ECE are unaffected)."""
    width = max((len(r.probs or []) for r in recs), default=0)
    out = np.zeros((len(recs), width), dtype=float)
    for i, r in enumerate(recs):
        row = r.probs or []
        out[i, : len(row)] = row
    return out


def _cell(
    recs: Sequence[DecisionRecord],
    labels: Sequence[int],
    *,
    positive: int | None = 0,
    probs_flipped: np.ndarray | None = None,
) -> dict[str, Any]:
    probs = _probs(recs)
    y = list(labels)
    out: dict[str, Any] = (
        _metrics.summarize(probs, y, probs_flipped=probs_flipped)
        if probs_flipped is not None
        else _metrics.summarize(probs, y)
    )
    out["cov@5%"] = out.get("cov@5%")
    out["cov@10%"] = float(_metrics.coverage_at_risk(probs, y, target=0.10))
    counts = {
        int(k): int(v) for k, v in zip(*np.unique(np.asarray(y), return_counts=True), strict=True)
    }
    out["class_counts"] = counts
    out["n_neg"] = sum(v for k, v in counts.items() if positive is not None and k != positive)
    out["levels"] = sorted({r.level for r in recs})
    out["mean_p_true"] = (
        float(np.mean([r.p_true for r in recs if r.p_true is not None]))
        if recs and recs[0].kind == "noul"
        else None
    )
    return out


def _flip_probs(provider: Any, task: Task, items: Sequence[Item], level: str) -> np.ndarray | None:
    """Probabilities under the reversed option listing, mapped back to the canonical order."""
    q = task.question
    if q.kind != "choice" or task.meta.get("per_item_question"):
        return None
    rev = Question.choice(q.text, list(reversed(q.options)), name=q.id + ".reversed")
    try:
        recs = provider.decide_batch([s for s, _ in items], rev, level=level)
    except LevelUnavailable:
        return None
    return _probs(recs)[:, ::-1]


def _decide_items(
    provider: Any, task: Task, items: Sequence[Item], level: str
) -> list[DecisionRecord]:
    if task.meta.get("per_item_question"):
        out: list[DecisionRecord] = []
        for state, _ in items:
            out.extend(
                provider.decide_batch(
                    [state["state"]], state["question"], level=level, allow_over_cap=True
                )
            )
        return out
    recs: list[DecisionRecord] = list(
        provider.decide_batch([s for s, _ in items], task.question, level=level)
    )
    return recs


def run_task(
    provider: Any,
    task: Task,
    levels: Sequence[str] = LEVELS,
    *,
    loco: bool = True,
    flip_probe: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task": task.name,
        "question_id": task.meta.get("question_id"),
        "n_items": len(task.items),
        "class_counts": task.class_counts(),
        "license": task.license,
        "source": task.source,
        "notes": task.notes,
        "loco": loco,
        "cells": {},
    }
    positive = 0 if task.question.kind == "noul" else None
    labels = [lbl for _, lbl in task.items]
    backend = getattr(provider, "backend", None)
    for level in levels:
        if level not in task.levels_supported:
            result["cells"][level] = {
                "not_applicable": True,
                "reason": f"task supports {task.levels_supported}",
            }
            continue
        started = time.monotonic()
        prompts0 = int(getattr(backend, "prompts_seen", getattr(backend, "requests", 0)))
        if level in ("raw", "L0"):
            if hasattr(provider, "reset_running_prior"):
                provider.reset_running_prior()
            try:
                recs = _decide_items(provider, task, task.items, level)
            except LevelUnavailable as exc:
                result["cells"][level] = {"not_applicable": True, "reason": str(exc)}
                continue
            except Exception as exc:
                result["cells"][level] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            flipped = _flip_probs(provider, task, task.items, level) if flip_probe else None
            cell = _cell(recs, labels, positive=positive, probs_flipped=flipped)
            if task.question.kind == "noul":
                flips = [
                    r.diagnostics.get("order_flip_l0")
                    for r in recs
                    if r.diagnostics.get("order_flip_l0") is not None
                ]
                cell["flip"] = float(np.mean([float(f) for f in flips])) if flips else None  # type: ignore[arg-type]
            cell["missing_labels"] = int(getattr(provider, "last_missing_labels", 0))
        elif level == "L1":
            cell = (
                _fit_eval_loco(provider, task, level)
                if loco
                else {"not_applicable": True, "reason": "L1 needs --loco"}
            )
        else:
            cell = {
                "not_applicable": True,
                "reason": "L2 arrives with milestone M3 (local weights)",
            }
        if not cell.get("not_applicable"):
            cell["seconds"] = round(time.monotonic() - started, 2)
            cell["prompts"] = (
                int(getattr(backend, "prompts_seen", getattr(backend, "requests", 0))) - prompts0
            )
            cell["ms_per_decision"] = round(1000 * cell["seconds"] / max(1, len(task.items)), 1)
            cell["loco"] = loco if level == "L1" else False
            cell["masked"] = bool(task.meta.get("masked", False))
        result["cells"][level] = cell
    return result


def _fit_eval_loco(provider: Any, task: Task, level: str) -> dict[str, Any]:
    """Fit on every card but one, evaluate on the held-out card, pool the held-out decisions."""
    if not task.cards or len(set(task.cards)) < 2:
        return {"not_applicable": True, "reason": "leave-one-card-out needs at least two cards"}
    positive = 0 if task.question.kind == "noul" else None
    pooled_recs: list[DecisionRecord] = []
    pooled_labels: list[int] = []
    folds: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    q = task.question
    for card, test, train in task.leave_one_card_out():
        tr_counts = Task("f", q, list(train), "", "").class_counts()
        te_counts = Task("f", q, list(test), "", "").class_counts()
        if positive is not None and (
            min(tr_counts.values(), default=0) < MIN_TRAIN_PER_CLASS or len(tr_counts) < 2
        ):
            skipped[card] = f"insufficient_train_per_class {tr_counts}"
            continue
        if positive is not None and (
            min(te_counts.values(), default=0) < MIN_HELDOUT_PER_CLASS or len(te_counts) < 2
        ):
            skipped[card] = f"insufficient_heldout_per_class {te_counts}"
            continue
        if hasattr(provider, "reset_running_prior"):
            provider.reset_running_prior()
        provider.calibrate(q, [s for s, _ in train], [lbl for _, lbl in train])
        try:
            recs = provider.decide_batch([s for s, _ in test], q, level=level)
        finally:
            provider.decider._artifacts.pop(q.key, None)
        folds[card] = _cell(recs, [lbl for _, lbl in test], positive=positive)
        pooled_recs.extend(recs)
        pooled_labels.extend(lbl for _, lbl in test)
    if not pooled_recs:
        return {"not_applicable": True, "reason": "every fold skipped", "skipped": skipped}
    cell = _cell(pooled_recs, pooled_labels, positive=positive)
    cell["folds"] = folds
    cell["skipped_folds"] = skipped
    cell["n_folds"] = len(folds)
    return cell


def suggest_policy(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per question, the level/cell that reaches cov@5% > 0 with enough negatives, if any."""
    out: dict[str, Any] = {}
    for res in results:
        qid = res.get("question_id")
        if not qid:
            continue
        for level, cell in res["cells"].items():
            if cell.get("not_applicable") or cell.get("error") or not cell.get("loco"):
                continue
            if cell.get("n_neg", 0) < MIN_TRAIN_PER_CLASS or not cell.get("cov@5%"):
                continue
            out.setdefault(qid, {})[level] = {
                "cov@5%": cell["cov@5%"],
                "cov@10%": cell["cov@10%"],
                "ece": cell["ece"],
                "n": cell["n"],
                "n_neg": cell["n_neg"],
                "cite": f"{res['task']}.{level}",
            }
    return out


def write_results(
    results: Sequence[dict[str, Any]],
    out_dir: str | Path,
    slug: str,
    backend_kind: str,
    env: dict[str, Any],
    *,
    date: str,
) -> Path:
    directory = Path(out_dir) / date
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug}.{backend_kind}.json"
    payload = {
        "date": date,
        "backend": backend_kind,
        "environment": env,
        "tasks": {r["task"]: r for r in results},
        "policy_suggestions": suggest_policy(results),
    }
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    (directory / f"{slug}.{backend_kind}.md").write_text(markdown_table(payload), encoding="utf-8")
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _fmt(cell: dict[str, Any], key: str) -> str:
    value = cell.get(key)
    return "" if value is None else f"{float(value):.3f}"


def markdown_table(payload: dict[str, Any]) -> str:
    env = payload["environment"]
    date, backend = payload["date"], payload["backend"]
    lines = [
        f"# bench {date} · {backend} · {env.get('model')}",
        "",
        f"Every number below names `{date}/<slug>.{backend}.json`. Fake-backend rows are synthetic.",
        "",
        "| task | n | level | acc | ece | brier | cov@5% | cov@10% | flip | n_neg | prompts | ms/dec |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in payload["tasks"].items():
        for level, cell in res["cells"].items():
            if cell.get("not_applicable") or cell.get("error"):
                tag = "n/a" if cell.get("not_applicable") else "error"
                lines.append(f"| {name} | {res['n_items']} | {level} | {tag} | | | | | | | | |")
                continue
            tag = f"{level} (LOCO)" if cell.get("loco") else level
            cols = [_fmt(cell, k) for k in ("acc", "ece", "brier", "cov@5%", "cov@10%", "flip")]
            tail = [str(cell.get(k, "")) for k in ("n_neg", "prompts", "ms_per_decision")]
            lines.append(f"| {name} | {res['n_items']} | {tag} | " + " | ".join(cols + tail) + " |")
    return "\n".join(lines) + "\n"

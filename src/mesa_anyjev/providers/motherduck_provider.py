"""Hosted Jev through MotherDuck's ``prompt_jev`` (DESIGN D16, plan amendment C6).

The same frozen questions, rendered to the same ``State:`` text AnyJev prefills, go to
MotherDuck as SQL constants; the answers come back as ``STRUCT(choice, probabilities[],
confidence)`` (choice), a DOUBLE (noul) or a weighted position (score). Every record is
``provider='motherduck'``, ``method='hosted:prompt_jev'``, ``level='none'`` (no AnyJev
level exists for it), ``calibration='typesafe'`` with the full distribution re-ordered into
the frozen option order. A value the frozen options do not contain, a distribution that does
not sum to one, or a NULL answer never becomes a probability: the record abstains with
``calibration='none'`` and says why in ``diagnostics``.

Nothing here reads the token: ``MotherDuckRunner`` connects with ``md:`` and lets the DuckDB
extension read ``MOTHERDUCK_TOKEN`` (or the variable named in config) from the environment.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from anyjev import Question
from anyjev.state import render_state

from mesa_anyjev.config import MotherDuckConfig
from mesa_anyjev.providers.base import BackendCapabilities, DecisionRecord, rank_probs
from mesa_anyjev.states import state_sha256

PROVIDER = "motherduck"
METHOD = "hosted:prompt_jev"
MODEL = "motherduck:prompt_jev"
RESULT_KEYS = ("choice", "probabilities", "confidence")  # the observed STRUCT, pinned
SUM_TOLERANCE = 1e-3


def sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def instructions_for(question: Question) -> str:
    """The question text is the instruction; the options are the choice list (constants)."""
    return str(question.text)


def sql_for(question: Question, text_col: str = "state_text") -> str:
    """The ``prompt_jev`` call for one question, as it is sent (also pinned in the lock)."""
    instr = sql_literal(instructions_for(question))
    if question.kind == "noul":
        return f"prompt_jev({text_col}, {instr}, noul := TRUE)"
    opts = ", ".join(sql_literal(str(o)) for o in question.options)
    if question.kind == "score":
        return f"prompt_jev({text_col}, {instr}, score := [{opts}])"
    return f"prompt_jev({text_col}, {instr}, choice := [{opts}])"


class Runner(Protocol):
    """Executes one batch: ``(texts, question) -> one raw result per text`` (``None`` = NULL)."""

    def score(self, texts: Sequence[str], question: Question) -> list[Any]: ...

    @property
    def name(self) -> str: ...


class MotherDuckRunner:
    """Runs the real thing over an ``md:`` connection (the token comes from the environment)."""

    name = "motherduck"

    def __init__(self, cfg: MotherDuckConfig, *, connect: Callable[[], Any] | None = None):
        self.cfg = cfg
        self._connect = connect or self._default_connect
        self._con: Any | None = None
        self.requests = 0

    def _default_connect(self) -> Any:
        import os

        import duckdb

        if not os.environ.get(self.cfg.token_env):
            raise RuntimeError(f"{self.cfg.token_env} is not set; hosted Jev needs it (D16)")
        con = duckdb.connect("md:")
        con.execute("INSTALL motherduck")
        con.execute("LOAD motherduck")
        con.execute(f"CREATE DATABASE IF NOT EXISTS {self.cfg.database}")
        con.execute(f"USE {self.cfg.database}")
        return con

    @property
    def con(self) -> Any:
        if self._con is None:
            self._con = self._connect()
        return self._con

    def score(self, texts: Sequence[str], question: Question) -> list[Any]:
        con = self.con
        con.execute(
            "CREATE OR REPLACE TEMP TABLE mesa_anyjev_batch (i INTEGER, state_text VARCHAR)"
        )
        con.executemany(
            "INSERT INTO mesa_anyjev_batch VALUES (?, ?)", [[i, t] for i, t in enumerate(texts)]
        )
        self.requests += 1
        rows = con.execute(
            f"SELECT i, {sql_for(question)} AS r FROM mesa_anyjev_batch ORDER BY i"  # noqa: S608
        ).fetchall()
        out: list[Any] = [None] * len(texts)
        for i, r in rows:
            out[int(i)] = r
        return out

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None


class HostedJevProvider:
    """A ``DecisionProvider`` over a runner; never an AnyJev backend (no logits, no levels)."""

    def __init__(self, cfg: MotherDuckConfig, runner: Runner) -> None:
        self.cfg = cfg
        self.runner = runner
        self.model = MODEL
        self.served_model: str | None = None
        self.capabilities = BackendCapabilities(
            max_choice_k=26, hidden_states=False, logprob_top_k=None
        )
        self.last_missing_labels = 0
        self.rows_sent = 0

    def resolve_level(self, question: Question, level: str | None) -> str:
        return "none"

    def decide_batch(
        self, states: Sequence[dict[str, Any]], question: Question, *, level: str | None = None
    ) -> list[DecisionRecord]:
        if not states:
            return []
        texts = [render_state(s) for s in states]
        results: list[Any] = []
        step = max(1, int(self.cfg.batch_size))
        for start in range(0, len(texts), step):
            results.extend(self.runner.score(texts[start : start + step], question))
        self.rows_sent += len(texts)
        return [
            self.record(question, state, raw, text)
            for state, raw, text in zip(states, results, texts, strict=True)
        ]

    # -- parsing ----------------------------------------------------------------------------------
    def record(
        self, question: Question, state: dict[str, Any], raw: Any, text: str | None = None
    ) -> DecisionRecord:
        probs, diag = parse_result(question, raw)
        base: dict[str, Any] = {
            "question_id": question.id,
            "question_key": question.key,
            "kind": question.kind,
            "options": [str(o) for o in question.options],
            "state": state,
            "state_sha256": state_sha256(state),
            "provider": PROVIDER,
            "method": METHOD,
            "model": MODEL,
            "served_model": None,
            "level": "none",
            "diagnostics": {
                **diag,
                "egress_region": self.cfg.region,
                "prompt_sha256": hashlib.sha256((text or render_state(state)).encode()).hexdigest(),
            },
        }
        if probs is None:
            return DecisionRecord(
                calibration="none", probs=None, answer_index=-1, answer="", **base
            )
        idx, conf, margin = rank_probs(probs)
        return DecisionRecord(
            calibration="typesafe",
            probs=probs,
            answer_index=idx,
            answer=str(question.options[idx]),
            confidence=conf,
            p_true=probs[0] if question.kind == "noul" else None,
            margin=margin,
            score_value=float(diag["score_value"]) if question.kind == "score" else None,
            **base,
        )


def parse_result(question: Question, raw: Any) -> tuple[list[float] | None, dict[str, Any]]:
    """Turn one ``prompt_jev`` result into probabilities in the frozen option order, or
    ``None`` with the reason."""
    if raw is None:
        return None, {"hosted_null": True, "unavailable": True}
    if question.kind == "noul":
        try:
            p = float(raw)
        except (TypeError, ValueError):
            return None, {"hosted_shape": f"noul expected DOUBLE, got {type(raw).__name__}"}
        if not 0.0 <= p <= 1.0:
            return None, {"hosted_shape": f"noul probability out of range: {p}"}
        return [p, 1.0 - p], {"hosted_confidence": max(p, 1.0 - p)}
    if question.kind == "score":
        try:
            pos = float(raw)
        except (TypeError, ValueError):
            return None, {"hosted_shape": f"score expected DOUBLE, got {type(raw).__name__}"}
        k = len(question.options)
        probs = [0.0] * k
        lo = int(max(0, min(k - 1, pos)))
        hi = min(k - 1, lo + 1)
        frac = pos - lo
        probs[lo] += 1.0 - frac
        probs[hi] += frac
        total = sum(probs)
        return [p / total for p in probs], {"score_value": pos}
    if not isinstance(raw, dict) or tuple(k for k in RESULT_KEYS if k in raw) != RESULT_KEYS:
        return None, {
            "hosted_shape": f"choice expected STRUCT{RESULT_KEYS}, got {type(raw).__name__}"
        }
    by_value: dict[str, float] = {}
    for entry in raw.get("probabilities") or []:
        if isinstance(entry, dict) and "value" in entry:
            by_value[str(entry["value"])] = float(entry.get("probability") or 0.0)
    options = [str(o) for o in question.options]
    unknown = sorted(set(by_value) - set(options))
    if unknown:
        return None, {"hosted_unmatched": ",".join(unknown)[:200]}
    probs = [by_value.get(o, 0.0) for o in options]
    total = sum(probs)
    if abs(total - 1.0) > SUM_TOLERANCE:
        return None, {"hosted_sum": total}
    diag: dict[str, Any] = {"hosted_confidence": float(raw.get("confidence") or 0.0)}
    if str(raw.get("choice")) not in options:
        diag["hosted_choice_unmatched"] = str(raw.get("choice"))[:100]
    return probs, diag

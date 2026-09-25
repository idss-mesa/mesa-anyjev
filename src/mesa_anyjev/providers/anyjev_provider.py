"""AnyJev as a decision provider (DESIGN D3, D11; amendments A1, A4, A5).

Wraps one ``anyjev.Decider`` behind a lock (the Decider is stateful and not thread-safe) and
turns each ``Decision`` into a ``DecisionRecord`` with an honest level:

* the level is resolved here per *exact* question key (L2 iff a head exists for the key and
  the backend has hidden states, L1 iff a temperature artifact exists, else L0) and passed to
  the Decider explicitly, because ``Decider.route()`` would serve any yes/no question from any
  fitted yes/no head;
* a decision served through routing anyway is stored at L0 with ``routed_refused``;
* an L1 decision whose artifact froze no prior is stored at L0 with ``l1_prior_missing``;
* a choice with more options than the backend can read raises ``CapabilityError`` so the
  pipeline asks the noul twin;
* labels missing from the gateway's top-k are counted per batch (``last_missing_labels``).
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
from anyjev import Decider, Question
from anyjev.readout import build_prompt, render_chat_parts, resolve_labels
from anyjev.state import render_state

from mesa_anyjev.backends.factory import capabilities_of
from mesa_anyjev.config import DeciderConfig
from mesa_anyjev.providers.base import (
    BackendCapabilities,
    CapabilityError,
    DecisionRecord,
    Level,
    LevelUnavailable,
    rank_probs,
)
from mesa_anyjev.states import state_sha256

DIAG_KEYS = (
    "answer_mass",
    "prior_method",
    "prior_strength",
    "order_flip_raw",
    "order_flip_l0",
    "permutations",
    "shifts_used",
    "temperature",
    "n_calib",
    "routed_from",
    "adapted",
    "adapt_n",
    "blocks_executed",
    "exit_layer",
    "readout",
)


def _scalar(value: Any) -> float | int | str | bool | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value)
    return str(value)


class AnyJevProvider:
    name = "anyjev"

    def __init__(
        self,
        backend: Any,
        cfg: DeciderConfig,
        *,
        served_model: str | None = None,
        capabilities: BackendCapabilities | None = None,
        method: str = "anyjev",
    ) -> None:
        self.backend = backend
        self.cfg = cfg
        self.model: str = str(backend.name)
        self.served_model = served_model or getattr(backend, "served_model", None)
        self.capabilities = capabilities or capabilities_of(backend)
        self.method = method
        self.decider = Decider(
            backend,
            level="L0",
            prior=cfg.prior,
            adaptive_shifts=cfg.adaptive_shifts,
            max_permutations=cfg.max_permutations,
        )
        self.lock = threading.Lock()
        self.last_missing_labels = 0

    # -- levels ------------------------------------------------------------------------------
    def resolve_level(self, question: Question, requested: str | None = None) -> str:
        want = requested or self.cfg.level
        if want == "auto":
            if question.key in self.decider._heads and self.capabilities.hidden_states:
                return "L2"
            if question.key in self.decider._artifacts:
                return "L1"
            return "L0"
        if want == "L2" and not (
            self.capabilities.hidden_states and question.key in self.decider._heads
        ):
            raise LevelUnavailable(
                f"L2 needs a fitted head and local hidden states (backend {self.model})"
            )
        if want == "L1" and question.key not in self.decider._artifacts:
            raise LevelUnavailable(f"no L1 artifact for {question.id}")
        return want

    def supports_level(self, question: Question, level: str) -> bool:
        try:
            self.resolve_level(question, level)
        except LevelUnavailable:
            return False
        return True

    def reset_running_prior(self) -> None:
        """Make a run a function of (card, artifacts), not of what the process saw before (A6)."""
        self.decider._running.clear()

    # -- deciding ------------------------------------------------------------------------------
    def decide_batch(
        self, states: Sequence[dict[str, Any]], question: Question, *, level: str | None = None
    ) -> list[DecisionRecord]:
        if question.kind == "choice" and question.k > self.capabilities.max_choice_k:
            raise CapabilityError(
                f"{question.id}: K={question.k} exceeds max_choice_k={self.capabilities.max_choice_k}; ask the twin"
            )
        if not states:
            return []
        lv = self.resolve_level(question, level)
        with self.lock:
            before = int(getattr(self.backend, "missing_label_events", 0))
            decisions = self.decider.decide_batch(list(states), question, level=lv)
            self.last_missing_labels = (
                int(getattr(self.backend, "missing_label_events", 0)) - before
            )
        return [
            self._record(question, state, d) for state, d in zip(states, decisions, strict=True)
        ]

    def _prompt_sha(self, question: Question, state: dict[str, Any]) -> str:
        labels, _ = resolve_labels(self.backend.tokenizer, question)
        spec = build_prompt(
            render_state(state), question, list(range(question.k)), self.decider.system, labels
        )
        pre, suf = render_chat_parts(self.backend.tokenizer, spec)
        return hashlib.sha256((pre + suf).encode("utf-8")).hexdigest()

    def _record(self, question: Question, state: dict[str, Any], d: Any) -> DecisionRecord:
        probs = [float(p) for p in np.asarray(d.probs, dtype=float).tolist()]
        level: Level = d.level
        diag: dict[str, float | int | str | bool | None] = {
            k: _scalar(d.diagnostics.get(k)) for k in DIAG_KEYS if k in d.diagnostics
        }
        if d.diagnostics.get("routed_from"):
            level = "L0"
            diag["routed_refused"] = True
        prior_method = str(d.diagnostics.get("prior_method", ""))
        if level == "L1" and not prior_method.startswith("frozen:"):
            level = "L0"
            diag["l1_prior_missing"] = True
        idx, conf, margin = rank_probs(probs)
        return DecisionRecord(
            question_id=question.id,
            question_key=question.key,
            kind=question.kind,
            options=list(question.options),
            state=state,
            state_sha256=state_sha256(state),
            prompt_sha256=self._prompt_sha(question, state),
            provider=self.name,
            method=self.method,  # type: ignore[arg-type]
            model=self.model,
            served_model=self.served_model,
            level=level,
            calibration="anyjev",
            probs=probs,
            answer_index=idx,
            answer=question.options[idx],
            confidence=conf,
            p_true=probs[0] if question.kind == "noul" else None,
            margin=margin,
            score_value=float(d.value) if question.kind == "score" else None,
            diagnostics=diag,
        )

    # -- learning hooks (M2/M3) --------------------------------------------------------------
    def calibrate(
        self, question: Question, states: Sequence[Any], labels: Sequence[int]
    ) -> dict[str, Any]:
        with self.lock:
            out: dict[str, Any] = self.decider.calibrate(question, list(states), list(labels))
            return out

    def export(self) -> dict[str, Any]:
        out: dict[str, Any] = self.decider.export_artifacts(include_observations=True)
        return out

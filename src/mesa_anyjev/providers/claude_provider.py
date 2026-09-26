"""Claude as a *recorded second opinion* (M5). The Messages API exposes no logits, so Claude can
never be an AnyJev backend: it answers the same rendered ``State`` / ``Question`` / ``Options``
text through structured outputs (``client.messages.parse(output_format=...)``, adaptive
thinking, no forced ``tool_choice``), and every record is ``provider='claude'``,
``method='claude:structured_output'``, ``level='none'``, ``calibration='none'``, ``probs=None``.
A record with no AnyJev level never auto-writes (policy: ``auto`` needs an AnyJev calibration);
what it can do is agree or disagree with a proposed winner, which the pipeline records on the
group and turns a disagreement into ``escalated`` (a human decides)."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import Any, Literal

from anyjev import Question
from anyjev.state import render_state
from pydantic import BaseModel, ConfigDict, Field, create_model

from mesa_anyjev.config import ClaudeConfig
from mesa_anyjev.providers.base import BackendCapabilities, DecisionRecord, Method
from mesa_anyjev.states import state_sha256

logger = logging.getLogger(__name__)

PROVIDER = "claude"
METHOD: Method = "claude:structured_output"
SYSTEM = (
    "You are a decision function. You will be given a state and a question with a closed set "
    "of options. Pick exactly one option. Do not invent options, do not explain, do not "
    "hedge: answer with the option text as given."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def render_prompt(state: dict[str, Any], question: Question) -> str:
    """The same shape AnyJev prefills (state, question, lettered options) as plain text."""
    lines = [f"State:\n{render_state(state)}", "", f"Question: {question.text}", "Options:"]
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(question.options)]
    lines.append("Answer with the option text only.")
    return "\n".join(lines)


def answer_model(question: Question) -> type[BaseModel]:
    options = tuple(str(o) for o in question.options)
    literal = Literal[options]  # type: ignore[valid-type]
    return create_model(
        "Answer",
        __config__=ConfigDict(extra="forbid"),
        answer=(literal, Field(description="one of the options, verbatim")),
    )


class ClaudeStructuredProvider:
    def __init__(self, cfg: ClaudeConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self.served_model: str | None = None
        self.capabilities = BackendCapabilities(
            max_choice_k=26, hidden_states=False, logprob_top_k=None
        )
        self.last_missing_labels = 0
        self.requests = 0
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # the `claude` extra; credentials resolve through the SDK

            self._client = anthropic.Anthropic()
        return self._client

    def resolve_level(self, question: Question, level: str | None) -> str:
        return "none"

    def decide_batch(
        self, states: Sequence[dict[str, Any]], question: Question, *, level: str | None = None
    ) -> list[DecisionRecord]:
        return [self.decide(state, question) for state in states]

    def decide(self, state: dict[str, Any], question: Question) -> DecisionRecord:
        prompt = render_prompt(state, question)
        sha = hashlib.sha256((SYSTEM + prompt).encode("utf-8")).hexdigest()
        options = [str(o) for o in question.options]
        diag: dict[str, Any] = {"prompt_sha256": sha}
        answer_index = -1
        answer = ""
        try:
            self.requests += 1
            response = self._get_client().messages.parse(
                model=self.model,
                max_tokens=max(256, int(self.cfg.max_tokens)),
                system=SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"effort": self.cfg.effort},
                messages=[{"role": "user", "content": prompt}],
                output_format=answer_model(question),
            )
            u = getattr(response, "usage", None)
            diag["input_tokens"] = getattr(u, "input_tokens", None)
            diag["output_tokens"] = getattr(u, "output_tokens", None)
            if getattr(response, "stop_reason", None) == "refusal":
                diag["refusal"] = True
            else:
                parsed = getattr(response, "parsed_output", None)
                text = str(getattr(parsed, "answer", "") or "")
                if text in options:
                    answer_index = options.index(text)
                    answer = text
                else:
                    diag["unparsed"] = text[:100]
        except Exception as exc:
            logger.warning("claude provider failed (%s: %s)", type(exc).__name__, exc)
            diag["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        return DecisionRecord(
            question_id=question.id,
            question_key=question.key,
            kind=question.kind,
            options=options,
            state=state,
            state_sha256=state_sha256(state),
            provider=PROVIDER,
            method=METHOD,
            model=self.model,
            served_model=None,
            level="none",
            calibration="none",
            probs=None,
            answer_index=answer_index,
            answer=answer,
            confidence=None,
            p_true=None,
            margin=None,
            diagnostics=diag,
        )

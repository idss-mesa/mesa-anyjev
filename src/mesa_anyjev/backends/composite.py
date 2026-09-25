"""Gateway logprobs plus local hidden states under one name (DESIGN D11; M3).

``Decider(level="auto")`` serves L0/L1 from ``next_token_logprobs`` and L2 from
``hidden_states``; one backend object must expose both. The composite routes the logprob
side to the gateway (carc-fast) and the hidden-state side to local weights (Qwen/Qwen3-8B on
the GB10). Both sides must tokenise the answer labels identically, which the constructor
asserts; the two checkpoints are otherwise *not* assumed interchangeable (D5), so the name
under which artifacts are keyed is the local model's and ``served_model`` records the gateway.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mesa_anyjev.providers.base import BackendCapabilities

LABELS = [*[chr(ord("A") + i) for i in range(26)], "Yes", "No", *[str(i) for i in range(1, 10)]]


class CompositeBackend:
    def __init__(self, logprob_backend: Any, hidden_backend: Any, *, max_choice_k: int = 8) -> None:
        self.logprob = logprob_backend
        self.hidden = hidden_backend
        self.tokenizer = hidden_backend.tokenizer
        self.name = str(hidden_backend.name)
        self.served_model = getattr(
            logprob_backend, "served_model", getattr(logprob_backend, "name", None)
        )
        self.n_layers = int(hidden_backend.n_layers)
        self.hidden_size = int(getattr(hidden_backend, "hidden_size", 0))
        self.max_choice_k = int(max_choice_k)
        self.capabilities = BackendCapabilities(
            max_choice_k=max_choice_k,
            hidden_states=True,
            logprob_top_k=getattr(logprob_backend, "logprobs", None),
        )
        mismatch = [
            lab
            for lab in LABELS
            if self._ids(logprob_backend.tokenizer, lab) != self._ids(hidden_backend.tokenizer, lab)
        ]
        if mismatch:
            raise ValueError(
                f"the two tokenizers disagree on label tokens {mismatch}; a composite backend needs identical label ids"
            )

    @staticmethod
    def _ids(tok: Any, label: str) -> tuple[int, ...]:
        return tuple(tok.encode(label, add_special_tokens=False))

    @property
    def missing_label_events(self) -> int:
        return int(getattr(self.logprob, "missing_label_events", 0))

    @property
    def requests(self) -> int:
        return int(getattr(self.logprob, "requests", 0)) + int(
            getattr(self.hidden, "prompts_seen", 0)
        )

    # -- logprob side -----------------------------------------------------------------------------
    def next_token_logprobs(
        self, prompts: Sequence[str], token_ids: Sequence[Sequence[int]]
    ) -> list[np.ndarray]:
        out: list[np.ndarray] = self.logprob.next_token_logprobs(prompts, token_ids)
        return out

    # -- hidden-state side ----------------------------------------------------------------------------
    def hidden_states(
        self,
        prompts: Sequence[str],
        layers: Sequence[int] | None = None,
        token_ids: Any = None,
        positions: Any = None,
    ) -> Any:
        return self.hidden.hidden_states(prompts, layers, token_ids, positions)

    def hidden_states_to(
        self,
        prompts: Sequence[str],
        layers: Sequence[int],
        token_ids: Any = None,
        positions: Any = None,
        **kw: Any,
    ) -> Any:
        if hasattr(self.hidden, "hidden_states_to"):
            return self.hidden.hidden_states_to(prompts, layers, token_ids, positions, **kw)
        raise NotImplementedError("hidden backend has no block loop")

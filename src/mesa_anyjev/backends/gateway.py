"""The CARC LiteLLM -> vLLM gateway as an AnyJev backend (DESIGN D11; RESEARCH.md).

Composition rather than a subclass of ``anyjev.backends.vllm.VLLMBackend``: the upstream
constructor loads the tokenizer without a revision and its request body carries
``allowed_token_ids``, which this gateway silently drops. This class therefore

* loads the tokenizer from the *served* checkpoint at the pinned revision (label token ids
  must match what vLLM samples over),
* sends ``logprobs=20`` on every request regardless of K (top-20 over the full vocabulary is
  what comes back; 50 is a 400) and never ``allowed_token_ids``,
* floors a label missing from the top-20 at -30 like upstream and *counts* it
  (``missing_label_events``): the L0 log-mean over cyclic shifts erases such an option, so the
  policy caps the batch at rule/abstain,
* defines no ``hidden_states``/``n_layers``: L2 through the gateway is refused honestly
  because ``hasattr(backend, "hidden_states")`` is False,
* talks only to loopback with ``trust_env=False`` (mesa-nmdid ``GatewayClient`` pattern) and
  trips a circuit breaker after repeated failures.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np

MISSING_LOGPROB = -30.0
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class GatewayError(RuntimeError):
    pass


def assert_loopback(base_url: str) -> str:
    u = urlparse(base_url)
    if u.scheme != "http" or (u.hostname or "") not in LOOPBACK_HOSTS:
        raise GatewayError(
            f"the gateway must be reached over loopback, got {base_url!r} (RESEARCH.md)"
        )
    if u.path.rstrip("/").endswith("/v1"):
        raise GatewayError(
            "gateway_base_url must not end in /v1 (the backend appends /v1/completions)"
        )
    return base_url.rstrip("/")


@dataclass
class CircuitBreaker:
    failures: int = 3
    open_for: float = 60.0
    _count: int = 0
    _opened_at: float | None = None

    def check(self) -> None:
        if self._opened_at is not None:
            if time.monotonic() - self._opened_at < self.open_for:
                raise GatewayError("gateway circuit breaker is open")
            self._opened_at = None
            self._count = 0

    def success(self) -> None:
        self._count = 0

    def failure(self) -> None:
        self._count += 1
        if self._count >= self.failures:
            self._opened_at = time.monotonic()

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None and time.monotonic() - self._opened_at < self.open_for


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    liveness: bool
    logprobs_ok: bool
    labels_present: bool
    top_k: int
    latency_ms: int
    error: str | None = None


class GatewayBackend:
    """``anyjev.backends.base.Backend`` over ``POST {base_url}/v1/completions``."""

    def __init__(
        self,
        base_url: str,
        served_model: str,
        tokenizer: str,
        tokenizer_revision: str | None,
        canonical_model: str,
        api_key: str | None,
        *,
        logprobs: int = 20,
        max_choice_k: int = 8,
        workers: int = 8,
        timeout: float = 120.0,
        retries: int = 3,
        breaker: CircuitBreaker | None = None,
        tokenizer_obj: Any | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = assert_loopback(base_url)
        self.served_model = served_model
        self.name = canonical_model
        self.tokenizer_name = tokenizer
        self.tokenizer_revision = tokenizer_revision
        self.logprobs = int(logprobs)
        self.max_choice_k = int(max_choice_k)
        self.workers = int(workers)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.breaker = breaker or CircuitBreaker()
        self.missing_label_events = 0
        self.requests = 0
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout, connect=5.0),
            trust_env=False,
        )
        if tokenizer_obj is not None:
            self.tokenizer = tokenizer_obj
        else:
            from transformers import AutoTokenizer  # the `gateway` extra

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=tokenizer_revision)

    # -- the Backend protocol -----------------------------------------------------------------
    def next_token_logprobs(
        self, prompts: Sequence[str], token_ids: Sequence[Sequence[int]]
    ) -> list[np.ndarray]:
        import concurrent.futures as cf

        with cf.ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, prompts, token_ids))

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        self.breaker.check()
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self._client.post("/v1/completions", json=body)
                if resp.status_code == 400:
                    self.breaker.failure()
                    raise GatewayError(f"gateway rejected the request: {resp.text[:300]}")
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                resp.raise_for_status()
                self.breaker.success()
                self.requests += 1
                data: dict[str, Any] = resp.json()
                return data
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last = exc
                self.breaker.failure()
                time.sleep(0.5 * (2**attempt))
        raise GatewayError(f"gateway unreachable after {self.retries} attempts: {last}")

    def _one(self, prompt: str, ids: Sequence[int]) -> np.ndarray:
        body = {
            "model": self.served_model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": self.logprobs,
        }
        out = self._post(body)
        top: dict[str, float] = out["choices"][0]["logprobs"]["top_logprobs"][0]
        by_id: dict[int, float] = {}
        for tid in ids:
            for key in (self.tokenizer.decode([tid]), self.tokenizer.convert_ids_to_tokens(tid)):
                if key in top:
                    by_id[tid] = float(top[key])
                    break
        missing = sum(1 for tid in ids if tid not in by_id)
        self.missing_label_events += missing
        return np.array([by_id.get(tid, MISSING_LOGPROB) for tid in ids], dtype=np.float64)

    # -- health ---------------------------------------------------------------------------------
    def probe(self) -> ProbeResult:
        """Liveness, one two-option completion, and whether both labels came back in the top-k."""
        started = time.monotonic()
        try:
            live = self._client.get("/health/liveliness")
            liveness = live.status_code == 200
        except httpx.TransportError as exc:
            return ProbeResult(False, False, False, False, 0, 0, f"liveness: {exc}")
        # Render exactly what the Decider sends: the AnyJev readout through the chat template.
        from anyjev import Question
        from anyjev.readout import build_prompt, render_chat_parts, resolve_labels

        q = Question.noul("Is the sky blue?", name="probe")
        labels, ids = resolve_labels(self.tokenizer, q)
        pre, suf = render_chat_parts(
            self.tokenizer, build_prompt("The sky is blue today.", q, [0, 1], labels=labels)
        )
        prompt = pre + suf
        try:
            before = self.missing_label_events
            lp = self._one(prompt, ids)
            missing = self.missing_label_events - before
            latency = int((time.monotonic() - started) * 1000)
            return ProbeResult(
                ok=liveness and missing == 0,
                liveness=liveness,
                logprobs_ok=bool(np.isfinite(lp).all()),
                labels_present=missing == 0,
                top_k=self.logprobs,
                latency_ms=latency,
            )
        except GatewayError as exc:
            return ProbeResult(
                False,
                liveness,
                False,
                False,
                self.logprobs,
                int((time.monotonic() - started) * 1000),
                str(exc),
            )

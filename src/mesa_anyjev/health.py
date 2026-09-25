"""``mesa-anyjev doctor``: what this host can reach, honestly (RESEARCH.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mesa_anyjev import questions
from mesa_anyjev.config import Config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class HealthReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name, ok, detail))

    def lines(self) -> list[str]:
        return [
            f"[{'ok' if c.ok else 'FAIL'}] {c.name}{': ' + c.detail if c.detail else ''}"
            for c in self.checks
        ]


def doctor(cfg: Config, *, backend_kind: str | None = None) -> HealthReport:
    rep = HealthReport()
    drift = questions.lock_drift()
    rep.add(
        "questions lock", not drift, questions.lock_sha()[:12] if not drift else "; ".join(drift)
    )
    try:
        from mesa_anyjev.provenance.store import open_store

        store = open_store(cfg.provenance.dsn)
        store.close()
        rep.add("provenance store", True, cfg.provenance.dsn)
    except Exception as exc:
        rep.add("provenance store", False, f"{cfg.provenance.dsn}: {exc}")
    if cfg.eval_root:
        rep.add("eval root", Path(cfg.eval_root).expanduser().is_dir(), cfg.eval_root)
    kind = backend_kind or cfg.backend.kind
    rep.add("backend", True, f"kind={kind} planner={cfg.planner.kind} profile={cfg.policy.profile}")
    if kind == "gateway":
        _gateway_checks(cfg, rep)
    elif kind == "hf":
        rep.add("hf backend", False, "local weights arrive with milestone M3")
    return rep


def _gateway_checks(cfg: Config, rep: HealthReport) -> None:
    import httpx

    from mesa_anyjev.backends.gateway import GatewayBackend, GatewayError, assert_loopback

    try:
        base = assert_loopback(cfg.backend.gateway_base_url)
    except GatewayError as exc:
        rep.add("gateway url", False, str(exc))
        return
    try:
        live = httpx.get(f"{base}/health/liveliness", timeout=5.0, trust_env=False)
        rep.add("gateway liveness", live.status_code == 200, f"{base} -> {live.status_code}")
    except httpx.TransportError as exc:
        rep.add("gateway liveness", False, f"{base}: {exc} (is the tunnel up?)")
        return
    if not cfg.backend.gateway_api_key:
        rep.add(
            "gateway key",
            False,
            "MESA_ANYJEV_BACKEND__GATEWAY_API_KEY or MESA_LLM_API_KEY is unset",
        )
        return
    try:
        backend = GatewayBackend(
            base,
            cfg.backend.served_model,
            cfg.backend.tokenizer,
            cfg.backend.tokenizer_revision,
            cfg.backend.canonical_model,
            cfg.backend.gateway_api_key,
            logprobs=cfg.backend.logprobs,
            max_choice_k=cfg.backend.max_choice_k,
            workers=1,
            timeout=cfg.backend.timeout,
        )
    except Exception as exc:
        rep.add(
            "tokenizer", False, f"{cfg.backend.tokenizer}@{cfg.backend.tokenizer_revision}: {exc}"
        )
        return
    rep.add(
        "tokenizer",
        True,
        f"{cfg.backend.tokenizer}@{(cfg.backend.tokenizer_revision or 'main')[:12]}",
    )
    _label_checks(backend.tokenizer, rep)
    probe = backend.probe()
    rep.add(
        "gateway logprobs",
        probe.ok,
        f"top_k={probe.top_k} labels_present={probe.labels_present} {probe.latency_ms} ms"
        + (f" error={probe.error}" if probe.error else ""),
    )
    # vLLM's --max-logprobs cap: 26 must be rejected with a 400 (RESEARCH.md)
    try:
        backend._post(
            {
                "model": cfg.backend.served_model,
                "prompt": "Answer Yes or No.",
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 26,
            }
        )
        rep.add(
            "logprobs cap",
            False,
            "logprobs=26 was accepted; the doctor expected the vLLM default cap of 20; re-verify max_choice_k",
        )
    except GatewayError as exc:
        rep.add(
            "logprobs cap",
            "greater than max allowed" in str(exc) or "400" in str(exc),
            "26 -> 400 (cap 20)",
        )


def _label_checks(tokenizer: Any, rep: HealthReport) -> None:
    from anyjev.readout import LabelTokenError, map_label_tokens

    letters = [chr(ord("A") + i) for i in range(26)]
    try:
        map_label_tokens(tokenizer, letters)
        map_label_tokens(tokenizer, ["Yes", "No"])
        map_label_tokens(tokenizer, [str(i) for i in range(1, 10)])
        rep.add(
            "label tokens",
            True,
            "A..Z, Yes/No and 1..9 are single tokens (score falls back to letters above 9)",
        )
    except LabelTokenError as exc:
        rep.add("label tokens", False, str(exc))

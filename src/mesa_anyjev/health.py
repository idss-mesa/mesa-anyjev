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
    elif kind in ("hf", "composite"):
        _hf_checks(cfg, rep)
    _artifact_checks(cfg, rep, kind)
    return rep


def _hf_checks(cfg: Config, rep: HealthReport) -> None:
    try:
        import torch
    except ImportError:
        rep.add("torch", False, "install the `hf` extra (torch, transformers, accelerate)")
        return
    rep.add(
        "torch",
        True,
        f"{torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}",
    )
    if not torch.cuda.is_available() and cfg.backend.hf_device.startswith("cuda"):
        rep.add("cuda", False, "no CUDA device but backend.hf_device is cuda")
        return
    try:
        from anyjev.backends.hf import HFBackend

        from mesa_anyjev.backends.factory import prepare_torch

        prepare_torch(cfg.backend)
        backend = HFBackend(
            cfg.backend.hf_model,
            device=cfg.backend.hf_device,
            dtype=cfg.backend.hf_dtype,
            batch_size=cfg.backend.hf_batch_size,
        )
    except Exception as exc:
        rep.add(
            "hf model", False, f"{cfg.backend.hf_model}: {type(exc).__name__}: {str(exc)[:200]}"
        )
        return
    mem = round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0.0
    rep.add(
        "hf model",
        True,
        f"{cfg.backend.hf_model} n_layers={backend.n_layers} hidden_size={backend.hidden_size} mem={mem} GB",
    )
    _label_checks(backend.tokenizer, rep)
    from anyjev import Decider, Question

    dec = Decider(backend, level="raw")
    q = Question.noul("Is the sky blue?", name="probe")
    decs = dec.decide_batch(["The sky is blue today.", "It is raining and grey."], q, level="raw")
    mass = min(float(d.diagnostics.get("answer_mass", 0.0)) for d in decs)
    rep.add(
        "answer mass",
        mass > 0.5,
        f"min answer_mass {mass:.3f} on two probe states (label tokens carry the answer)",
    )
    rep.add("early stop", hasattr(backend, "hidden_states_to"), "block loop available for L2 heads")


def _artifact_checks(cfg: Config, rep: HealthReport, kind: str) -> None:
    from mesa_anyjev.artifacts import ArtifactStore

    model = (
        cfg.backend.canonical_model
        if kind == "gateway"
        else (cfg.backend.hf_model if kind in ("hf", "composite") else "fake")
    )
    store = ArtifactStore(
        cfg.artifacts.dir, model, questions.lock_sha(), strict=cfg.artifacts.strict
    )
    current = store.current()
    if current is None:
        rep.add(
            "artifacts", True, f"no promoted bundle for {model} under {store.dir} (serving at L0)"
        )
        return
    m = current.manifest
    fitted = m.get("fitted_on", {})
    ok = not cfg.artifacts.strict or fitted.get("backend_kind") in (None, kind)
    rep.add(
        "artifacts",
        ok,
        f"v{current.version} questions={list(m.get('per_question', {}))} fitted_on={fitted}",
    )


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

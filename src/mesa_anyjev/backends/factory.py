"""``make_backend(cfg)`` with lazy imports so CI never imports torch or transformers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from anyjev.backends.fake import FakeBackend

from mesa_anyjev.config import BackendConfig
from mesa_anyjev.providers.base import BackendCapabilities

ContentFn = Callable[[str, str], float]

_WORD = re.compile(r"[a-z0-9]+")


def _hash_unit(text: str) -> float:
    return (
        int(hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:8], 16) % 1000
    ) / 1000.0


def default_fake_content(state: str, option: str) -> float:
    """A deterministic, mildly sensible content function for the fake backend.

    Yes/No (term.fits, avu.keep, column.annotate, ontology_fits): rewards word overlap between
    the candidate/term block and the column/site text in the rendered JSON state. Choices:
    rewards an option whose leading word appears in the state. A hashed jitter keeps ties and
    order effects visible so L0 has something to remove.
    """
    s = state.lower()
    jitter = (_hash_unit(state + "|" + option) - 0.5) * 0.4
    if option in ("Yes", "No"):
        cand = re.search(r'"candidate":\s*\{(.*?)\}', s, re.S) or re.search(
            r'"term":\s*\{(.*?)\}', s, re.S
        )
        target = re.search(r'"column":\s*\{(.*?)\}', s, re.S) or re.search(
            r'"site":\s*\{(.*?)\}', s, re.S
        )
        if cand and target:
            cw = set(_WORD.findall(cand.group(1))) - {
                "label",
                "curie",
                "ontology",
                "id",
                "description",
                "synonyms",
                "has",
                "children",
                "true",
                "false",
            }
            tw = set(_WORD.findall(target.group(1)))
            overlap = len(cw & tw)
            score = 1.5 if overlap else -1.0
        else:
            score = 0.5 if '"identifier"' not in s else -1.5
        return (score if option == "Yes" else -score) + jitter
    head = option.split(":")[0].split(" ")[0].lower()
    return (1.0 if head and head in s else 0.0) + jitter


def capabilities_of(backend: Any) -> BackendCapabilities:
    if hasattr(backend, "capabilities"):
        caps: BackendCapabilities = backend.capabilities
        return caps
    hidden = hasattr(backend, "hidden_states") or hasattr(backend, "hidden_states_to")
    max_k = int(getattr(backend, "max_choice_k", 26))
    return BackendCapabilities(max_choice_k=max_k, hidden_states=hidden, logprob_top_k=None)


def count_prompts(backend: Any) -> Any:
    """Give a backend without a request counter a ``prompts_seen`` total over every prefill
    (logprobs, hidden states, block-loop hidden states), so runs and bench cells can report
    prompts for local weights the way they do for the gateway (``requests``)."""
    if hasattr(backend, "prompts_seen"):
        return backend
    backend.prompts_seen = 0

    def _wrap(name: str) -> None:
        orig = getattr(backend, name, None)
        if orig is None:
            return

        def counted(prompts: Any, *args: Any, **kwargs: Any) -> Any:
            backend.prompts_seen += len(prompts)
            return orig(prompts, *args, **kwargs)

        setattr(backend, name, counted)

    for name in ("next_token_logprobs", "hidden_states", "hidden_states_to"):
        _wrap(name)
    return backend


def prepare_torch(cfg: BackendConfig) -> None:
    """Apply process-wide torch settings before local weights load (idempotent)."""
    if cfg.hf_native_triton:
        return
    try:
        from torch._native import registry as native_registry
    except ImportError:  # older torch has no native-op overrides
        return
    native_registry.deregister_op_overrides(disable_dsl_names="triton")


def make_backend(cfg: BackendConfig, *, fake_content: ContentFn | None = None) -> Any:
    if cfg.kind == "fake":
        return FakeBackend(
            content=fake_content or default_fake_content,
            position_bias=[0.6, 0.2, 0.0, -0.1],
            temperature=0.7,
        )
    if cfg.kind == "gateway":
        from mesa_anyjev.backends.gateway import GatewayBackend

        backend = GatewayBackend(
            cfg.gateway_base_url,
            cfg.served_model,
            cfg.tokenizer,
            cfg.tokenizer_revision,
            cfg.canonical_model,
            cfg.gateway_api_key,
            logprobs=cfg.logprobs,
            max_choice_k=cfg.max_choice_k,
            workers=cfg.workers,
            timeout=cfg.timeout,
        )
        backend.capabilities = BackendCapabilities(cfg.max_choice_k, False, cfg.logprobs)  # type: ignore[attr-defined]
        return backend
    if cfg.kind == "hf":
        from anyjev.backends.hf import HFBackend  # the `hf` extra

        prepare_torch(cfg)

        return count_prompts(
            HFBackend(
                cfg.hf_model,
                device=cfg.hf_device,
                dtype=cfg.hf_dtype,
                batch_size=cfg.hf_batch_size,
            )
        )
    if cfg.kind == "composite":
        from anyjev.backends.hf import HFBackend  # the `hf` extra

        prepare_torch(cfg)

        from mesa_anyjev.backends.composite import CompositeBackend
        from mesa_anyjev.backends.gateway import GatewayBackend

        gateway = GatewayBackend(
            cfg.gateway_base_url,
            cfg.served_model,
            cfg.tokenizer,
            cfg.tokenizer_revision,
            cfg.hf_model,
            cfg.gateway_api_key,
            logprobs=cfg.logprobs,
            max_choice_k=cfg.max_choice_k,
            workers=cfg.workers,
            timeout=cfg.timeout,
        )
        local = count_prompts(
            HFBackend(
                cfg.hf_model,
                device=cfg.hf_device,
                dtype=cfg.hf_dtype,
                batch_size=cfg.hf_batch_size,
            )
        )
        return CompositeBackend(gateway, local, max_choice_k=cfg.max_choice_k)
    raise ValueError(f"unknown backend kind {cfg.kind!r}")

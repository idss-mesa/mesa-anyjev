"""Configuration model and loader.

Precedence (highest wins): explicit flag > environment variable > YAML file > default.
Environment variables use the ``MESA_ANYJEV_`` prefix and ``__`` to descend into a section:
``MESA_ANYJEV_BACKEND__KIND=gateway`` maps to ``config.backend.kind``. Two mesa-mcp names are
honoured as fallbacks so an existing ``mesa-mcp/.env`` keeps working: ``MESA_LLM_BASE_URL``
(a trailing ``/v1`` is stripped, the gateway backend appends its own paths) and
``MESA_LLM_API_KEY``. Shape is validated here; whether a host answers is the doctor's job.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

ENV_PREFIX = "MESA_ANYJEV_"
ENV_DELIM = "__"

BackendKind = Literal["fake", "gateway", "hf", "composite"]
PlannerKind = Literal["gateway", "claude", "static"]
Level = Literal["raw", "L0", "L1", "L2", "auto"]
Profile = Literal["prod", "dev"]
HostedMode = Literal["off", "allowlist"]
FixtureMode = Literal["off", "record", "replay", "auto"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackendConfig(_Section):
    """The AnyJev backend that reads next-token logprobs (and, locally, hidden states)."""

    kind: BackendKind = "fake"
    # Loopback only; the CARC gateway is reached through an SSH tunnel (RESEARCH.md).
    gateway_base_url: str = "http://127.0.0.1:8000"
    gateway_api_key: str | None = None
    served_model: str = "carc-fast"
    # The served checkpoint; label token ids must match it exactly (DESIGN D5).
    tokenizer: str = "RedHatAI/Qwen3-8B-NVFP4"
    tokenizer_revision: str | None = "e391349c110709b87bfc2ad2fde3f50dc5839fd8"
    # Artifact slug. Stays the served repo id until tokenizer parity is recorded (D5).
    canonical_model: str = "RedHatAI/Qwen3-8B-NVFP4"
    hf_model: str = "Qwen/Qwen3-8B"
    hf_device: str = "cuda"
    hf_dtype: str = "bfloat16"
    hf_batch_size: int = 16
    hf_native_triton: bool = False
    """torch>=2.14 routes some eager aten ops through Triton kernels that are compiled against
    Python.h at first use; on hosts without the CPython headers that compile fails inside the
    forward pass. Off (default) deregisters the Triton overrides so the stock aten kernels run."""
    # Always 20 on the gateway: allowed_token_ids is dropped and 20 is vLLM's cap (D11).
    logprobs: int = 20
    # Raised only when the doctor measures zero label erasure at a higher K (D11).
    max_choice_k: int = 8
    workers: int = 8
    timeout: float = 120.0


class DeciderConfig(_Section):
    level: Level = "auto"
    prior: Literal["batch", "content_free", "none"] = "batch"
    adaptive_shifts: bool = True
    max_permutations: int | None = None


class PlannerConfig(_Section):
    """The reasoning model: it plans, never decides (DESIGN D2)."""

    kind: PlannerKind = "gateway"
    gateway_model: str = "carc-tools"
    claude_model: str = "claude-opus-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    timeout: float = 600.0


class ClaudeConfig(_Section):
    """The Claude structured-output provider (M5); credentials resolve through the SDK."""

    model: str = "claude-opus-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_tokens: int = 1024
    # M5: ask Claude the term.fits question over a proposed group's top candidates and record
    # the answers as a second opinion (level none); a disagreement escalates to a human.
    second_opinion: bool = False
    second_opinion_top_k: int = 8


class PolicyConfig(_Section):
    profile: Profile = "prod"
    defaults_file: str = "policy_defaults.yaml"
    max_candidates: int = 12
    max_avus: int = 25
    proposal_size: int = 8
    max_wait_s: float = 30.0
    # M5: specificity (a child term replaces its parent when p(child) >= p(parent) + delta)
    # and the planner audit (dataset.ontology_applies per registry entry, recorded only).
    specificity: bool = True
    specificity_delta: float = 0.10
    audit_planner: bool = True
    # Hosted providers send state off this host (DESIGN D16).
    hosted_providers: HostedMode = "off"
    hosted_allow_project_roots: list[str] = Field(default_factory=list)
    hosted_allow_local_sources: bool = True


class ArtifactsConfig(_Section):
    dir: str = "~/.mesa/anyjev/artifacts"
    strict: bool = True


class ProvenanceConfig(_Section):
    dsn: str = "duckdb:///~/.mesa/anyjev/provenance.duckdb"
    parquet_export: bool = False
    push_to_irods: bool = False


class DuckLakeConfig(_Section):
    catalog_dsn: str | None = "duckdb:///~/.mesa/anyjev/ducklake-catalog.duckdb"
    cache_dir: str | None = "~/.mesa/anyjev/ducklake-cache"
    local_project_root: str = "/local/mesa-anyjev"
    local_zone: str = "local"


class OLSConfig(_Section):
    base_url: str = "https://www.ebi.ac.uk/ols4/api/v2"
    fixtures: FixtureMode = "off"
    fixtures_dir: str = "tests/fixtures/ols"


class MotherDuckConfig(_Section):
    """Hosted Jev through MotherDuck ``prompt_jev`` (M4b). The token is read only from the
    environment variable named here, never stored in config."""

    database: str = "mesa_anyjev"
    token_env: str = "MOTHERDUCK_TOKEN"
    region: Literal["us-east-1", "us-west-2"] = "us-east-1"
    batch_size: int = 32
    timeout: float = 600.0


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: BackendConfig = Field(default_factory=BackendConfig)
    decider: DeciderConfig = Field(default_factory=DeciderConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    provenance: ProvenanceConfig = Field(default_factory=ProvenanceConfig)
    ducklake: DuckLakeConfig = Field(default_factory=DuckLakeConfig)
    ols: OLSConfig = Field(default_factory=OLSConfig)
    motherduck: MotherDuckConfig = Field(default_factory=MotherDuckConfig)
    # The neon-avu-eval checkout used for silver labels and the bench. No default path: a
    # personal home directory must not be baked into the package (plan amendment B11).
    eval_root: str | None = None


def _coerce(raw: str) -> Any:
    """Environment values are strings; let YAML decide what they mean (``true``, ``12``)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        rest = key[len(ENV_PREFIX) :]
        if ENV_DELIM in rest:
            section, field = rest.split(ENV_DELIM, 1)
            out.setdefault(section.lower(), {})[field.lower()] = _coerce(value)
        elif rest.lower() in Config.model_fields:
            out[rest.lower()] = _coerce(value)
        # Anything else at the top level (MESA_ANYJEV_ENGINE, _LIVE, _E2E, _MOTHERDUCK,
        # _TEST_PG_*) is a test-tier gate read by conftest, not a setting.
    # mesa-mcp compatibility fallbacks (only when the native name is absent).
    backend = out.setdefault("backend", {})
    if "gateway_base_url" not in backend and env.get("MESA_LLM_BASE_URL"):
        base = env["MESA_LLM_BASE_URL"].rstrip("/")
        backend["gateway_base_url"] = base[:-3] if base.endswith("/v1") else base
    if "gateway_api_key" not in backend and env.get("MESA_LLM_API_KEY"):
        backend["gateway_api_key"] = env["MESA_LLM_API_KEY"]
    if not backend:
        del out["backend"]
    return out


def _deep_merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path | None = None,
    *,
    flag_overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Build a :class:`Config` from YAML, environment and flags (flags win)."""
    data: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: the configuration file must be a mapping")
        known = set(Config.model_fields)
        for section in list(loaded):
            if section not in known:
                logger.warning("config: unknown section %r ignored", section)
                loaded.pop(section)
        data = loaded
    data = _deep_merge(data, _env_overrides(os.environ if env is None else env))
    if flag_overrides:
        data = _deep_merge(data, flag_overrides)
    return Config.model_validate(data)


def config_sha256(cfg: Config) -> str:
    """A stable hash of the effective configuration with secrets removed (stored on runs)."""
    public = cfg.model_dump(mode="json")
    public["backend"]["gateway_api_key"] = None
    payload = json.dumps(public, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_active: Config | None = None


def set_active_config(cfg: Config | None) -> None:
    global _active
    _active = cfg


def get_active_config() -> Config:
    global _active
    if _active is None:
        _active = load_config()
    return _active

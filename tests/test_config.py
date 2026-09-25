from __future__ import annotations

from pathlib import Path

import pytest

from mesa_anyjev.config import Config, config_sha256, load_config


def test_defaults() -> None:
    cfg = load_config(env={})
    assert cfg.backend.kind == "fake"
    assert cfg.backend.logprobs == 20 and cfg.backend.max_choice_k == 8
    assert cfg.planner.kind == "gateway"
    assert cfg.policy.profile == "prod" and cfg.policy.hosted_providers == "off"
    assert cfg.eval_root is None


def test_env_precedence_and_mesa_mcp_fallbacks(tmp_path: Path) -> None:
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text(
        "backend:\n  kind: hf\n  workers: 3\nunknown_section:\n  x: 1\n", encoding="utf-8"
    )
    env = {
        "MESA_ANYJEV_BACKEND__WORKERS": "5",
        "MESA_LLM_BASE_URL": "http://127.0.0.1:18000/v1",
        "MESA_LLM_API_KEY": "sk-test",
        "MESA_ANYJEV_POLICY__PROFILE": "dev",
    }
    cfg = load_config(yaml_file, env=env, flag_overrides={"backend": {"workers": 9}})
    assert cfg.backend.kind == "hf"
    assert cfg.backend.workers == 9  # flag > env > yaml
    assert cfg.backend.gateway_base_url == "http://127.0.0.1:18000"  # /v1 stripped
    assert cfg.backend.gateway_api_key == "sk-test"
    assert cfg.policy.profile == "dev"


def test_native_name_wins_over_fallback() -> None:
    cfg = load_config(
        env={
            "MESA_ANYJEV_BACKEND__GATEWAY_BASE_URL": "http://127.0.0.1:8000",
            "MESA_LLM_BASE_URL": "http://127.0.0.1:18000/v1",
        }
    )
    assert cfg.backend.gateway_base_url == "http://127.0.0.1:8000"


def test_config_sha_excludes_the_key() -> None:
    a = load_config(env={"MESA_LLM_API_KEY": "one"})
    b = load_config(env={"MESA_LLM_API_KEY": "two"})
    assert config_sha256(a) == config_sha256(b)
    assert "one" not in config_sha256(a)


def test_extra_fields_rejected() -> None:
    with pytest.raises(Exception, match="extra"):
        Config.model_validate({"backend": {"bogus": 1}})


def test_gate_variables_are_not_settings() -> None:
    cfg = load_config(
        env={
            "MESA_ANYJEV_ENGINE": "gateway",
            "MESA_ANYJEV_LIVE": "1",
            "MESA_ANYJEV_EVAL_ROOT": "/x",
        }
    )
    assert cfg.eval_root == "/x"

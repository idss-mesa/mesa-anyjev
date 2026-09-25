"""End to end over the real MCP protocol: the ElicitationChooser answers mesa-mcp's own
``term_choice`` form inside ``mesa_avu_apply_term`` (zero mesa-mcp change, plan B13).

Gated like mesa-ducklake's harness: ``MESA_E2E_IRODS_ROOT`` (a collection this user may write
under, already ``mesa_ducklake_init_project``-enabled), iRODS credentials in the environment
(``MESA_MCP_IRODS_USER``/``MESA_MCP_IRODS_PASSWORD`` or ``IRODS_ENVIRONMENT_FILE``), the ``e2e``
extra (``mcp``), ``mesa-mcp`` on PATH, and live EBI OLS. The chooser runs on the fake backend
unless ``MESA_ANYJEV_BACKEND__KIND`` says otherwise."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e
ROOT = os.environ.get("MESA_E2E_IRODS_ROOT")
if not ROOT:
    pytest.skip("MESA_E2E_IRODS_ROOT not set", allow_module_level=True)
pytest.importorskip("mcp")


def _payload(result: object) -> dict[str, object]:
    sc = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if isinstance(sc, dict):
        return sc
    content = getattr(result, "content", None) or []
    text = getattr(content[0], "text", "{}") if content else "{}"
    out: dict[str, object] = json.loads(text)
    return out


def test_chooser_answers_apply_term_over_stdio(tmp_path: Path) -> None:
    from mesa_anyjev.backends.factory import make_backend
    from mesa_anyjev.chooser import ElicitationChooser
    from mesa_anyjev.config import load_config
    from mesa_anyjev.policy import load_policy
    from mesa_anyjev.providers.anyjev_provider import AnyJevProvider
    from tests.e2e.harness import open_mesa_mcp

    cfg = load_config(env=os.environ)
    provider = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    chooser = ElicitationChooser(provider, load_policy())
    catalog = tmp_path / "e2e-catalog.duckdb"
    target = f"{ROOT.rstrip('/')}/mesa-anyjev-e2e-{uuid.uuid4().hex[:8]}.csv"

    async def scenario() -> tuple[dict[str, object], list[dict[str, object]]]:
        async with open_mesa_mcp(
            catalog_dsn=f"duckdb:///{catalog}",
            cache_dir=tmp_path / "cache",
            stderr_log=tmp_path / "mesa-mcp.log",
        ) as (client, broker):
            broker.chooser = chooser
            await client.call_tool("ds_write_file", {"path": target, "content": "a,b\n1,2\n"})
            res = await client.call_tool(
                "mesa_avu_apply_term",
                {"path": target, "ontology_id": "envo", "value": "tropical moist broadleaf forest"},
            )
            out = _payload(res)
            avus = _payload(await client.call_tool("ds_list_avus", {"path": target}))
            await client.call_tool("ds_delete_file", {"path": target})
            return {"apply": out, "avus": avus}, broker.log

    outcome, log = asyncio.run(scenario())
    assert log, "mesa-mcp did not elicit a term choice"
    assert chooser.log and chooser.log[-1]["level"] in ("L0", "L1", "L2")
    chosen = log[-1]["content"]
    if chosen is None:
        assert outcome["apply"].get("error", {}).get("code") == "invalid_argument"  # declined
    else:
        assert "error" not in outcome["apply"]
        assert any(chosen["iri"].rsplit("_", 1)[-1] in json.dumps(a) for a in [outcome["avus"]])

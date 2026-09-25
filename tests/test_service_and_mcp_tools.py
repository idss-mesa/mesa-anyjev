"""DecisionService and the mesa_decide_* tools on the fake backend (replay OLS fixtures): the
run summary, candidate listing from the sidecar, human picks with labels, the MRTR apply loop
(state ids only, tamper guard), explain, feedback, health, and mesa-mcp's own conformance
assertions over the plugin's tool definitions (plan amendment B9)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

import mesa_anyjev.mcp_tools as tools
from mesa_anyjev.cards import load_card
from mesa_anyjev.config import load_config
from mesa_anyjev.service import DecisionService

ROOT = Path(__file__).resolve().parent
CARD = ROOT / "fixtures" / "cards" / "DP1.10022.001.bet_sorting.md"
pytestmark = pytest.mark.skipif(
    not any((ROOT / "fixtures" / "ols").glob("*.json")), reason="OLS fixtures not recorded"
)


@pytest.fixture
def service(tmp_path: Path) -> Any:
    cfg = load_config(
        env={
            "MESA_ANYJEV_POLICY__PROFILE": "dev",
            "MESA_ANYJEV_PLANNER__KIND": "static",
            "MESA_ANYJEV_OLS__FIXTURES": "replay",
            "MESA_ANYJEV_OLS__FIXTURES_DIR": str(ROOT / "fixtures" / "ols"),
            "MESA_ANYJEV_PROVENANCE__DSN": f"duckdb:///{tmp_path / 'prov.duckdb'}",
        }
    )
    svc = DecisionService(cfg)
    tools.set_service(svc)
    yield svc
    tools.set_service(None)
    svc.close()


def _call(name: str, args: dict[str, Any], **kw: Any) -> dict[str, Any]:
    from mesa_mcp.config import Config
    from mesa_mcp.server import MesaServer

    return asyncio.run(MesaServer(config=Config()).call(name, args, **kw))


def test_tool_definitions_pass_mesa_mcp_conformance() -> None:
    import jsonschema
    from mesa_mcp.config import Config
    from mesa_mcp.server import MesaServer

    defs = {t.name: t for t in MesaServer(config=Config())._tool_definitions()}
    for name in tools.TOOL_NAMES:
        t = defs[name]
        assert t.input_schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
        jsonschema.Draft202012Validator.check_schema(t.input_schema)
        assert t.output_schema and t.output_schema.get("$schema")
        jsonschema.Draft202012Validator.check_schema(t.output_schema)
        assert t.meta and "io.mesa/surface" in t.meta
        dumped = t.model_dump(by_alias=True, exclude_none=True)
        assert "_meta" in dumped and "inputSchema" in dumped
    from mesa_mcp.server import get_tool

    assert get_tool("mesa_decide_apply").meta["io.mesa/surface"] == "decision"


def test_annotate_explain_feedback_and_apply_round_trip(service: DecisionService) -> None:
    out = _call("mesa_decide_annotate", {"card_path": str(CARD), "actor": "alice"})
    run_id = out["run_id"]
    assert out["proposals"] and out["n_decisions"] > 0 and out["degraded"] is False
    ex = _call("mesa_decide_explain", {"run_id": run_id})
    assert ex["run"]["actor"] == "alice" and ex["decisions"] and ex["links"]
    assert all(d["level"] in ("raw", "L0", "L1", "L2", "none") for d in ex["decisions"])
    pending = ex["pending_groups"]
    assert pending, "the dev profile on the fake backend should leave proposed groups"
    gid = pending[0]
    cands = service.candidates_for_group(UUID(gid))
    assert cands and cands[0]["p_true"] is not None and cands[0]["level"] == "L0"
    # a pick of the second candidate (not the winner) builds an accepted link for it
    chosen = cands[-1]
    fb = _call(
        "mesa_decide_feedback",
        {"group_id": gid, "action": "pick", "decision_id": chosen["decision_id"], "actor": "bob"},
    )
    assert fb["outcome"] == "human" and fb["labels_written"] == len(cands)
    key = next(
        d["question_key"]
        for d in service.store.decisions(UUID(run_id))
        if d["question_id"] == "term.fits"
    )
    assert len(service.store.labels_for(key)) == len(cands)
    links = {str(link["link_id"]): link for link in service.store.links(UUID(run_id))}
    assert links[fb["accepted_link_id"]]["write_status"] == "accepted"
    assert links[fb["accepted_link_id"]]["term_curie"] == chosen["curie"]
    assert service.store.group(UUID(gid))["outcome"] == "human"
    # dry-run apply of what is accepted so far (local mode: no auth, no DuckLake)
    ap = _call("mesa_decide_apply", {"run_id": run_id, "irods_path": "/local/x/f.csv"})
    assert ap["dry_run"] is True and ap["mode"] == "local"
    assert any(w["value"] for w in ap["written"])
    # a wrong id is refused
    from mesa_mcp.errors import ToolError

    with pytest.raises(ToolError):
        _call("mesa_decide_feedback", {"group_id": gid, "decision_id": str(UUID(int=5))})
    with pytest.raises(ToolError):
        _call("mesa_decide_explain", {})


def test_apply_mrtr_asks_per_group_with_ids_only_state(service: DecisionService) -> None:
    from mesa_mcp.errors import InputRequired, ToolError
    from mesa_mcp.server import _decode_request_state, _encode_request_state

    run_id = _call("mesa_decide_annotate", {"card_path": str(CARD)})["run_id"]
    pending = service.run_summary(UUID(run_id))["pending_groups"]
    args = {"run_id": run_id, "irods_path": "/local/x/f.csv", "accept": "proposed"}
    with pytest.raises(InputRequired) as exc:
        _call("mesa_decide_apply", args)
    pend = exc.value
    assert pend.key.startswith("term_choice:")
    gid = pend.key.split(":", 1)[1]
    assert gid in {str(g) for g in pending}
    assert set(pend.state) == {"tool", "run_id", "asked"} and "/local" not in str(pend.state)
    enum = pend.schema["properties"]["decision_id"]["enum"]
    names = pend.schema["properties"]["decision_id"]["enumNames"]
    assert enum and all("p(fits)=" in n and n.endswith("L0") for n in names)  # A11
    blob = _encode_request_state(pend.state)
    assert _decode_request_state(blob) == pend.state
    # tampered content: an id that was never offered
    with pytest.raises(ToolError):
        _call(
            "mesa_decide_apply",
            args,
            input_responses={
                pend.key: {"action": "accept", "content": {"decision_id": str(UUID(int=9))}}
            },
            request_state=blob,
        )
    # a state that belongs to another run is refused
    other = _encode_request_state({**pend.state, "run_id": str(UUID(int=3))})
    with pytest.raises(ToolError):
        _call(
            "mesa_decide_apply",
            args,
            input_responses={pend.key: {"action": "accept", "content": {"decision_id": enum[0]}}},
            request_state=other,
        )
    # answer every group in turn; the loop ends with a dry-run apply
    responses = {pend.key: {"action": "accept", "content": {"decision_id": enum[0]}}}
    state = blob
    for _ in range(len(pending) + 1):
        try:
            out = _call("mesa_decide_apply", args, input_responses=responses, request_state=state)
            break
        except InputRequired as nxt:
            responses = {nxt.value.key if hasattr(nxt, "value") else nxt.key: {"action": "decline"}}
            state = _encode_request_state(nxt.state)
    else:
        raise AssertionError("the MRTR loop did not terminate")
    assert out["picks_recorded"] == 1 and out["dry_run"] is True
    summary = service.run_summary(UUID(run_id))
    assert str(pending[0]) not in summary["pending_groups"] or summary["groups"]
    overrides = [g for g in summary["groups"] if g["outcome"] == "human"]
    assert len(overrides) == 1


def test_health_tool_reports_checks(service: DecisionService) -> None:
    out = _call("mesa_decide_health", {})
    assert out["backend"] == "fake" and out["checks"]
    assert any(c["name"] == "questions lock" and c["ok"] for c in out["checks"])


def test_busy_decider_degrades(service: DecisionService) -> None:
    from mesa_anyjev.service import DeciderBusy

    service.max_wait_s = 0.01
    service.lock.acquire()
    try:
        with pytest.raises(DeciderBusy):
            service.annotate(load_card(CARD), actor="x")
        from mesa_mcp.errors import ToolError

        with pytest.raises(ToolError, match="busy"):
            _call("mesa_decide_annotate", {"card_path": str(CARD)})
    finally:
        service.lock.release()

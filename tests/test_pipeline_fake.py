"""The whole decide phase on every fixture card with the fake backend and recorded OLS
responses: every decision honest (level in the five values, probs iff calibration), every
AVU canonical, proposals written to the sidecar with snapshot_id NULL."""

from __future__ import annotations

from pathlib import Path

import pytest

from mesa_anyjev.avu import build_avu
from mesa_anyjev.backends.factory import make_backend
from mesa_anyjev.cards import load_card
from mesa_anyjev.config import load_config
from mesa_anyjev.ols import OLSLayer, RecordingOLS
from mesa_anyjev.pipeline import Annotator
from mesa_anyjev.planner.static_planner import StaticPlanner
from mesa_anyjev.policy import load_policy
from mesa_anyjev.provenance.store import DuckDBStore
from mesa_anyjev.providers.anyjev_provider import AnyJevProvider

ROOT = Path(__file__).resolve().parent
CARDS = sorted((ROOT / "fixtures" / "cards").glob("*.md"))
FIXTURES = ROOT / "fixtures" / "ols"

pytestmark = pytest.mark.skipif(
    not any(FIXTURES.glob("*.json")), reason="OLS fixtures not recorded"
)


def _annotator(store: DuckDBStore) -> Annotator:
    cfg = load_config(
        env={"MESA_ANYJEV_POLICY__PROFILE": "dev", "MESA_ANYJEV_OLS__FIXTURES": "replay"}
    )
    provider = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    ols = OLSLayer(RecordingOLS(None, FIXTURES, "replay"), max_candidates=cfg.policy.max_candidates)
    return Annotator(
        provider=provider,
        planner=StaticPlanner(),
        ols=ols,
        policy=load_policy(),
        store=store,
        cfg=cfg,
        actor="test",
    )


@pytest.mark.parametrize("card_path", CARDS, ids=[c.stem for c in CARDS])
def test_card_end_to_end(card_path: Path, duckdb_store: DuckDBStore) -> None:
    run = _annotator(duckdb_store).annotate(load_card(card_path))
    rows = duckdb_store.decisions(run.run_id)
    assert rows and len(rows) == run.n_decisions
    for r in rows:
        assert r["level"] in ("raw", "L0", "L1", "L2", "none")
        assert (r["probs"] is None) == (r["calibration"] == "none")
        assert r["outcome"] in (
            "auto",
            "proposed",
            "human",
            "escalated",
            "abstain",
            "rejected",
            "rule",
            "decider_unavailable",
        )
        if r["kind"] == "noul" and r["probs"] is not None:
            assert r["p_true"] is not None
    assert not any(r["outcome"] == "auto" for r in rows)  # shipped policy is proposed-only
    links = duckdb_store.links(run.run_id)
    assert links and all(link["snapshot_id"] is None for link in links)
    assert len([link for link in links if link["write_status"] == "proposed"]) == len(run.proposals)
    for p in run.proposals:
        assert p.avu == build_avu(p.candidate, p.avu["value"])
        assert p.avu["unit"] == p.candidate.curie
        assert p.avu["attribute"].startswith(p.candidate.ontology_id + ".")
    result = run.to_eval_result()
    assert result["card"] == load_card(card_path).name and result["parse_ok"]
    assert duckdb_store.run(run.run_id)["status"] == "decided"  # type: ignore[index]
    assert duckdb_store.groups(run.run_id)


def test_two_runs_on_one_card_agree(duckdb_store: DuckDBStore) -> None:
    """With the running prior reset per run, the fake backend is deterministic (A6)."""
    ann = _annotator(duckdb_store)
    card = load_card(CARDS[0])
    a = ann.annotate(card).to_eval_result()["avus"]
    b = ann.annotate(card).to_eval_result()["avus"]
    assert [(x["attribute"], x["value"], x["unit"]) for x in a] == [
        (x["attribute"], x["value"], x["unit"]) for x in b
    ]

"""Engine tier: the real CARC gateway at L0 (MESA_ANYJEV_ENGINE=gateway, the tunnel up, the
LiteLLM key in MESA_LLM_API_KEY or MESA_ANYJEV_BACKEND__GATEWAY_API_KEY)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mesa_anyjev.backends.factory import make_backend
from mesa_anyjev.cards import load_card
from mesa_anyjev.config import load_config
from mesa_anyjev.ols import OLSLayer, RecordingOLS
from mesa_anyjev.pipeline import Annotator
from mesa_anyjev.planner.static_planner import StaticPlanner
from mesa_anyjev.policy import load_policy
from mesa_anyjev.provenance.store import DuckDBStore
from mesa_anyjev.providers.anyjev_provider import AnyJevProvider
from mesa_anyjev.questions import Q_COLUMN_ANNOTATE, Q_TERM_FITS
from mesa_anyjev.states import column_state

pytestmark = pytest.mark.engine
ROOT = Path(__file__).resolve().parents[1]

if os.environ.get("MESA_ANYJEV_ENGINE") != "gateway":
    pytest.skip("MESA_ANYJEV_ENGINE=gateway not set", allow_module_level=True)


@pytest.fixture(scope="module")
def cfg():  # type: ignore[no-untyped-def]
    c = load_config(
        env={
            **os.environ,
            "MESA_ANYJEV_BACKEND__KIND": "gateway",
            "MESA_ANYJEV_POLICY__PROFILE": "dev",
        }
    )
    if not c.backend.gateway_api_key:
        pytest.skip("no gateway key")
    return c


@pytest.fixture(scope="module")
def provider(cfg):  # type: ignore[no-untyped-def]
    backend = make_backend(cfg.backend)
    assert backend.probe().ok, "gateway probe failed"
    return AnyJevProvider(backend, cfg.decider, served_model=cfg.backend.served_model)


def test_noul_l0_is_debiased_and_honest(provider, card) -> None:  # type: ignore[no-untyped-def]
    states = [column_state(card, c) for c in card.columns]
    recs = provider.decide_batch(states, Q_COLUMN_ANNOTATE)
    assert all(r.level == "L0" and r.calibration == "anyjev" for r in recs)
    assert all(r.diagnostics.get("permutations") == 2 for r in recs)
    assert provider.last_missing_labels == 0
    assert provider.backend.requests >= 2 * len(states)


def test_term_fits_missing_labels_are_counted(provider, card) -> None:  # type: ignore[no-untyped-def]
    from mesa_anyjev.states import candidate_state

    cand = {
        "label": "distance",
        "curie": "PATO:0000040",
        "ontology_id": "pato",
        "description": "a spatial extent",
    }
    recs = provider.decide_batch(
        [candidate_state(card, "column", card.column("observerDistance"), "measurement", cand, 1)],
        Q_TERM_FITS,
    )
    assert 0.0 <= (recs[0].p_true or 0.0) <= 1.0
    assert isinstance(provider.last_missing_labels, int)


def test_one_card_end_to_end_at_l0(provider, cfg, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    store = DuckDBStore(tmp_path / "prov.duckdb")
    store.ensure_schema()
    ols = OLSLayer(
        RecordingOLS(None, ROOT / "fixtures" / "ols", "replay"),
        max_candidates=cfg.policy.max_candidates,
    )
    ann = Annotator(
        provider=provider,
        planner=StaticPlanner(),
        ols=ols,
        policy=load_policy(),
        store=store,
        cfg=cfg,
        actor="engine-test",
    )
    run = ann.annotate(load_card(ROOT / "fixtures" / "cards" / "DP1.10022.001.bet_sorting.md"))
    rows = store.decisions(run.run_id)
    assert rows and {r["level"] for r in rows if r["calibration"] == "anyjev"} == {"L0"}
    assert run.n_prompts > 0
    store.close()

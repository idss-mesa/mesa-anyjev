"""Parquet export of a run and snapshot reconciliation against a local DuckLake."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import duckdb

from mesa_anyjev.apply import apply
from mesa_anyjev.backends.factory import make_backend
from mesa_anyjev.cards import load_card
from mesa_anyjev.cli import main
from mesa_anyjev.config import load_config
from mesa_anyjev.ols import OLSLayer, RecordingOLS
from mesa_anyjev.pipeline import Annotator
from mesa_anyjev.planner.static_planner import StaticPlanner
from mesa_anyjev.policy import load_policy
from mesa_anyjev.provenance.export import export_run, reconcile
from mesa_anyjev.provenance.store import DuckDBStore
from mesa_anyjev.providers.anyjev_provider import AnyJevProvider

ROOT = Path(__file__).resolve().parent
CARD = ROOT / "fixtures" / "cards" / "DP1.10022.001.bet_sorting.md"


def _run(store: DuckDBStore) -> UUID:
    cfg = load_config(
        env={"MESA_ANYJEV_POLICY__PROFILE": "dev", "MESA_ANYJEV_OLS__FIXTURES": "replay"}
    )
    provider = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    ols = OLSLayer(RecordingOLS(None, ROOT / "fixtures" / "ols", "replay"), max_candidates=12)
    run = Annotator(
        provider=provider,
        planner=StaticPlanner(),
        ols=ols,
        policy=load_policy(),
        store=store,
        cfg=cfg,
        actor="t",
    ).annotate(load_card(CARD))
    return run.run_id


def test_export_and_reconcile(tmp_path: Path) -> None:
    from mesa_ducklake import DuckLakeClient

    store = DuckDBStore(tmp_path / "prov.duckdb")
    store.ensure_schema()
    run_id = _run(store)
    lake = DuckLakeClient(
        catalog_dsn=f"duckdb:///{tmp_path / 'lake.duckdb'}",
        irods_session=None,
        cache_dir=tmp_path / "cache",
        cache_cap_bytes=0,
    )
    path = "/local/mesa-anyjev/DP1.10022.001/bet_sorting.csv"
    res = apply(run_id, store=store, actor="t", irods_path=path, accept="proposed", ducklake=lake)
    assert res.snapshot_id is not None and res.written
    # forget the snapshot on every link, then reconcile it back from the DuckLake history
    store._con.execute(
        "UPDATE mesa_anyjev.avu_links SET snapshot_id = NULL WHERE run_id = ?", [str(run_id)]
    )
    assert all(
        link["snapshot_id"] is None
        for link in store.links(run_id)
        if link["write_status"] == "written"
    )
    fixed = reconcile(store, run_id, lake)
    assert fixed == len(res.written)
    assert {
        link["snapshot_id"] for link in store.links(run_id) if link["write_status"] == "written"
    } == {res.snapshot_id}
    written = export_run(store, run_id, tmp_path / "export")
    assert set(written) == {"runs", "decisions", "decision_groups", "avu_links"}
    con = duckdb.connect()
    n_links = con.execute(
        "SELECT count(*) FROM read_parquet(?)", [written["avu_links"]]
    ).fetchone()[0]
    assert n_links == len(store.links(run_id))
    cols = [
        r[0]
        for r in con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [written["decisions"]]
        ).fetchall()
    ]
    assert {"decision_id", "level", "calibration", "p_true", "state_json"} <= set(cols)
    con.close()
    lake.close()
    store.close()
    # the CLI verbs
    assert (
        main(
            [
                "provenance",
                "export",
                "--dsn",
                f"duckdb:///{tmp_path / 'prov.duckdb'}",
                "--run-id",
                str(run_id),
                "--out",
                str(tmp_path / "e2"),
            ]
        )
        == 0
    )
    assert (tmp_path / "e2" / str(run_id) / "runs.parquet").exists()

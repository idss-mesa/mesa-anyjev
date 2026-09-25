"""apply() in local mode: one DuckLake snapshot per (run, path), links get the snapshot id,
the source tag is visible in get_history, and an empty acceptance never calls record_changes."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from mesa_ducklake import DuckLakeClient

from mesa_anyjev.apply import apply
from mesa_anyjev.provenance.models import AvuLinkRow, RunRow
from mesa_anyjev.provenance.store import DuckDBStore

PATH = "/local/mesa-anyjev/DP1.10003.001/brd_countdata.csv"


def _seed(store: DuckDBStore) -> UUID:
    run = RunRow(
        actor="alice",
        card_name="c",
        card_sha256="s" * 64,
        planner="static",
        backend_kind="fake",
        anyjev_commit="x",
        mesa_anyjev_version="0",
        questions_lock_sha="l",
        policy_profile="dev",
        config_sha256="c",
    )
    store.begin_run(run)
    store.insert_links(
        [
            AvuLinkRow(
                run_id=run.run_id,
                attribute="ncbitaxon.aves",
                value="Aves",
                unit="NCBITaxon:8782",
                write_status="proposed",
            ),
            AvuLinkRow(
                run_id=run.run_id,
                attribute="uo.meter",
                value="meter",
                unit="UO:0000008",
                write_status="proposed",
                column_name="observerDistance",
            ),
            AvuLinkRow(
                run_id=run.run_id,
                attribute="pato.age",
                value="age",
                unit="PATO:0000011",
                write_status="rejected",
            ),
        ]
    )
    return run.run_id


@pytest.fixture
def lake(tmp_path: Path) -> DuckLakeClient:
    client = DuckLakeClient(
        catalog_dsn=f"duckdb:///{tmp_path / 'lake.duckdb'}",
        irods_session=None,
        cache_dir=tmp_path / "cache",
        cache_cap_bytes=0,
    )
    yield client  # type: ignore[misc]
    client.close()


def test_apply_proposed_writes_one_snapshot(
    duckdb_store: DuckDBStore, lake: DuckLakeClient
) -> None:
    run_id = _seed(duckdb_store)
    res = apply(
        run_id, store=duckdb_store, actor="alice", irods_path=PATH, accept="proposed", ducklake=lake
    )
    assert len(res.written) == 2 and res.snapshot_id is not None and res.mirror_error is None
    links = duckdb_store.links(run_id)
    written = [link for link in links if link["write_status"] == "written"]
    assert len(written) == 2 and all(link["snapshot_id"] == res.snapshot_id for link in written)
    assert duckdb_store.run(run_id)["status"] == "applied"  # type: ignore[index]
    project = lake.find_project_by_path("/local/mesa-anyjev/DP1.10003.001")
    assert project is not None
    history = lake.get_history(project.project_id, PATH)
    assert {h.source for h in history} == {"mesa-anyjev:apply:human"}
    assert {h.attribute for h in lake.get_avus(project.project_id, PATH)} == {
        "ncbitaxon.aves",
        "uo.meter",
    }


def test_auto_accepts_nothing_when_nothing_was_auto(
    duckdb_store: DuckDBStore, lake: DuckLakeClient
) -> None:
    run_id = _seed(duckdb_store)
    res = apply(
        run_id, store=duckdb_store, actor="alice", irods_path=PATH, accept="auto", ducklake=lake
    )
    assert res.written == [] and res.snapshot_id is None
    assert (
        lake.find_project_by_path("/local/mesa-anyjev/DP1.10003.001") is None
    )  # record_changes never called


def test_dry_run_and_explicit_ids(duckdb_store: DuckDBStore, lake: DuckLakeClient) -> None:
    run_id = _seed(duckdb_store)
    link_ids = [
        str(link["link_id"])
        for link in duckdb_store.links(run_id)
        if link["attribute"] == "uo.meter"
    ]
    res = apply(
        run_id,
        store=duckdb_store,
        actor="alice",
        irods_path=PATH,
        accept=link_ids,
        ducklake=lake,
        dry_run=True,
    )
    assert res.dry_run and len(res.written) == 1 and res.snapshot_id is None
    assert [
        link["write_status"]
        for link in duckdb_store.links(run_id)
        if link["attribute"] == "uo.meter"
    ] == ["dry_run"]


def test_mirror_failure_is_reported_not_raised(duckdb_store: DuckDBStore) -> None:
    run_id = _seed(duckdb_store)

    class Broken:
        def find_project_by_path(self, p: str) -> None:
            raise RuntimeError("catalog down")

    res = apply(
        run_id,
        store=duckdb_store,
        actor="alice",
        irods_path=PATH,
        accept="proposed",
        ducklake=Broken(),
    )
    assert res.mirror_error and "catalog down" in res.mirror_error
    assert {
        link["write_status"]
        for link in duckdb_store.links(run_id)
        if link["attribute"] != "pato.age"
    } == {"mirror_failed"}
    assert duckdb_store.run(run_id)["status"] == "partial"  # type: ignore[index]

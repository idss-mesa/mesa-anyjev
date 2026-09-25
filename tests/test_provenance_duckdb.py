from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from mesa_anyjev.provenance.migrate import apply_migrations, migration_files
from mesa_anyjev.provenance.models import (
    AvuLinkRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    HumanOverrideRow,
    LabelRow,
    RunRow,
)
from mesa_anyjev.provenance.store import TABLES, DuckDBStore, open_store


def _run() -> RunRow:
    return RunRow(
        actor="alice",
        card_name="DP1.x",
        card_sha256="s" * 64,
        planner="static",
        backend_kind="fake",
        anyjev_commit="795a497",
        mesa_anyjev_version="0.1.0.dev0",
        questions_lock_sha="l" * 64,
        policy_profile="dev",
        config_sha256="c" * 64,
    )


def test_schema_tables_exist(duckdb_store: DuckDBStore) -> None:
    for table in TABLES:
        assert duckdb_store.columns(table), table


def test_round_trip(duckdb_store: DuckDBStore) -> None:
    run = _run()
    duckdb_store.begin_run(run)
    group = DecisionGroupRow(
        run_id=run.run_id,
        question_id="term.fits",
        scope="column",
        column_name="c",
        search_json={"queries": ["distance"]},
        n_candidates=2,
        outcome="proposed",
    )
    duckdb_store.insert_group(group)
    d1 = DecisionRow(
        run_id=run.run_id,
        seq=1,
        group_id=group.group_id,
        question_id="term.fits",
        question_key="k" * 16,
        kind="noul",
        k=2,
        scope="column",
        column_name="c",
        state_sha256="a" * 64,
        state_json={"card": {}, "candidate": {"curie": "PATO:1"}},
        provider="anyjev",
        method="anyjev",
        model="fake",
        level="L0",
        calibration="anyjev",
        probs=[0.7, 0.3],
        answer_index=0,
        answer="Yes",
        confidence=0.7,
        p_true=0.7,
        margin=0.4,
        outcome="proposed",
    )
    d2 = DecisionRow(
        run_id=run.run_id,
        seq=2,
        question_id="column.annotate",
        question_key="j" * 16,
        kind="noul",
        k=2,
        scope="column",
        column_name="uid",
        state_sha256="b" * 64,
        state_json={},
        provider="rule",
        method="rule",
        model="n/a",
        level="none",
        calibration="none",
        probs=None,
        answer_index=1,
        answer="No",
        outcome="rule",
    )
    opts = [
        DecisionOptionRow(decision_id=d1.decision_id, option_index=0, option_text="Yes", prob=0.7),
        DecisionOptionRow(decision_id=d1.decision_id, option_index=1, option_text="No", prob=0.3),
    ]
    assert duckdb_store.insert_decisions([d1, d2], opts) == 2
    link = AvuLinkRow(
        run_id=run.run_id,
        group_id=group.group_id,
        decision_id=d1.decision_id,
        irods_path="/local/x/f.csv",
        attribute="pato.distance",
        value="distance",
        unit="PATO:1",
        column_name="c",
        write_status="proposed",
    )
    assert duckdb_store.insert_links([link]) == 1
    duckdb_store.set_link_status([link.link_id], "written", written_at=datetime.now(tz=UTC))
    assert duckdb_store.link_snapshot(run.run_id, "/local/x/f.csv", None, 7) == 1
    duckdb_store.insert_override(
        HumanOverrideRow(
            group_id=group.group_id,
            actor="alice",
            action="pick",
            chosen_index=0,
            offered=[{"curie": "PATO:1"}],
        )
    )
    labels = [
        LabelRow(
            question_id="term.fits",
            question_key="k" * 16,
            state_sha256="a" * 64,
            state_json={},
            label_index=0,
            label_source="curator",
            weight=1.0,
            card="DP1.x",
        ),
        LabelRow(
            question_id="term.fits",
            question_key="k" * 16,
            state_sha256="a" * 64,
            state_json={},
            label_index=0,
            label_source="curator",
            weight=1.0,
            card="DP1.x",
        ),
    ]
    assert duckdb_store.insert_labels(labels) == 1  # UNIQUE (key, state, source)
    duckdb_store.finish_run(run.run_id, "applied", n_prompts=4, seconds=1.5)

    got = duckdb_store.run(run.run_id)
    assert got and got["status"] == "applied" and got["n_prompts"] == 4
    decisions = duckdb_store.decisions(run.run_id)
    assert [d["seq"] for d in decisions] == [1, 2]
    assert decisions[0]["probs"] == [0.7, 0.3] and decisions[1]["probs"] is None
    links = duckdb_store.links(run.run_id)
    assert links[0]["snapshot_id"] == 7 and links[0]["write_status"] == "written"
    assert duckdb_store.labels_for("k" * 16, min_weight=0.6) and not duckdb_store.labels_for(
        "k" * 16, exclude_cards=["DP1.x"]
    )
    joined = duckdb_store.decisions_for_path("/local/x/f.csv")
    assert joined[0]["level"] == "L0" and joined[0]["snapshot_id"] == 7


def test_probs_null_iff_calibration_none_is_enforced_by_the_store(
    duckdb_store: DuckDBStore,
) -> None:
    run = _run()
    duckdb_store.begin_run(run)
    bad = DecisionRow(
        run_id=run.run_id,
        seq=1,
        question_id="q",
        question_key="k" * 16,
        kind="noul",
        k=2,
        scope="column",
        state_sha256="a" * 64,
        state_json={},
        provider="p",
        method="rule",
        model="m",
        level="none",
        calibration="none",
        probs=[0.5, 0.5],
        answer_index=0,
        answer="Yes",
        outcome="rule",
    )
    with pytest.raises(Exception, match=r"CHECK|constraint"):
        duckdb_store.insert_decisions([bad])


def test_finish_run_rejects_unknown_columns(duckdb_store: DuckDBStore) -> None:
    run = _run()
    duckdb_store.begin_run(run)
    with pytest.raises(ValueError):
        duckdb_store.finish_run(run.run_id, "failed", status_reason="x")


def test_open_store_and_migrate_duckdb(tmp_path: Path) -> None:
    dsn = f"duckdb:///{tmp_path / 'p.duckdb'}"
    assert apply_migrations(dsn) == 1
    store = open_store(dsn)
    assert store.run(uuid4()) is None
    store.close()
    with pytest.raises(ValueError):
        open_store("sqlite:///x")


def test_packaged_migrations_are_numbered() -> None:
    files = migration_files()
    assert [v for v, _, _ in files] == [1]
    assert "CREATE SCHEMA IF NOT EXISTS mesa_anyjev" in files[0][2]

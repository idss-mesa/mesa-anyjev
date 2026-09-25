"""Postgres sidecar (``pg`` extra): the same round trip as the DuckDB store against a real
server. Gated: ``MESA_ANYJEV_TEST_PG_DSN=postgresql://user:pw@host:port/db`` (a throwaway
database; the test creates and drops schema ``mesa_anyjev``)."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from mesa_anyjev.provenance.models import (
    AvuLinkRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    HumanOverrideRow,
    LabelRow,
    RunRow,
)
from mesa_anyjev.provenance.store import TABLES, open_store

pytestmark = pytest.mark.requires_postgres
DSN = os.environ.get("MESA_ANYJEV_TEST_PG_DSN")
if not DSN:
    pytest.skip("MESA_ANYJEV_TEST_PG_DSN not set", allow_module_level=True)
pytest.importorskip("psycopg")


@pytest.fixture
def pg():  # type: ignore[no-untyped-def]
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as con:
        con.execute("DROP SCHEMA IF EXISTS mesa_anyjev CASCADE")
    store = open_store(DSN)
    yield store
    store.close()
    with psycopg.connect(DSN, autocommit=True) as con:
        con.execute("DROP SCHEMA IF EXISTS mesa_anyjev CASCADE")


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


def test_postgres_round_trip(pg) -> None:  # type: ignore[no-untyped-def]
    assert pg.ensure_schema() >= 1
    for table in TABLES:
        assert pg.columns(table), table
    run = _run()
    pg.begin_run(run)
    group = DecisionGroupRow(
        run_id=run.run_id,
        question_id="term.fits",
        scope="column",
        column_name="c",
        search_json={"queries": ["distance"]},
        n_candidates=2,
        outcome="proposed",
    )
    pg.insert_group(group)
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
    opts = [
        DecisionOptionRow(decision_id=d1.decision_id, option_index=0, option_text="Yes", prob=0.7),
        DecisionOptionRow(decision_id=d1.decision_id, option_index=1, option_text="No", prob=0.3),
    ]
    assert pg.insert_decisions([d1], opts) == 1
    link = AvuLinkRow(
        run_id=run.run_id,
        group_id=group.group_id,
        decision_id=d1.decision_id,
        irods_path="/iplant/home/x/f.csv",
        attribute="pato.distance",
        value="distance",
        unit="PATO:1",
        column_name="c",
        write_status="proposed",
    )
    assert pg.insert_links([link]) == 1
    pg.set_link_status([link.link_id], "written", written_at=datetime.now(tz=UTC))
    assert pg.link_snapshot(run.run_id, "/iplant/home/x/f.csv", None, 7) == 1
    pg.update_group(group.group_id, outcome="human", winner_decision_id=d1.decision_id)
    pg.insert_override(
        HumanOverrideRow(
            group_id=group.group_id, actor="alice", action="pick", chosen_index=0, offered=[{}]
        )
    )
    label = LabelRow(
        question_id="term.fits",
        question_key="k" * 16,
        state_sha256="a" * 64,
        state_json={},
        label_index=0,
        label_source="curator",
        weight=1.0,
        card="DP1.x",
    )
    assert pg.insert_labels([label, label.model_copy(update={"label_id": label.label_id})]) == 1
    pg.finish_run(run.run_id, "applied", n_prompts=4, seconds=1.5)
    got = pg.run(run.run_id)
    assert got and got["status"] == "applied" and got["n_prompts"] == 4
    assert pg.group(group.group_id)["outcome"] == "human"
    decisions = pg.decisions(run.run_id)
    assert (
        decisions[0]["probs"] == [0.7, 0.3]
        and decisions[0]["state_json"]["candidate"]["curie"] == "PATO:1"
    )
    links = pg.links(run.run_id)
    assert links[0]["snapshot_id"] == 7 and links[0]["write_status"] == "written"
    assert pg.labels_for("k" * 16, min_weight=0.6) and not pg.labels_for(
        "k" * 16, exclude_cards=["DP1.x"]
    )
    joined = pg.decisions_for_path("/iplant/home/x/f.csv")
    assert joined[0]["level"] == "L0" and joined[0]["snapshot_id"] == 7
    # the CHECK (probs IS NULL) = (calibration = 'none') holds in Postgres too
    bad = d1.model_copy(
        update={"decision_id": d1.decision_id.__class__(int=1), "seq": 2, "calibration": "none"}
    )
    with pytest.raises(Exception, match=r"check|constraint"):
        pg.insert_decisions([bad])

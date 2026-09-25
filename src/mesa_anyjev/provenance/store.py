"""Provenance stores: a DuckDB file for development and tests, Postgres for production.

``open_store(dsn)`` dispatches on the DSN scheme the way ``mesa_ducklake.catalog.open_catalog``
does. The DuckDB store is single-writer and always its own file (never the mesa-ducklake
catalog file, which its ``DuckDBCatalogStore`` holds under a process-wide lock). Its DDL is the
DuckDB dialect of ``migrations/0001_mesa_anyjev.sql`` (JSON not JSONB, TEXT ids, no foreign
keys); ``tests/test_provenance_ddl.py`` asserts the two dialects expose the same columns.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID

import duckdb

from mesa_anyjev.provenance.models import (
    AvuLinkRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    HumanOverrideRow,
    LabelRow,
    RunRow,
)

SCHEMA_VERSION = 1

# One statement per table, DuckDB dialect. Column order matches the Postgres migration.
DUCKDB_DDL: tuple[str, ...] = (
    "CREATE SCHEMA IF NOT EXISTS mesa_anyjev",
    """CREATE TABLE IF NOT EXISTS mesa_anyjev.schema_versions (
        version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS mesa_anyjev.runs (
        run_id TEXT PRIMARY KEY, started_at TIMESTAMPTZ NOT NULL, finished_at TIMESTAMPTZ,
        status TEXT NOT NULL, actor TEXT NOT NULL, irods_path TEXT, project_id TEXT,
        card_name TEXT NOT NULL, card_sha256 TEXT NOT NULL, planner TEXT NOT NULL,
        planner_model TEXT, planner_prompt_sha256 TEXT, plan_json JSON NOT NULL,
        planner_fallback BOOLEAN NOT NULL DEFAULT FALSE, backend_kind TEXT NOT NULL,
        canonical_model TEXT, served_model TEXT, served_revision TEXT, tokenizer TEXT,
        tokenizer_revision TEXT, anyjev_commit TEXT NOT NULL, mesa_anyjev_version TEXT NOT NULL,
        questions_lock_sha TEXT NOT NULL, artifacts_sha256 TEXT, policy_profile TEXT NOT NULL,
        config_sha256 TEXT NOT NULL, write_mode TEXT NOT NULL DEFAULT 'none',
        hosted_provider_used BOOLEAN NOT NULL DEFAULT FALSE,
        data_left_host BOOLEAN NOT NULL DEFAULT FALSE, egress_region TEXT,
        missing_labels INTEGER NOT NULL DEFAULT 0, n_prompts INTEGER, seconds REAL)""",
    """CREATE TABLE IF NOT EXISTS mesa_anyjev.decisions (
        decision_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, seq INTEGER NOT NULL, group_id TEXT,
        parent_decision_id TEXT, question_id TEXT NOT NULL, question_key TEXT NOT NULL,
        kind TEXT NOT NULL, k INTEGER NOT NULL, scope TEXT NOT NULL, column_name TEXT,
        site_code TEXT, state_sha256 TEXT NOT NULL, state_json JSON NOT NULL, prompt_sha256 TEXT,
        provider TEXT NOT NULL, method TEXT NOT NULL, model TEXT NOT NULL, served_model TEXT,
        level TEXT NOT NULL, calibration TEXT NOT NULL, probs JSON, answer_index INTEGER NOT NULL,
        answer TEXT NOT NULL, confidence REAL, p_true REAL, margin REAL, score_value REAL,
        answer_mass REAL, prior_method TEXT, order_flip_l0 REAL, permutations INTEGER,
        temperature REAL, n_calib INTEGER, routed_from TEXT, adapted BOOLEAN,
        blocks_executed INTEGER, masked_mass REAL, threshold_auto REAL, threshold_propose REAL,
        outcome TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
        CHECK ((probs IS NULL) = (calibration = 'none')))""",
    """CREATE TABLE IF NOT EXISTS mesa_anyjev.decision_options (
        decision_id TEXT NOT NULL, option_index INTEGER NOT NULL, option_text TEXT NOT NULL,
        curie TEXT, iri TEXT, ontology_id TEXT, prob REAL, rank INTEGER,
        PRIMARY KEY (decision_id, option_index))""",
    """CREATE TABLE IF NOT EXISTS mesa_anyjev.decision_groups (
        group_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, question_id TEXT NOT NULL,
        scope TEXT NOT NULL, column_name TEXT, site_code TEXT, aspect TEXT, ontology_id TEXT,
        search_json JSON NOT NULL, n_candidates INTEGER NOT NULL, winner_decision_id TEXT,
        top_p REAL, group_margin REAL, level TEXT, outcome TEXT NOT NULL,
        missing_labels INTEGER NOT NULL DEFAULT 0, escalated_from TEXT, ts TIMESTAMPTZ NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS mesa_anyjev.avu_links (
        link_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, group_id TEXT, decision_id TEXT,
        project_id TEXT, snapshot_id BIGINT, irods_path TEXT, target_type TEXT NOT NULL,
        attribute TEXT NOT NULL, value TEXT NOT NULL, unit TEXT NOT NULL,
        op TEXT NOT NULL DEFAULT 'add', term_curie TEXT, term_iri TEXT, term_label TEXT,
        ontology_id TEXT, aspect TEXT, column_name TEXT, site_code TEXT, value_kind TEXT,
        source TEXT, write_status TEXT NOT NULL, duplicate_of TEXT, written_at TIMESTAMPTZ)""",
    """CREATE TABLE IF NOT EXISTS mesa_anyjev.human_overrides (
        override_id TEXT PRIMARY KEY, group_id TEXT, decision_id TEXT, link_id TEXT,
        actor TEXT NOT NULL, action TEXT NOT NULL, chosen_index INTEGER, chosen_curie TEXT,
        elicitation_key TEXT, offered JSON NOT NULL, ts TIMESTAMPTZ NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS mesa_anyjev.labels (
        label_id TEXT PRIMARY KEY, question_id TEXT NOT NULL, question_key TEXT NOT NULL,
        state_sha256 TEXT NOT NULL, state_json JSON NOT NULL, label_index INTEGER NOT NULL,
        label_source TEXT NOT NULL, weight REAL NOT NULL, source_ref TEXT, card TEXT,
        decision_id TEXT, batch_id TEXT, consumed_in_bundle TEXT, ts TIMESTAMPTZ NOT NULL,
        UNIQUE (question_key, state_sha256, label_source))""",
)

TABLES: tuple[str, ...] = (
    "runs",
    "decisions",
    "decision_options",
    "decision_groups",
    "avu_links",
    "human_overrides",
    "labels",
    "schema_versions",
)


class ProvenanceStore(Protocol):
    """The store contract the pipeline, the tools and the learners use."""

    def ensure_schema(self) -> int: ...
    def begin_run(self, run: RunRow) -> UUID: ...
    def finish_run(self, run_id: UUID, status: str, **stats: Any) -> None: ...
    def insert_decisions(
        self, rows: Sequence[DecisionRow], options: Sequence[DecisionOptionRow] = ()
    ) -> int: ...
    def insert_group(self, row: DecisionGroupRow) -> UUID: ...
    def insert_links(self, rows: Sequence[AvuLinkRow]) -> int: ...
    def set_link_status(
        self, link_ids: Iterable[UUID], status: str, *, written_at: datetime | None = None
    ) -> int: ...
    def link_snapshot(
        self, run_id: UUID, irods_path: str, project_id: UUID | None, snapshot_id: int
    ) -> int: ...
    def insert_override(self, row: HumanOverrideRow) -> UUID: ...
    def insert_labels(self, rows: Sequence[LabelRow]) -> int: ...
    def run(self, run_id: UUID) -> dict[str, Any] | None: ...
    def decisions(self, run_id: UUID) -> list[dict[str, Any]]: ...
    def links(self, run_id: UUID) -> list[dict[str, Any]]: ...
    def labels_for(
        self, question_key: str, *, min_weight: float = 0.0, exclude_cards: Sequence[str] = ()
    ) -> list[dict[str, Any]]: ...
    def decisions_for_path(self, irods_path: str, limit: int = 100) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _cell(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict | list):
        return _json(value)
    return value


def _row_values(row: Any) -> tuple[list[str], list[Any]]:
    data = row.model_dump()
    return list(data), [_cell(v) for v in data.values()]


class DuckDBStore:
    """A single-writer DuckDB file (development, tests, the GB10 bench host)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.path))

    # -- schema ------------------------------------------------------------------------------
    def ensure_schema(self) -> int:
        for stmt in DUCKDB_DDL:
            self._con.execute(stmt)
        self._con.execute(
            "INSERT INTO mesa_anyjev.schema_versions (version) SELECT ? WHERE NOT EXISTS "
            "(SELECT 1 FROM mesa_anyjev.schema_versions WHERE version = ?)",
            [SCHEMA_VERSION, SCHEMA_VERSION],
        )
        return SCHEMA_VERSION

    def columns(self, table: str) -> list[tuple[str, bool]]:
        rows = self._con.execute(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'mesa_anyjev' AND table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        return [(str(name), str(nullable).upper() == "YES") for name, nullable in rows]

    # -- writes -------------------------------------------------------------------------------
    def _insert(self, table: str, rows: Sequence[Any]) -> int:
        if not rows:
            return 0
        cols, _ = _row_values(rows[0])
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO mesa_anyjev.{table} ({', '.join(cols)}) VALUES ({placeholders})"  # noqa: S608
        self._con.executemany(sql, [_row_values(r)[1] for r in rows])
        return len(rows)

    def begin_run(self, run: RunRow) -> UUID:
        self._insert("runs", [run])
        return run.run_id

    def finish_run(self, run_id: UUID, status: str, **stats: Any) -> None:
        allowed = {
            "finished_at",
            "n_prompts",
            "seconds",
            "missing_labels",
            "irods_path",
            "project_id",
            "write_mode",
            "hosted_provider_used",
            "data_left_host",
            "egress_region",
            "artifacts_sha256",
        }
        sets = ["status = ?"]
        values: list[Any] = [status]
        for key, value in stats.items():
            if key not in allowed:
                raise ValueError(f"finish_run: {key} is not an updatable run column")
            sets.append(f"{key} = ?")
            values.append(_cell(value))
        values.append(str(run_id))
        self._con.execute(f"UPDATE mesa_anyjev.runs SET {', '.join(sets)} WHERE run_id = ?", values)  # noqa: S608

    def insert_decisions(
        self, rows: Sequence[DecisionRow], options: Sequence[DecisionOptionRow] = ()
    ) -> int:
        n = self._insert("decisions", rows)
        self._insert("decision_options", options)
        return n

    def insert_group(self, row: DecisionGroupRow) -> UUID:
        self._insert("decision_groups", [row])
        return row.group_id

    def insert_links(self, rows: Sequence[AvuLinkRow]) -> int:
        return self._insert("avu_links", rows)

    def set_link_status(
        self, link_ids: Iterable[UUID], status: str, *, written_at: datetime | None = None
    ) -> int:
        ids = [str(i) for i in link_ids]
        if not ids:
            return 0
        self._con.executemany(
            "UPDATE mesa_anyjev.avu_links SET write_status = ?, written_at = ? WHERE link_id = ?",
            [[status, written_at, i] for i in ids],
        )
        return len(ids)

    def link_snapshot(
        self, run_id: UUID, irods_path: str, project_id: UUID | None, snapshot_id: int
    ) -> int:
        cur = self._con.execute(
            "UPDATE mesa_anyjev.avu_links SET snapshot_id = ?, project_id = ? "
            "WHERE run_id = ? AND irods_path = ? AND write_status = 'written' AND snapshot_id IS NULL",
            [snapshot_id, _cell(project_id), str(run_id), irods_path],
        )
        count = cur.fetchone()
        return int(count[0]) if count else 0

    def insert_override(self, row: HumanOverrideRow) -> UUID:
        self._insert("human_overrides", [row])
        return row.override_id

    def insert_labels(self, rows: Sequence[LabelRow]) -> int:
        inserted = 0
        for row in rows:
            cols, values = _row_values(row)
            placeholders = ", ".join("?" for _ in cols)
            cur = self._con.execute(
                f"INSERT OR IGNORE INTO mesa_anyjev.labels ({', '.join(cols)}) VALUES ({placeholders})",  # noqa: S608
                values,
            )
            got = cur.fetchone()
            inserted += int(got[0]) if got else 0
        return inserted

    # -- reads ----------------------------------------------------------------------------------
    def _select(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self._con.execute(sql, list(params))
        names = [d[0] for d in cur.description or []]
        out = []
        for values in cur.fetchall():
            row = dict(zip(names, values, strict=True))
            for key in ("plan_json", "state_json", "probs", "search_json", "offered"):
                if key in row and isinstance(row[key], str):
                    row[key] = json.loads(row[key])
            out.append(row)
        return out

    def run(self, run_id: UUID) -> dict[str, Any] | None:
        rows = self._select("SELECT * FROM mesa_anyjev.runs WHERE run_id = ?", [str(run_id)])
        return rows[0] if rows else None

    def decisions(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._select(
            "SELECT * FROM mesa_anyjev.decisions WHERE run_id = ? ORDER BY seq", [str(run_id)]
        )

    def links(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._select(
            "SELECT * FROM mesa_anyjev.avu_links WHERE run_id = ? ORDER BY attribute, value, unit",
            [str(run_id)],
        )

    def labels_for(
        self, question_key: str, *, min_weight: float = 0.0, exclude_cards: Sequence[str] = ()
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM mesa_anyjev.labels WHERE question_key = ? AND weight >= ?"
        params: list[Any] = [question_key, min_weight]
        if exclude_cards:
            sql += (
                " AND (card IS NULL OR card NOT IN (" + ", ".join("?" for _ in exclude_cards) + "))"
            )
            params.extend(exclude_cards)
        return self._select(sql + " ORDER BY ts", params)

    def decisions_for_path(self, irods_path: str, limit: int = 100) -> list[dict[str, Any]]:
        return self._select(
            "SELECT l.attribute, l.value, l.unit, l.snapshot_id, l.write_status, l.source, "
            "d.question_id, d.level, d.calibration, d.confidence, d.p_true, d.model, d.served_model, d.outcome "
            "FROM mesa_anyjev.avu_links l LEFT JOIN mesa_anyjev.decisions d USING (decision_id) "
            "WHERE l.irods_path = ? ORDER BY l.attribute, l.value, l.unit LIMIT ?",
            [irods_path, limit],
        )

    def close(self) -> None:
        self._con.close()


_DUCKDB_RE = re.compile(r"^duckdb:///(.+)$")


def open_store(dsn: str) -> ProvenanceStore:
    """Dispatch on the DSN scheme: ``duckdb:///path`` or ``*.duckdb`` -> :class:`DuckDBStore`;
    ``postgresql://`` -> the Postgres store (``pg`` extra, M4)."""
    if not dsn or not dsn.strip():
        raise ValueError("provenance DSN is empty")
    if m := _DUCKDB_RE.match(dsn):
        store = DuckDBStore(m.group(1))
    elif dsn.endswith(".duckdb"):
        store = DuckDBStore(dsn)
    elif dsn.startswith(("postgresql://", "postgres://")):
        from mesa_anyjev.provenance.store_postgres import PostgresStore  # lazy: psycopg optional

        return cast(ProvenanceStore, PostgresStore(dsn))
    else:
        raise ValueError(f"unsupported provenance DSN {dsn!r} (duckdb:///... or postgresql://...)")
    store.ensure_schema()
    return store

"""Postgres provenance store (production; ``pg`` extra). Same protocol as
:class:`~mesa_anyjev.provenance.store.DuckDBStore`; the DDL is the packaged migration
(``migrations/0001_mesa_anyjev.sql``, schema ``mesa_anyjev``, JSONB, UUIDs, foreign keys).
The store never touches mesa-ducklake's own schema; both may live in one database (D4)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from mesa_anyjev.provenance.models import (
    AvuLinkRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    HumanOverrideRow,
    LabelRow,
    RunRow,
)
from mesa_anyjev.provenance.store import SCHEMA_VERSION

_JSON_COLUMNS = ("plan_json", "state_json", "probs", "search_json", "offered")


def _pg_cell(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    if isinstance(value, UUID):
        return value
    if isinstance(value, dict | list):
        return Jsonb(value)
    return value


def _row(row: Any) -> tuple[list[str], list[Any]]:
    data = row.model_dump()
    return list(data), [_pg_cell(v) for v in data.values()]


class PostgresStore:
    """One connection, autocommit off, one commit per public call."""

    def __init__(self, dsn: str) -> None:
        import psycopg

        self.dsn = dsn
        self._con = psycopg.connect(dsn)

    # -- schema ------------------------------------------------------------------------------
    def ensure_schema(self) -> int:
        from mesa_anyjev.provenance.migrate import apply_migrations

        version = apply_migrations(self.dsn)
        return int(version or SCHEMA_VERSION)

    def columns(self, table: str) -> list[tuple[str, bool]]:
        with self._con.cursor() as cur:
            cur.execute(
                "SELECT column_name, is_nullable FROM information_schema.columns "
                "WHERE table_schema = 'mesa_anyjev' AND table_name = %s ORDER BY ordinal_position",
                [table],
            )
            rows = cur.fetchall()
        return [(str(name), str(nullable).upper() == "YES") for name, nullable in rows]

    # -- writes -------------------------------------------------------------------------------
    def _insert(self, table: str, rows: Sequence[Any], *, ignore: bool = False) -> int:
        if not rows:
            return 0
        cols, _ = _row(rows[0])
        placeholders = ", ".join("%s" for _ in cols)
        sql = f"INSERT INTO mesa_anyjev.{table} ({', '.join(cols)}) VALUES ({placeholders})"  # noqa: S608
        if ignore:
            sql += " ON CONFLICT DO NOTHING"
        inserted = 0
        with self._con.cursor() as cur:
            for r in rows:
                cur.execute(sql, _row(r)[1])
                inserted += max(cur.rowcount, 0)
        self._con.commit()
        return inserted

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
        sets = ["status = %s"]
        values: list[Any] = [status]
        for key, value in stats.items():
            if key not in allowed:
                raise ValueError(f"finish_run: {key} is not an updatable run column")
            sets.append(f"{key} = %s")
            values.append(_pg_cell(value))
        values.append(run_id)
        with self._con.cursor() as cur:
            cur.execute(f"UPDATE mesa_anyjev.runs SET {', '.join(sets)} WHERE run_id = %s", values)  # noqa: S608
        self._con.commit()

    def insert_decisions(
        self, rows: Sequence[DecisionRow], options: Sequence[DecisionOptionRow] = ()
    ) -> int:
        n = self._insert("decisions", rows)
        self._insert("decision_options", options)
        return n

    def insert_group(self, row: DecisionGroupRow) -> UUID:
        self._insert("decision_groups", [row])
        return row.group_id

    def update_group(self, group_id: UUID, **summary: Any) -> None:
        allowed = {
            "winner_decision_id",
            "top_p",
            "group_margin",
            "level",
            "outcome",
            "missing_labels",
            "n_candidates",
        }
        sets, values = [], []
        for key, value in summary.items():
            if key not in allowed:
                raise ValueError(f"update_group: {key} is not a summary column")
            sets.append(f"{key} = %s")
            values.append(_pg_cell(value))
        if not sets:
            return
        values.append(group_id)
        with self._con.cursor() as cur:
            cur.execute(
                f"UPDATE mesa_anyjev.decision_groups SET {', '.join(sets)} WHERE group_id = %s",  # noqa: S608
                values,
            )
        self._con.commit()

    def insert_links(self, rows: Sequence[AvuLinkRow]) -> int:
        return self._insert("avu_links", rows)

    def set_link_status(
        self,
        link_ids: Iterable[UUID],
        status: str,
        *,
        written_at: datetime | None = None,
        irods_path: str | None = None,
    ) -> int:
        ids = list(link_ids)
        if not ids:
            return 0
        with self._con.cursor() as cur:
            if irods_path is None:
                cur.executemany(
                    "UPDATE mesa_anyjev.avu_links SET write_status = %s, written_at = %s "
                    "WHERE link_id = %s",
                    [[status, written_at, i] for i in ids],
                )
            else:
                cur.executemany(
                    "UPDATE mesa_anyjev.avu_links SET write_status = %s, written_at = %s, "
                    "irods_path = %s WHERE link_id = %s",
                    [[status, written_at, irods_path, i] for i in ids],
                )
        self._con.commit()
        return len(ids)

    def link_snapshot(
        self, run_id: UUID, irods_path: str, project_id: UUID | None, snapshot_id: int
    ) -> int:
        with self._con.cursor() as cur:
            cur.execute(
                "UPDATE mesa_anyjev.avu_links SET snapshot_id = %s, project_id = %s "
                "WHERE run_id = %s AND irods_path = %s AND write_status = 'written' "
                "AND snapshot_id IS NULL",
                [snapshot_id, project_id, run_id, irods_path],
            )
            count = cur.rowcount
        self._con.commit()
        return max(int(count), 0)

    def insert_override(self, row: HumanOverrideRow) -> UUID:
        self._insert("human_overrides", [row])
        return row.override_id

    def insert_labels(self, rows: Sequence[LabelRow]) -> int:
        return self._insert("labels", rows, ignore=True)

    # -- reads ----------------------------------------------------------------------------------
    def _select(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._con.cursor() as cur:
            cur.execute(sql, list(params))
            names = [d.name for d in cur.description or []]
            rows = cur.fetchall()
        self._con.rollback()  # end the read transaction
        out = []
        for values in rows:
            row = dict(zip(names, values, strict=True))
            for key in _JSON_COLUMNS:
                if key in row and isinstance(row[key], str):
                    row[key] = json.loads(row[key])
            for key, value in list(row.items()):
                if isinstance(value, UUID):
                    row[key] = str(value)
            out.append(row)
        return out

    def run(self, run_id: UUID) -> dict[str, Any] | None:
        rows = self._select("SELECT * FROM mesa_anyjev.runs WHERE run_id = %s", [run_id])
        return rows[0] if rows else None

    def groups(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._select(
            "SELECT * FROM mesa_anyjev.decision_groups WHERE run_id = %s ORDER BY ts", [run_id]
        )

    def group(self, group_id: UUID) -> dict[str, Any] | None:
        rows = self._select(
            "SELECT * FROM mesa_anyjev.decision_groups WHERE group_id = %s", [group_id]
        )
        return rows[0] if rows else None

    def decisions(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._select(
            "SELECT * FROM mesa_anyjev.decisions WHERE run_id = %s ORDER BY seq", [run_id]
        )

    def links(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._select(
            "SELECT * FROM mesa_anyjev.avu_links WHERE run_id = %s ORDER BY attribute, value, unit",
            [run_id],
        )

    def labels_for(
        self, question_key: str, *, min_weight: float = 0.0, exclude_cards: Sequence[str] = ()
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM mesa_anyjev.labels WHERE question_key = %s AND weight >= %s"
        params: list[Any] = [question_key, min_weight]
        if exclude_cards:
            sql += (
                " AND (card IS NULL OR card NOT IN ("
                + ", ".join("%s" for _ in exclude_cards)
                + "))"
            )
            params.extend(exclude_cards)
        return self._select(sql + " ORDER BY ts", params)

    def decisions_for_path(self, irods_path: str, limit: int = 100) -> list[dict[str, Any]]:
        return self._select(
            "SELECT l.attribute, l.value, l.unit, l.snapshot_id, l.write_status, l.source, "
            "d.question_id, d.level, d.calibration, d.confidence, d.p_true, d.model, "
            "d.served_model, d.outcome "
            "FROM mesa_anyjev.avu_links l LEFT JOIN mesa_anyjev.decisions d USING (decision_id) "
            "WHERE l.irods_path = %s ORDER BY l.attribute, l.value, l.unit LIMIT %s",
            [irods_path, limit],
        )

    def close(self) -> None:
        self._con.close()

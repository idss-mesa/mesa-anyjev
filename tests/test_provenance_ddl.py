"""The DuckDB dialect and the Postgres migration expose the same columns and nullability."""

from __future__ import annotations

import re

from mesa_anyjev.provenance.migrate import migration_files
from mesa_anyjev.provenance.store import TABLES, DuckDBStore

_TABLE = re.compile(r"CREATE TABLE IF NOT EXISTS mesa_anyjev\.(\w+) \((.*?)\n\);", re.S)
_COL = re.compile(r"^\s*([a-z0-9_]+)\s+[A-Z]+", re.M)


def _pg_columns(sql: str) -> dict[str, list[tuple[str, bool]]]:
    out: dict[str, list[tuple[str, bool]]] = {}
    for m in _TABLE.finditer(sql):
        cols: list[tuple[str, bool]] = []
        for line in m.group(2).splitlines():
            stripped = line.strip().rstrip(",")
            if not stripped or stripped.startswith(("--", "CHECK", "PRIMARY", "UNIQUE")):
                continue
            head = stripped.split("--")[0]
            cm = _COL.match(head)
            if cm:
                nullable = "NOT NULL" not in head and "PRIMARY KEY" not in head
                cols.append((cm.group(1), nullable))
        out[m.group(1)] = cols
    return out


def test_dialects_agree(duckdb_store: DuckDBStore) -> None:
    pg = _pg_columns(migration_files()[0][2])
    assert set(pg) == set(TABLES)
    for table in TABLES:
        duck = duckdb_store.columns(table)
        assert [c for c, _ in duck] == [c for c, _ in pg[table]], table
        assert [n for _, n in duck] == [n for _, n in pg[table]], f"{table}: nullability differs"

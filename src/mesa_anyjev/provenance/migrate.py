"""Migration runner for the Postgres sidecar and schema bootstrap for DuckDB files.

Modelled on ``mesa_ducklake.schema.apply_migrations`` but self-contained: migrations are
packaged (``importlib.resources``), applied one per transaction in numeric order, and recorded
in ``mesa_anyjev.schema_versions``. A ``duckdb:///`` DSN runs :meth:`DuckDBStore.ensure_schema`.
"""

from __future__ import annotations

import re
from importlib import resources

from mesa_anyjev.provenance.store import _DUCKDB_RE, DuckDBStore

_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def migration_files() -> list[tuple[int, str, str]]:
    """``(version, filename, sql)`` for every packaged migration, ascending."""
    out: list[tuple[int, str, str]] = []
    for entry in resources.files("mesa_anyjev.provenance.migrations").iterdir():
        m = _NAME.match(entry.name)
        if m:
            out.append((int(m.group(1)), entry.name, entry.read_text(encoding="utf-8")))
    return sorted(out)


def apply_migrations(dsn: str, target: int | None = None) -> int:
    """Apply pending migrations; return the schema version now in force."""
    if _DUCKDB_RE.match(dsn) or dsn.endswith(".duckdb"):
        path = _DUCKDB_RE.match(dsn).group(1) if _DUCKDB_RE.match(dsn) else dsn  # type: ignore[union-attr]
        store = DuckDBStore(path)
        try:
            return store.ensure_schema()
        finally:
            store.close()
    if not dsn.startswith(("postgresql://", "postgres://")):
        raise ValueError(f"unsupported DSN {dsn!r}")
    import psycopg

    applied = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS mesa_anyjev")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS mesa_anyjev.schema_versions "
                "(version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            cur.execute("SELECT COALESCE(MAX(version), 0) FROM mesa_anyjev.schema_versions")
            row = cur.fetchone()
            current = int(row[0]) if row else 0
        conn.commit()
        for version, _name, sql in migration_files():
            if version <= current or (target is not None and version > target):
                continue
            with conn.cursor() as cur:
                cur.execute(sql)  # the file carries its own BEGIN/COMMIT
            applied = version
    return applied or current

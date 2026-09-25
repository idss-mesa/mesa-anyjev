"""Postgres provenance store (production; ``pg`` extra). Implemented in M4."""

from __future__ import annotations


class PostgresStore:
    def __init__(self, dsn: str) -> None:
        raise NotImplementedError(
            "the Postgres provenance store lands in milestone M4; use duckdb:///... meanwhile"
        )

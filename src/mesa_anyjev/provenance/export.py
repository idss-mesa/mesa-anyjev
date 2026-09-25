"""Parquet copies of a run's sidecar rows under ``<project>/.mesa/anyjev/`` (never inside
``.mesa/ducklake/``, which belongs to mesa-ducklake), and ``reconcile`` for links whose
snapshot id is still NULL because the mirror step failed after the iRODS write.

The export goes through an in-memory DuckDB (``read_json_auto`` -> ``COPY ... TO parquet``) so
it works for both stores without pandas or pyarrow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import duckdb

from mesa_anyjev.provenance.store import ProvenanceStore

EXPORT_TABLES = ("runs", "decisions", "decision_groups", "avu_links")


def export_run(store: ProvenanceStore, run_id: UUID, out_dir: str | Path) -> dict[str, str]:
    """Write ``runs``, ``decisions``, ``decision_groups`` and ``avu_links`` for one run as
    ``<out_dir>/<run_id>/<table>.parquet``; returns table -> path."""
    run = store.run(run_id)
    if run is None:
        raise KeyError(f"run {run_id} not found")
    target = Path(out_dir).expanduser() / str(run_id)
    target.mkdir(parents=True, exist_ok=True)
    frames: dict[str, list[dict[str, Any]]] = {
        "runs": [run],
        "decisions": store.decisions(run_id),
        "decision_groups": store.groups(run_id),
        "avu_links": store.links(run_id),
    }
    written: dict[str, str] = {}
    con = duckdb.connect()
    try:
        for table, rows in frames.items():
            path = target / f"{table}.parquet"
            if not rows:
                continue
            tmp = target / f"{table}.jsonl"
            with tmp.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(_plain(row), default=str) + "\n")
            src, dst = _lit(tmp), _lit(path)
            con.execute(  # COPY targets cannot be bound parameters; paths are quoted literals
                f"COPY (SELECT * FROM read_json_auto({src}, format='newline_delimited')) "  # noqa: S608
                f"TO {dst} (FORMAT PARQUET)"
            )
            tmp.unlink()
            written[table] = str(path)
    finally:
        con.close()
    return written


def _lit(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _plain(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, UUID):
            out[key] = str(value)
        elif isinstance(value, dict | list):
            out[key] = json.dumps(value, sort_keys=True, default=str)
        else:
            out[key] = value
    return out


def reconcile(store: ProvenanceStore, run_id: UUID, ducklake: Any) -> int:
    """Fill ``snapshot_id`` on written links that lost the mirror step: match the DuckLake
    history (``source`` starting with ``mesa-anyjev:``) on the exact AVU triple."""
    run = store.run(run_id)
    if run is None:
        raise KeyError(f"run {run_id} not found")
    fixed = 0
    links = [
        link
        for link in store.links(run_id)
        if link["write_status"] in ("written", "mirror_failed") and link["snapshot_id"] is None
    ]
    if not links:
        return 0
    project_id = run.get("project_id")
    if project_id is None:
        return 0
    by_path: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        by_path.setdefault(str(link["irods_path"]), []).append(link)
    for path, path_links in by_path.items():
        history = ducklake.get_history(UUID(str(project_id)), path, limit=1000)
        rows = [r for r in history if str(getattr(r, "source", "")).startswith("mesa-anyjev:")]
        for link in path_links:
            match = next(
                (
                    r
                    for r in rows
                    if (r.attribute, r.value, r.unit or "")
                    == (link["attribute"], link["value"], link["unit"] or "")
                    and r.op == "add"
                ),
                None,
            )
            if match is not None and match.snapshot_id is not None:
                store.set_link_status([UUID(str(link["link_id"]))], "written", irods_path=path)
                store.link_snapshot(run_id, path, UUID(str(project_id)), int(match.snapshot_id))
                fixed += 1
    return fixed

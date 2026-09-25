"""The write phase (DESIGN D10, D13; amendments B2, B3).

``apply(run_id)`` takes stored proposals and writes the accepted ones. Two explicit modes,
recorded on the run: ``local`` (the CLI and tests; the public ``DuckLakeClient`` API only, no
iRODS) and ``irods`` (the MCP path; ``assert_allowed`` + the shared ``add_avu_to_irods`` helper,
lands with the mesa-mcp tools in M4). One ``record_changes`` per (run, path); never with an
empty list. A mirror failure after a successful iRODS write becomes ``mirror_failed`` on the
links and a partial-failure result, never an exception.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from mesa_anyjev.provenance.store import ProvenanceStore

SOURCE_PREFIX = "mesa-anyjev"
Accept = Literal["auto", "proposed", "all"]
Mode = Literal["local", "irods"]


@dataclass
class ApplyResult:
    run_id: UUID
    mode: Mode
    written: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    snapshot_id: int | None = None
    project_id: str | None = None
    mirror_error: str | None = None
    dry_run: bool = False


def _accepted(links: list[dict[str, Any]], accept: Accept | Sequence[str]) -> list[dict[str, Any]]:
    if isinstance(accept, str):
        if accept == "all":
            return [link for link in links if link["write_status"] in ("proposed", "accepted")]
        if accept == "proposed":
            return [link for link in links if link["write_status"] in ("proposed", "accepted")]
        if accept == "auto":
            return [link for link in links if link["write_status"] == "accepted"]
        raise ValueError(f"unknown accept mode {accept!r}")
    wanted = {str(x) for x in accept}
    return [link for link in links if str(link["link_id"]) in wanted]


def apply(
    run_id: UUID,
    *,
    store: ProvenanceStore,
    actor: str,
    irods_path: str,
    accept: Accept | Sequence[str] = "auto",
    mode: Mode = "local",
    ducklake: Any | None = None,
    zone: str = "local",
    target_type: str = "data_object",
    dry_run: bool = False,
    session: Any | None = None,
    auth_value: Any | None = None,
    outcome_tag: str | None = None,
) -> ApplyResult:
    """Write the accepted links of ``run_id`` for ``irods_path``.

    ``accept``: ``auto`` (links a cited threshold accepted), ``proposed`` (every proposal; the
    curator accepted the whole run), ``all`` (alias), or explicit link ids.
    """
    run = store.run(run_id)
    if run is None:
        raise KeyError(f"run {run_id} not found")
    links = store.links(run_id)
    chosen = _accepted(links, accept)
    result = ApplyResult(run_id=run_id, mode=mode, dry_run=dry_run)
    result.skipped = [link for link in links if link not in chosen]
    if not chosen:
        store.finish_run(
            run_id, "dry_run" if dry_run else "decided", write_mode=mode, irods_path=irods_path
        )
        return result
    if dry_run:
        result.written = chosen
        store.set_link_status(
            [UUID(str(link["link_id"])) for link in chosen], "dry_run", irods_path=irods_path
        )
        store.finish_run(run_id, "dry_run", write_mode=mode, irods_path=irods_path)
        return result

    tag = outcome_tag or (
        "human"
        if (isinstance(accept, str) and accept in ("proposed", "all"))
        or not isinstance(accept, str)
        else "auto"
    )
    source = f"{SOURCE_PREFIX}:apply:{tag}"
    now = datetime.now(tz=UTC)

    if mode == "irods":
        _write_irods(chosen, irods_path, target_type, session=session, auth_value=auth_value)
    written_ids = [UUID(str(link["link_id"])) for link in chosen]
    store.set_link_status(written_ids, "written", written_at=now, irods_path=irods_path)
    result.written = chosen

    if ducklake is None:
        store.finish_run(run_id, "applied", write_mode=mode, irods_path=irods_path)
        return result
    try:
        from mesa_ducklake import AvuChange

        project = ducklake.find_project_by_path(_project_root(irods_path, mode))
        if project is None:
            project = ducklake.register_project(
                irods_path=_project_root(irods_path, mode), actor=actor, zone=zone
            )
        changes = [
            AvuChange(
                irods_path=irods_path,
                target_type=target_type,
                attribute=link["attribute"],
                value=link["value"],
                unit=link["unit"] or "",
                op="add",
                actor=actor,
                source=source,
            )
            for link in chosen
        ]
        snapshot = ducklake.record_changes(
            project.project_id, actor, changes, note=f"mesa-anyjev run {run_id}", session=session
        )
        result.snapshot_id = int(snapshot.snapshot_id)
        result.project_id = str(project.project_id)
        store.link_snapshot(run_id, irods_path, project.project_id, int(snapshot.snapshot_id))
        store.finish_run(
            run_id, "applied", write_mode=mode, irods_path=irods_path, project_id=project.project_id
        )
    except Exception as exc:
        result.mirror_error = f"{type(exc).__name__}: {exc}"
        store.set_link_status(written_ids, "mirror_failed", written_at=now)
        store.finish_run(run_id, "partial", write_mode=mode, irods_path=irods_path)
    return result


def _project_root(irods_path: str, mode: Mode) -> str:
    """Local mode: the project is the card's parent collection under /local/mesa-anyjev."""
    parts = irods_path.rstrip("/").split("/")
    return "/".join(parts[:-1]) if mode == "local" and len(parts) > 2 else irods_path


def _write_irods(
    chosen: list[dict[str, Any]],
    irods_path: str,
    target_type: str,
    *,
    session: Any,
    auth_value: Any,
) -> None:
    if session is None or auth_value is None:
        raise RuntimeError(
            "irods mode needs an authenticated session and AuthValue (mesa-mcp tools, M4)"
        )
    from mesa_mcp.irods._avu_helpers import add_avu_to_irods, resolve_path_target
    from mesa_mcp.irods.access import assert_allowed

    norm = assert_allowed(irods_path, auth_value)
    target = resolve_path_target(session, norm, hint=target_type)
    for link in chosen:
        add_avu_to_irods(
            session,
            norm,
            target,
            {"attribute": link["attribute"], "value": link["value"], "unit": link["unit"] or ""},
        )

"""``mesa_decide_*``: mesa-anyjev's tools inside mesa-mcp's registry (DESIGN D14).

Importing this module registers the tools; mesa-mcp's ``mesa_mcp.tools`` entry point does that
import once the loader lands upstream (PR 1), and ``import mesa_anyjev.mcp_tools`` before
``MesaServer()`` does it today. The surface tag ``decision`` is written into ``ToolSpec.meta``
so the loader can publish it in ``_meta``.

* ``mesa_decide_annotate``: the decide phase on a dataset card (no writes).
* ``mesa_decide_apply``: the write phase. ``accept="proposed"`` asks the user to pick, one
  candidate group per round trip (MRTR ``term_choice:<group_id>``), with a state that carries
  ids only; on resume every label, probability and level comes from the sidecar (D8).
* ``mesa_decide_explain``, ``mesa_decide_feedback``, ``mesa_decide_health``.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import BaseModel, Field

from mesa_anyjev.cards import load_card, parse_card
from mesa_anyjev.config import Config, load_config
from mesa_anyjev.service import DeciderBusy, DecisionService

Handler = Callable[..., Any]
TOOL_SURFACE = "decision"
ELICIT_PREFIX = "term_choice:"
DEFAULT_ACTOR = "mesa-anyjev"

_service: DecisionService | None = None
_factory: Callable[[], DecisionService] | None = None


def set_service(service: DecisionService | None) -> None:
    """Tests and embedders install a ready service; ``None`` returns to lazy construction."""
    global _service
    _service = service


def get_service() -> DecisionService:
    global _service
    if _service is None:
        cfg: Config = load_config()
        _service = _factory() if _factory else DecisionService(cfg)
    return _service


# -- schemas ---------------------------------------------------------------------------------------
class AnnotateIn(BaseModel):
    card_text: str | None = Field(
        default=None, description="Dataset card markdown (neon-avu-eval card shape)."
    )
    card_path: str | None = Field(default=None, description="Local path to a dataset card.")
    actor: str = Field(default=DEFAULT_ACTOR, description="Who is deciding (iRODS user if any).")


class ProposalOut(BaseModel):
    link_id: str
    group_id: str | None
    attribute: str
    value: str
    unit: str
    outcome: str
    p: float | None
    level: str
    column_name: str | None = None
    site_code: str | None = None
    rationale: str = ""


class AnnotateOut(BaseModel):
    run_id: str
    card: str
    proposals: list[ProposalOut]
    n_rejected: int
    n_decisions: int
    n_prompts: int
    missing_labels: int
    degraded: bool
    outcomes: dict[str, int]
    seconds: float
    next_step: str


class ApplyIn(BaseModel):
    run_id: str
    irods_path: str = Field(description="Data object or collection the AVUs describe.")
    accept: Literal["auto", "proposed"] = Field(
        default="auto",
        description="auto: only links the policy or a human already accepted; proposed: ask "
        "the user to pick for every open candidate group first (one round trip each).",
    )
    dry_run: bool = Field(default=True, description="Report what would be written; write nothing.")
    target_type: Literal["data_object", "collection"] = "data_object"
    actor: str = DEFAULT_ACTOR


class ApplyOut(BaseModel):
    run_id: str
    mode: str
    dry_run: bool
    written: list[dict[str, str]]
    skipped: int
    snapshot_id: int | None
    project_id: str | None
    mirror_error: str | None
    picks_recorded: int


class ExplainIn(BaseModel):
    run_id: str | None = None
    irods_path: str | None = None
    limit: int = Field(default=100, ge=1, le=1000)


class ExplainOut(BaseModel):
    run: dict[str, Any] | None
    decisions: list[dict[str, Any]]
    links: list[dict[str, Any]]
    pending_groups: list[str]


class FeedbackIn(BaseModel):
    group_id: str
    action: Literal["pick", "reject", "decline"] = "pick"
    decision_id: str | None = Field(
        default=None,
        description="The chosen candidate (from mesa_decide_explain); omit for 'none of these'.",
    )
    actor: str = DEFAULT_ACTOR


class FeedbackOut(BaseModel):
    override_id: str
    labels_written: int
    accepted_link_id: str | None
    outcome: str


class HealthOut(BaseModel):
    ok: bool
    backend: str
    checks: list[dict[str, Any]]


# -- helpers ---------------------------------------------------------------------------------------
def _tool_error(code: str, message: str, **details: Any) -> Exception:
    from mesa_mcp.errors import ToolError

    err: Exception = ToolError(code=code, message=message, details=details)
    return err


def _uuid(value: str, what: str) -> UUID:
    try:
        return UUID(str(value))
    except ValueError as exc:
        raise _tool_error("invalid_argument", f"{what} is not a UUID: {value!r}") from exc


def _actor(args_actor: str, auth_value: Any) -> str:
    username = getattr(auth_value, "username", None)
    if username and not getattr(auth_value, "is_anonymous", lambda: False)():
        return str(username)
    return args_actor


def _slim(row: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "decision_id",
        "seq",
        "group_id",
        "question_id",
        "scope",
        "column_name",
        "site_code",
        "provider",
        "level",
        "calibration",
        "answer",
        "confidence",
        "p_true",
        "margin",
        "outcome",
        "model",
    )
    return {k: (str(row[k]) if isinstance(row.get(k), UUID) else row.get(k)) for k in keep}


def _link_out(link: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "link_id",
        "group_id",
        "decision_id",
        "attribute",
        "value",
        "unit",
        "term_curie",
        "column_name",
        "site_code",
        "write_status",
        "snapshot_id",
        "source",
    )
    return {k: (str(link[k]) if isinstance(link.get(k), UUID) else link.get(k)) for k in keep}


def _group_schema(cands: list[dict[str, Any]]) -> dict[str, Any]:
    """The MRTR form: enum of decision ids, names rendered as ``label (CURIE) p(fits)=0.83 L1``
    (plan amendment A11); an unranked group renders without p or level."""
    names = []
    for c in cands:
        base = f"{c['label']} ({c['curie']})" if c.get("curie") else str(c.get("label"))
        if c.get("p_true") is not None and c.get("level") not in (None, "none"):
            names.append(f"{base} p(fits)={float(c['p_true']):.2f} {c['level']}")
        else:
            names.append(f"{base} (unranked: decider unavailable)")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "decision_id": {
                "type": "string",
                "title": "Ontology term",
                "description": "The candidate to write; decline to leave the group unwritten.",
                "enum": [c["decision_id"] for c in cands],
                "enumNames": names,
            }
        },
        "required": ["decision_id"],
    }


def _response_content(response: Any) -> tuple[str | None, dict[str, Any] | None]:
    action = getattr(response, "action", None)
    if action is None and isinstance(response, dict):
        action = response.get("action")
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    return (str(action) if action else None), (content if isinstance(content, dict) else None)


# -- tools -----------------------------------------------------------------------------------------
def _register() -> None:
    from mesa_mcp.errors import InputRequired
    from mesa_mcp.server import get_tool
    from mesa_mcp.server import register_tool as _rt

    register_tool = cast("Callable[..., Callable[[Handler], Handler]]", _rt)

    @register_tool(
        "mesa_decide_annotate",
        "Decide which ontology terms describe a dataset card and propose AVUs, with a calibrated "
        "probability and an AnyJev level per decision, recorded in the mesa-anyjev sidecar. "
        "Writes nothing to iRODS; follow with mesa_decide_apply.",
        input_model=AnnotateIn,
        output_model=AnnotateOut,
    )
    async def mesa_decide_annotate(args: AnnotateIn, auth_value: Any = None) -> dict[str, Any]:
        if not args.card_text and not args.card_path:
            raise _tool_error("invalid_argument", "card_text or card_path is required")
        try:
            card = parse_card(args.card_text) if args.card_text else load_card(str(args.card_path))
        except (OSError, ValueError) as exc:
            raise _tool_error("invalid_argument", f"cannot read the card: {exc}") from exc
        service = get_service()
        try:
            run = service.annotate(card, actor=_actor(args.actor, auth_value))
        except DeciderBusy as exc:
            raise _tool_error("decider_unavailable", str(exc)) from exc
        proposals = [
            ProposalOut(
                link_id=str(p.link_id),
                group_id=str(p.group_id) if p.group_id else None,
                attribute=p.avu["attribute"],
                value=p.avu["value"],
                unit=p.avu["unit"],
                outcome=p.outcome,
                p=p.p,
                level=p.level,
                column_name=p.column_name,
                site_code=p.site_code,
                rationale=p.rationale,
            )
            for p in run.proposals
        ]
        return AnnotateOut(
            run_id=str(run.run_id),
            card=run.card.name,
            proposals=proposals,
            n_rejected=len(run.rejected),
            n_decisions=run.n_decisions,
            n_prompts=run.n_prompts,
            missing_labels=run.missing_labels,
            degraded=run.missing_labels > 0,
            outcomes=dict(run.outcomes),
            seconds=round(run.seconds, 2),
            next_step=(
                f"mesa_decide_apply(run_id={run.run_id!s}, irods_path=..., accept='proposed', "
                "dry_run=False) asks you to confirm each proposed term before writing"
            ),
        ).model_dump()

    @register_tool(
        "mesa_decide_apply",
        "Write the accepted AVUs of a mesa_decide_annotate run to an iRODS path and mirror them "
        "into the MESA DuckLake as one snapshot per run and path. accept='proposed' asks you to "
        "pick the term for each open candidate group first (one question per round trip). "
        "dry_run=True (default) reports without writing.",
        input_model=ApplyIn,
        output_model=ApplyOut,
    )
    async def mesa_decide_apply(
        args: ApplyIn, auth_value: Any = None, elicited: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        service = get_service()
        run_id = _uuid(args.run_id, "run_id")
        try:
            summary = service.run_summary(run_id)
        except KeyError as exc:
            raise _tool_error("not_found", str(exc)) from exc
        actor = _actor(args.actor, auth_value)
        state = (elicited or {}).get("state") or {}
        if elicited and (
            state.get("tool") != "mesa_decide_apply" or state.get("run_id") != args.run_id
        ):
            raise _tool_error("invalid_argument", "request state does not belong to this call")
        for key, response in ((elicited or {}).get("responses") or {}).items():
            if not str(key).startswith(ELICIT_PREFIX):
                continue
            group_id = _uuid(str(key)[len(ELICIT_PREFIX) :], "group_id")
            if str(group_id) not in {str(g) for g in summary["pending_groups"]}:
                raise _tool_error("invalid_argument", "answered group is not open in this run")
            action, content = _response_content(response)
            if action in ("decline", "cancel") or content is None:
                service.record_human_pick(
                    group_id,
                    actor=actor,
                    chosen_decision_id=None,
                    action="decline",
                    elicitation_key=str(key),
                )
                continue
            chosen = str(content.get("decision_id") or "")
            cands = service.candidates_for_group(group_id)
            if chosen not in {c["decision_id"] for c in cands}:
                raise _tool_error(
                    "invalid_argument", "selected candidate was not among the offered ones"
                )
            service.record_human_pick(
                group_id,
                actor=actor,
                chosen_decision_id=UUID(chosen),
                action="pick",
                elicitation_key=str(key),
            )
        if args.accept == "proposed":
            summary = service.run_summary(run_id)
            asked = {str(x) for x in state.get("asked", [])}
            for gid in summary["pending_groups"]:
                if str(gid) in asked:
                    continue
                cands = service.candidates_for_group(UUID(str(gid)))
                if not cands:
                    continue
                group = next(g for g in summary["groups"] if str(g["group_id"]) == str(gid))
                target = group.get("column_name") or group.get("site_code") or group["scope"]
                raise InputRequired(
                    message=f"Which {str(group.get('ontology_id') or '').upper()} term fits {target!r}?",
                    schema=_group_schema(cands),
                    state={
                        "tool": "mesa_decide_apply",
                        "run_id": args.run_id,
                        "asked": sorted(asked | {str(gid)}),
                    },
                    key=f"{ELICIT_PREFIX}{gid}",
                )
        session: Any = None
        ducklake: Any = None
        mode: Literal["local", "irods"] = "local"
        if auth_value is not None and not getattr(auth_value, "is_anonymous", lambda: True)():
            from mesa_mcp.ducklake.client import get_default_client
            from mesa_mcp.irods.client_pool import default_pool

            session = default_pool().get(auth_value)
            ducklake = get_default_client()
            mode = "irods"
        try:
            res = service.apply(
                run_id,
                actor=actor,
                irods_path=args.irods_path,
                accept="auto",
                mode=mode,
                ducklake=ducklake,
                zone=str(getattr(auth_value, "zone", None) or "local"),
                target_type=args.target_type,
                dry_run=args.dry_run,
                session=session,
                auth_value=auth_value,
                outcome_tag="human" if elicited else None,
            )
        except (RuntimeError, LookupError) as exc:
            raise _tool_error("invalid_argument", str(exc)) from exc
        picks = sum(1 for g in service.run_summary(run_id)["groups"] if g["outcome"] == "human")
        return ApplyOut(
            run_id=str(run_id),
            mode=res.mode,
            dry_run=res.dry_run,
            written=[
                {"attribute": w["attribute"], "value": w["value"], "unit": w["unit"] or ""}
                for w in res.written
            ],
            skipped=len(res.skipped),
            snapshot_id=res.snapshot_id,
            project_id=res.project_id,
            mirror_error=res.mirror_error,
            picks_recorded=picks,
        ).model_dump()

    @register_tool(
        "mesa_decide_explain",
        "Explain a mesa-anyjev run (every decision with its level, calibration and probability, "
        "the AVU links and their write status) or list the decisions behind the AVUs on an "
        "iRODS path.",
        input_model=ExplainIn,
        output_model=ExplainOut,
    )
    async def mesa_decide_explain(args: ExplainIn) -> dict[str, Any]:
        service = get_service()
        if not args.run_id and not args.irods_path:
            raise _tool_error("invalid_argument", "run_id or irods_path is required")
        run = decisions = links = None
        pending: list[str] = []
        if args.run_id:
            run_id = _uuid(args.run_id, "run_id")
            try:
                summary = service.run_summary(run_id)
            except KeyError as exc:
                raise _tool_error("not_found", str(exc)) from exc
            run = {
                k: (str(v) if isinstance(v, UUID) else v)
                for k, v in summary["run"].items()
                if k != "plan_json"
            }
            decisions = [_slim(d) for d in service.store.decisions(run_id)][: args.limit]
            links = [_link_out(link) for link in summary["links"]]
            pending = [str(g) for g in summary["pending_groups"]]
        else:
            decisions = [
                {k: (str(v) if isinstance(v, UUID) else v) for k, v in r.items()}
                for r in service.store.decisions_for_path(str(args.irods_path), limit=args.limit)
            ]
        return ExplainOut(
            run=run, decisions=decisions or [], links=links or [], pending_groups=pending
        ).model_dump(mode="json")

    @register_tool(
        "mesa_decide_feedback",
        "Record a curator's verdict on a candidate group: pick a candidate (decision_id from "
        "mesa_decide_explain), reject the group, or decline. Picks are authoritative, become "
        "labels for the next calibration, and accept the link for mesa_decide_apply.",
        input_model=FeedbackIn,
        output_model=FeedbackOut,
    )
    async def mesa_decide_feedback(args: FeedbackIn, auth_value: Any = None) -> dict[str, Any]:
        service = get_service()
        group_id = _uuid(args.group_id, "group_id")
        chosen = _uuid(args.decision_id, "decision_id") if args.decision_id else None
        try:
            out = service.record_human_pick(
                group_id,
                actor=_actor(args.actor, auth_value),
                chosen_decision_id=chosen,
                action=args.action,
            )
        except KeyError as exc:
            raise _tool_error("not_found", str(exc)) from exc
        except ValueError as exc:
            raise _tool_error("invalid_argument", str(exc)) from exc
        return FeedbackOut(**out).model_dump()

    @register_tool(
        "mesa_decide_health",
        "Check the mesa-anyjev decision service: questions lock, provenance store, backend "
        "and promoted artifacts (local weights are not loaded by this check).",
        output_model=HealthOut,
    )
    async def mesa_decide_health() -> dict[str, Any]:
        from mesa_anyjev.health import doctor

        cfg = get_service().cfg
        kind = cfg.backend.kind
        rep = doctor(cfg, backend_kind="fake" if kind in ("hf", "composite") else None)
        checks = [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in rep.checks]
        if kind in ("hf", "composite"):
            checks.append(
                {
                    "name": "backend",
                    "ok": True,
                    "detail": f"kind={kind}; run `mesa-anyjev doctor --backend {kind}` for the weights",
                }
            )
        return HealthOut(ok=rep.ok, backend=kind, checks=checks).model_dump()

    for name in (
        "mesa_decide_annotate",
        "mesa_decide_apply",
        "mesa_decide_explain",
        "mesa_decide_feedback",
        "mesa_decide_health",
    ):
        get_tool(name).meta["io.mesa/surface"] = TOOL_SURFACE


TOOL_NAMES = (
    "mesa_decide_annotate",
    "mesa_decide_apply",
    "mesa_decide_explain",
    "mesa_decide_feedback",
    "mesa_decide_health",
)

with contextlib.suppress(ValueError):  # already registered (module re-imported)
    _register()

__all__ = ["TOOL_NAMES", "TOOL_SURFACE", "get_service", "json", "set_service"]

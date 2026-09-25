"""The ``mesa-anyjev`` console script. Milestone M0 verbs: ``questions``, ``provenance
migrate``, ``doctor`` (config and lock checks only until M1 adds the backends)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mesa_anyjev import __version__
from mesa_anyjev.config import Config, load_config
from mesa_anyjev.policy import DEFAULTS_PATH

EXIT_OK, EXIT_FAIL, EXIT_CONFIG = 0, 1, 2


def _cmd_questions(args: argparse.Namespace, _cfg: Config) -> int:
    from mesa_anyjev import questions

    if args.update_lock:
        sha = questions.write_lock()
        print(f"questions.lock.json updated; lock_sha {sha}")
        return EXIT_OK
    drift = questions.lock_drift()
    if drift:
        for line in drift:
            print(f"DRIFT: {line}", file=sys.stderr)
        return EXIT_FAIL
    print(f"questions in sync; lock_sha {questions.lock_sha()}")
    return EXIT_OK


def _cmd_provenance(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_anyjev.provenance.migrate import apply_migrations

    if args.verb == "migrate":
        version = apply_migrations(args.dsn or cfg.provenance.dsn, target=args.target)
        print(f"provenance schema version {version} at {args.dsn or cfg.provenance.dsn}")
        return EXIT_OK
    print(f"provenance {args.verb}: not implemented until a later milestone", file=sys.stderr)
    return EXIT_FAIL


def _cmd_doctor(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_anyjev.health import doctor

    report = doctor(cfg, backend_kind=getattr(args, "backend", None))
    print("\n".join(report.lines()))
    return EXIT_OK if report.ok else EXIT_FAIL


def _make_collaborators(cfg: Config, args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    """(provider, planner, ols_layer, policy, store) from config plus flags."""
    from mesa_anyjev.backends.factory import make_backend
    from mesa_anyjev.ols import OLSLayer, RecordingOLS
    from mesa_anyjev.planner.static_planner import StaticPlanner
    from mesa_anyjev.policy import load_policy
    from mesa_anyjev.provenance.store import open_store
    from mesa_anyjev.providers.anyjev_provider import AnyJevProvider

    backend = make_backend(cfg.backend)
    provider = AnyJevProvider(
        backend,
        cfg.decider,
        served_model=cfg.backend.served_model if cfg.backend.kind == "gateway" else None,
    )
    planner: Any
    kind = getattr(args, "planner", None) or cfg.planner.kind
    if kind == "gateway":
        from mesa_anyjev.planner.gateway_planner import GatewayPlanner

        planner = GatewayPlanner(
            cfg.backend.gateway_base_url,
            cfg.backend.gateway_api_key,
            cfg.planner.gateway_model,
            cfg.planner.timeout,
        )
    elif kind == "claude":
        print("planner=claude arrives with milestone M2; using static", file=sys.stderr)
        planner = StaticPlanner()
    else:
        planner = StaticPlanner()
    inner: Any = None
    if cfg.ols.fixtures != "replay":  # replay is strictly offline; auto/record/off reach EBI OLS
        from mesa_mcp.ols.client import OLSClient

        inner = OLSClient(cfg.ols.base_url)
    client: Any = (
        RecordingOLS(inner, cfg.ols.fixtures_dir, cfg.ols.fixtures)
        if cfg.ols.fixtures != "off"
        else inner
    )
    ols = OLSLayer(client, max_candidates=cfg.policy.max_candidates)
    policy = load_policy(
        cfg.policy.defaults_file if Path(cfg.policy.defaults_file).is_absolute() else DEFAULTS_PATH
    )
    store = open_store(getattr(args, "provenance", None) or cfg.provenance.dsn)
    return provider, planner, ols, policy, store


def _cmd_plan(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_anyjev.cards import load_card

    _, planner, _, _, store = _make_collaborators(cfg, args)
    store.close()
    result = planner.plan(load_card(args.card))
    print(
        json.dumps(
            {
                "planner": result.planner,
                "model": result.model,
                "fallback": result.fallback,
                "usage": result.usage,
                "plan": result.plan.model_dump(),
            },
            indent=1,
        )
    )
    return EXIT_OK


def _cmd_annotate(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_anyjev.cards import load_card
    from mesa_anyjev.pipeline import Annotator

    provider, planner, ols, policy, store = _make_collaborators(cfg, args)
    try:
        run = Annotator(
            provider=provider,
            planner=planner,
            ols=ols,
            policy=policy,
            store=store,
            cfg=cfg,
            actor=args.actor,
        ).annotate(load_card(args.card))
    finally:
        store.close()
    result = run.to_eval_result()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8"
        )
    print(
        f"run {run.run_id}: {len(run.proposals)} proposals, {len(run.rejected)} rejected, {run.n_decisions} decisions, "
        f"{run.n_prompts} prompts, missing_labels={run.missing_labels}, {run.seconds:.1f}s; outcomes {run.outcomes}"
    )
    for p in run.proposals:
        print(
            f"  {p.outcome:9s} {p.avu['attribute']} = {p.avu['value']!r} [{p.avu['unit']}] {p.rationale}"
        )
    return EXIT_OK


def _cmd_apply(args: argparse.Namespace, cfg: Config) -> int:
    from uuid import UUID

    from mesa_anyjev.apply import apply
    from mesa_anyjev.provenance.store import open_store

    store = open_store(args.provenance or cfg.provenance.dsn)
    ducklake = None
    dsn = args.local_ducklake or cfg.ducklake.catalog_dsn
    if dsn and not args.no_ducklake:
        from mesa_ducklake import DuckLakeClient

        cache = Path(cfg.ducklake.cache_dir).expanduser() if cfg.ducklake.cache_dir else None
        ducklake = DuckLakeClient(
            catalog_dsn=_expand_dsn(dsn), irods_session=None, cache_dir=cache, cache_cap_bytes=0
        )
    accept: Any = (
        args.accept if args.accept in ("auto", "proposed", "all") else args.accept.split(",")
    )
    try:
        res = apply(
            UUID(args.run_id),
            store=store,
            actor=args.actor,
            irods_path=args.irods_path,
            accept=accept,
            mode="local",
            ducklake=ducklake,
            zone=cfg.ducklake.local_zone,
            dry_run=args.dry_run,
        )
    finally:
        store.close()
        if ducklake is not None:
            ducklake.close()
    summary = (
        f"run {res.run_id}: {len(res.written)} written, {len(res.skipped)} skipped, "
        f"snapshot_id={res.snapshot_id}, project_id={res.project_id}, dry_run={res.dry_run}"
    )
    if res.mirror_error:
        summary += f", mirror_error={res.mirror_error}"
    print(summary)
    return EXIT_OK if res.mirror_error is None else EXIT_FAIL


def _cmd_explain(args: argparse.Namespace, cfg: Config) -> int:
    from uuid import UUID

    from mesa_anyjev.provenance.store import open_store

    store = open_store(args.provenance or cfg.provenance.dsn)
    try:
        if args.run_id:
            rows = store.decisions(UUID(args.run_id))
            for r in rows:
                target = str(r["column_name"] or r["site_code"] or "")
                stat = r["p_true"] if r["p_true"] is not None else r["confidence"]
                print(
                    f"{r['seq']:4d} {r['question_id']:22s} {r['scope']:7s} {target:28s} "
                    f"{r['level']:4s} {r['calibration']:8s} p={stat} {r['outcome']}"
                )
        if args.path:
            for r in store.decisions_for_path(args.path):
                print(json.dumps(r, default=str))
    finally:
        store.close()
    return EXIT_OK


def _expand_dsn(dsn: str) -> str:
    if dsn.startswith("duckdb:///~"):
        return "duckdb:///" + str(Path(dsn[len("duckdb:///") :]).expanduser())
    return dsn


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mesa-anyjev", description="Declarative, calibrated MESA decisions."
    )
    p.add_argument("--version", action="version", version=f"mesa-anyjev {__version__}")
    p.add_argument("--config", help="YAML config file (env MESA_ANYJEV_* and flags override it)")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("questions", help="check or update questions.lock.json")
    g = q.add_mutually_exclusive_group()
    g.add_argument(
        "--check", action="store_true", help="fail if any Question.key drifted (default)"
    )
    g.add_argument("--update-lock", action="store_true", help="rewrite the lock from the code")
    q.set_defaults(func=_cmd_questions)

    prov = sub.add_parser("provenance", help="sidecar schema management")
    prov.add_argument("verb", choices=["migrate", "export", "reconcile"])
    prov.add_argument("--dsn", help="duckdb:///path or postgresql://... (default: config)")
    prov.add_argument("--target", type=int, help="apply migrations up to this version")
    prov.set_defaults(func=_cmd_provenance)

    d = sub.add_parser("doctor", help="what this host can reach")
    d.add_argument(
        "--backend",
        choices=["fake", "gateway", "hf", "composite"],
        help="override backend.kind for the checks",
    )
    d.set_defaults(func=_cmd_doctor)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--provenance", help="sidecar DSN (default: config)")
    common.add_argument(
        "--actor",
        default=os.environ.get("USER", "mesa-anyjev"),
        help="actor recorded on runs and AVUs",
    )

    pl = sub.add_parser(
        "plan", parents=[common], help="run the planner on a card and print the plan"
    )
    pl.add_argument("--card", required=True)
    pl.add_argument("--planner", choices=["static", "gateway", "claude"])
    pl.set_defaults(func=_cmd_plan)

    an = sub.add_parser(
        "annotate", parents=[common], help="decide and propose AVUs for a dataset card (no writes)"
    )
    an.add_argument("--card", required=True)
    an.add_argument(
        "--backend", choices=["fake", "gateway", "hf", "composite"], help="override backend.kind"
    )
    an.add_argument("--planner", choices=["static", "gateway", "claude"])
    an.add_argument(
        "--level", choices=["raw", "L0", "L1", "L2", "auto"], help="override decider.level"
    )
    an.add_argument("--out", help="write the run in the neon-avu-eval result shape")
    an.set_defaults(func=_cmd_annotate)

    ap = sub.add_parser(
        "apply",
        parents=[common],
        help="write the accepted proposals of a run (local DuckLake mode)",
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--irods-path", required=True, help="the (local) path the AVUs are bound to")
    ap.add_argument("--accept", default="auto", help="auto | proposed | all | <link_id,...>")
    ap.add_argument(
        "--local-ducklake",
        help="duckdb:///... DuckLake catalog (default: config.ducklake.catalog_dsn)",
    )
    ap.add_argument(
        "--no-ducklake", action="store_true", help="skip the DuckLake snapshot (provenance only)"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.set_defaults(func=_cmd_apply)

    ex = sub.add_parser(
        "explain",
        parents=[common],
        help="decisions of a run, or the decisions behind a path's AVUs",
    )
    ex.add_argument("--run-id")
    ex.add_argument("--path")
    ex.set_defaults(func=_cmd_explain)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    if getattr(args, "backend", None) and args.cmd == "annotate":
        overrides.setdefault("backend", {})["kind"] = args.backend
    if getattr(args, "level", None):
        overrides.setdefault("decider", {})["level"] = args.level
    if getattr(args, "planner", None):
        overrides.setdefault("planner", {})["kind"] = args.planner
    try:
        cfg = load_config(args.config, flag_overrides=overrides)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    result: int = args.func(args, cfg)
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

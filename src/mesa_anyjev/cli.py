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


def _make_collaborators(
    cfg: Config, args: argparse.Namespace, *, store: Any | None = None
) -> tuple[Any, Any, Any, Any, Any]:
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
        from mesa_anyjev.planner.claude_planner import ClaudePlanner

        planner = ClaudePlanner(cfg.planner)
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
    if store is None:
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


def _artifact_store(cfg: Config) -> Any:
    from mesa_anyjev.artifacts import ArtifactStore
    from mesa_anyjev.questions import lock_sha

    return ArtifactStore(
        cfg.artifacts.dir,
        cfg.backend.canonical_model if cfg.backend.kind != "fake" else "fake",
        lock_sha(),
        strict=cfg.artifacts.strict,
    )


def _cmd_learn(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_anyjev.provenance.store import open_store

    store = open_store(args.provenance or cfg.provenance.dsn)
    try:
        if args.verb == "ingest":
            from mesa_anyjev.learn.labels import TermResolver, ingest_neon_eval
            from mesa_anyjev.ols import RecordingOLS

            root = args.eval_root or cfg.eval_root
            if not root:
                print("learn ingest needs --eval-root or MESA_ANYJEV_EVAL_ROOT", file=sys.stderr)
                return EXIT_CONFIG
            inner: Any = None
            if cfg.ols.fixtures != "replay":
                from mesa_mcp.ols.client import OLSClient

                inner = OLSClient(cfg.ols.base_url)
            client: Any = (
                RecordingOLS(inner, cfg.ols.fixtures_dir, cfg.ols.fixtures)
                if cfg.ols.fixtures != "off"
                else inner
            )
            report = ingest_neon_eval(store, root, TermResolver(client))
            print(json.dumps(report.summary(), indent=1))
            if report.terms_missing:
                print(
                    f"unresolved CURIEs ({len(report.terms_missing)}): {', '.join(report.terms_missing[:20])}",
                    file=sys.stderr,
                )
            return EXIT_OK
        if args.verb == "fit":
            from mesa_anyjev.learn.fit import fit_question

            provider, _, _, policy, _ = _make_collaborators(cfg, args, store=store)
            fit_report = fit_question(
                provider,
                store,
                _artifact_store(cfg),
                args.question,
                policy=policy,
                level=args.level,
                loco=not args.no_loco,
                backend_kind=cfg.backend.kind,
            )
            out = {k: v for k, v in fit_report.__dict__.items() if k != "folds"}
            out["folds"] = {
                c: {k: f.get(k) for k in ("n", "acc", "ece", "cov@5%", "n_neg", "prior_frozen")}
                for c, f in fit_report.folds.items()
            }
            print(json.dumps(out, indent=1, default=str))
            return EXIT_OK if fit_report.status == "fitted" else EXIT_FAIL
        if args.verb == "promote":
            from mesa_anyjev.learn.fit import promote

            ok, msg = promote(_artifact_store(cfg), args.version, args.question)
            print(msg)
            return EXIT_OK if ok else EXIT_FAIL
    finally:
        store.close()
    return EXIT_FAIL


def _cmd_bench(args: argparse.Namespace, cfg: Config) -> int:
    from datetime import UTC, datetime

    from mesa_anyjev.bench.run import environment, markdown_table, run_task, write_results
    from mesa_anyjev.bench.tasks.neon import tasks_from_store
    from mesa_anyjev.provenance.store import open_store

    if args.verb == "table":
        payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
        print(markdown_table(payload))
        return EXIT_OK
    store = open_store(args.provenance or cfg.provenance.dsn, read_only=True)
    try:
        tasks = tasks_from_store(store)
        provider, _, _, _, _ = _make_collaborators(cfg, args, store=store)
    finally:
        store.close()
    wanted = args.tasks.split(",") if args.tasks else sorted(tasks)
    missing = [t for t in wanted if t not in tasks]
    if missing:
        print(f"unknown or empty tasks: {missing}; available: {sorted(tasks)}", file=sys.stderr)
        return EXIT_FAIL
    levels = args.levels.split(",")
    results = []
    for name in wanted:
        print(
            f"== {name}: {len(tasks[name].items)} items",
            file=sys.stderr,
        )
        results.append(
            run_task(
                provider, tasks[name], levels, loco=not args.no_loco, flip_probe=not args.no_flip
            )
        )
    date = args.date or datetime.now(tz=UTC).strftime("%Y-%m-%d")
    slug = (cfg.backend.canonical_model if cfg.backend.kind != "fake" else "fake").replace(
        "/", "__"
    )
    path = write_results(
        results,
        args.out,
        slug,
        cfg.backend.kind,
        environment(provider, cfg.backend.kind),
        date=date,
    )
    print(path)
    print(markdown_table(json.loads(path.read_text(encoding="utf-8"))))
    return EXIT_OK


def _cmd_artifacts(args: argparse.Namespace, cfg: Config) -> int:
    store = _artifact_store(cfg)
    current = store.current()
    for v in store.versions():
        m = v.manifest
        mark = "*" if current and current.version == v.version else " "
        val = m.get("validation", {})
        loco = {q: val[q].get("loco") for q in val}
        print(f"{mark} v{v.version}  {v.path}")
        print(f"    fitted_on={m.get('fitted_on')} questions={list(m.get('per_question', {}))}")
        print(f"    loco={loco}")
    if not store.versions():
        print(f"no artifacts under {store.dir}")
    return EXIT_OK


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

    le = sub.add_parser("learn", parents=[common], help="labels in, calibrated artifacts out")
    le.add_argument("verb", choices=["ingest", "fit", "promote"])
    le.add_argument("--eval-root", help="neon-avu-eval checkout (ingest)")
    le.add_argument("--question", default="term.fits", help="question id (fit, promote)")
    le.add_argument("--level", choices=["L1", "L2"], default="L1")
    le.add_argument(
        "--no-loco", action="store_true", help="skip leave-one-card-out (exploratory only)"
    )
    le.add_argument("--version", type=int, help="artifact version (promote)")
    le.add_argument(
        "--backend", choices=["fake", "gateway", "hf", "composite"], help="override backend.kind"
    )
    le.set_defaults(func=_cmd_learn)

    be = sub.add_parser("bench", parents=[common], help="run the bench or render a results table")
    be.add_argument("verb", choices=["run", "table"])
    be.add_argument("--tasks", help="comma-separated task names (default: all with items)")
    be.add_argument("--levels", default="raw,L0,L1", help="comma-separated levels")
    be.add_argument("--no-loco", action="store_true")
    be.add_argument("--no-flip", action="store_true")
    be.add_argument("--out", default="bench/results")
    be.add_argument("--date", help="results directory name (default: today, UTC)")
    be.add_argument("--results", help="results JSON (table)")
    be.add_argument(
        "--backend", choices=["fake", "gateway", "hf", "composite"], help="override backend.kind"
    )
    be.set_defaults(func=_cmd_bench)

    ar = sub.add_parser(
        "artifacts", parents=[common], help="list artifact bundles for the configured model"
    )
    ar.add_argument("verb", choices=["list"], nargs="?", default="list")
    ar.add_argument(
        "--backend", choices=["fake", "gateway", "hf", "composite"], help="override backend.kind"
    )
    ar.set_defaults(func=_cmd_artifacts)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    if getattr(args, "backend", None) and args.cmd in ("annotate", "learn", "bench", "artifacts"):
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

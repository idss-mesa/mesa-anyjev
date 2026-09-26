"""The ``mesa-anyjev`` console script. Milestone M0 verbs: ``questions``, ``provenance
migrate``, ``doctor`` (config and lock checks only until M1 adds the backends)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mesa_anyjev import __version__
from mesa_anyjev.config import Config, load_config

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


def _cmd_feedback(args: argparse.Namespace, cfg: Config) -> int:
    from uuid import UUID

    from mesa_anyjev.provenance.store import open_store
    from mesa_anyjev.service import DecisionService

    store = open_store(args.provenance or cfg.provenance.dsn)
    try:
        service = DecisionService(
            cfg, collaborators=(None, None, None, None, store)
        )  # feedback needs no model: candidates and labels come from the sidecar
        out = service.record_human_pick(
            UUID(args.group_id),
            actor=args.actor,
            chosen_decision_id=UUID(args.decision_id) if args.decision_id else None,
            action=args.action,
        )
    except (KeyError, ValueError) as exc:
        print(f"feedback: {exc}", file=sys.stderr)
        return EXIT_FAIL
    finally:
        store.close()
    print(json.dumps(out))
    return EXIT_OK


def _cmd_hosted(args: argparse.Namespace, cfg: Config) -> int:
    """``hosted score``: MotherDuck prompt_jev over stored states (D16: policy-gated)."""
    from uuid import UUID

    from mesa_anyjev.learn.hosted import batch_from_labels, batch_from_run, score_batch, summarize
    from mesa_anyjev.policy import DEFAULTS_PATH, load_policy
    from mesa_anyjev.provenance.store import open_store
    from mesa_anyjev.providers.motherduck_provider import HostedJevProvider, MotherDuckRunner
    from mesa_anyjev.service import HostedDisabled, require_hosted

    try:
        require_hosted(cfg, local_source=True)
    except HostedDisabled as exc:
        print(f"hosted_disabled: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    store = open_store(args.provenance or cfg.provenance.dsn)
    try:
        questions = [q.strip() for q in args.question.split(",") if q.strip()]
        policy = load_policy(DEFAULTS_PATH)
        batches = []
        for qid in questions:
            if args.source == "labels":
                batches.append(
                    batch_from_labels(
                        store, qid, min_weight=policy.thresholds(qid).min_weight, limit=args.limit
                    )
                )
            else:
                if not args.run_id:
                    print("--source run needs --run-id", file=sys.stderr)
                    return EXIT_CONFIG
                batches.append(batch_from_run(store, UUID(args.run_id), qid, limit=args.limit))
        runner = MotherDuckRunner(cfg.motherduck)
        provider = HostedJevProvider(cfg.motherduck, runner)
        reports = [
            score_batch(
                provider,
                b,
                store=store,
                cfg=cfg,
                actor=args.actor,
                out_dir=args.out,
                date=args.date,
                dry_run=args.dry_run,
            )
            for b in batches
        ]
        print(json.dumps(summarize(reports), indent=1, default=str))
        if not args.dry_run:
            runner.close()
        return EXIT_OK
    finally:
        store.close()


def _cmd_provenance(args: argparse.Namespace, cfg: Config) -> int:
    from uuid import UUID

    from mesa_anyjev.provenance.migrate import apply_migrations

    dsn = args.dsn or cfg.provenance.dsn
    if args.verb == "migrate":
        version = apply_migrations(dsn, target=args.target)
        print(f"provenance schema version {version} at {dsn}")
        return EXIT_OK
    from mesa_anyjev.provenance.export import export_run, reconcile
    from mesa_anyjev.provenance.store import open_store

    if not args.run_id:
        print("provenance export|reconcile need --run-id", file=sys.stderr)
        return EXIT_CONFIG
    store = open_store(dsn)
    try:
        if args.verb == "export":
            written = export_run(store, UUID(args.run_id), args.out)
            for table, path in written.items():
                print(f"{table:16s} {path}")
            return EXIT_OK
        if not args.local_ducklake:
            print("provenance reconcile needs --local-ducklake <dsn>", file=sys.stderr)
            return EXIT_CONFIG
        from mesa_ducklake import DuckLakeClient

        cache = Path(cfg.ducklake.cache_dir).expanduser() if cfg.ducklake.cache_dir else None
        ducklake = DuckLakeClient(
            catalog_dsn=args.local_ducklake, irods_session=None, cache_dir=cache, cache_cap_bytes=0
        )
        try:
            fixed = reconcile(store, UUID(args.run_id), ducklake)
        finally:
            ducklake.close()
        print(f"reconciled {fixed} link(s) for run {args.run_id}")
        return EXIT_OK
    finally:
        store.close()


def _cmd_doctor(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_anyjev.health import doctor

    report = doctor(cfg, backend_kind=getattr(args, "backend", None))
    print("\n".join(report.lines()))
    return EXIT_OK if report.ok else EXIT_FAIL


def _make_collaborators(
    cfg: Config, args: argparse.Namespace, *, store: Any | None = None
) -> tuple[Any, Any, Any, Any, Any]:
    """(provider, planner, ols_layer, policy, store) from config plus flags (service.py)."""
    from mesa_anyjev.service import build_collaborators

    return build_collaborators(
        cfg,
        planner_kind=getattr(args, "planner", None),
        provenance_dsn=getattr(args, "provenance", None),
        store=store,
    )


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
    second: Any = None
    if args.second_opinion or cfg.claude.second_opinion:
        from mesa_anyjev.providers.claude_provider import ClaudeStructuredProvider

        second = ClaudeStructuredProvider(cfg.claude)
    try:
        run = Annotator(
            provider=provider,
            planner=planner,
            ols=ols,
            policy=policy,
            store=store,
            cfg=cfg,
            actor=args.actor,
            second_opinion=second,
        ).annotate(load_card(args.card))
    finally:
        store.close()
    result = run.to_eval_result()
    if run.audit:
        print(
            f"planner audit: agree={run.audit['agree']} planner_only={run.audit['planner_only']} "
            f"model_only={run.audit['model_only']}"
        )
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


def _cmd_datacite(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_anyjev.cards import load_card
    from mesa_anyjev.datacite import card_record, classify

    if not args.card and not (args.vocabulary and args.text):
        print("datacite needs --card or --vocabulary with --text", file=sys.stderr)
        return EXIT_CONFIG
    provider, _planner, _ols, policy, store = _make_collaborators(cfg, args)
    profile = policy.profile(cfg.policy.profile)
    try:
        if args.card:
            out = card_record(provider, load_card(args.card), policy=policy, profile=profile)
        else:
            out = classify(
                provider, args.vocabulary, args.text, policy=policy, profile=profile
            ).as_dict()
    finally:
        store.close()
    print(json.dumps(out, indent=1, default=str))
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
        cfg.artifacts.dir, _model_for(cfg), lock_sha(), strict=cfg.artifacts.strict
    )


def _model_for(cfg: Config) -> str:
    from mesa_anyjev.service import model_for

    return model_for(cfg)


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


def _cmd_bench_e2e(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_anyjev.bench.e2e import consensus_from_validated, run_e2e
    from mesa_anyjev.pipeline import Annotator

    root = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    cards = (
        [Path(c) for c in args.cards.split(",")]
        if args.cards
        else sorted((root / "cards").glob("*.md"))
    )
    validated = (
        Path(args.validated)
        if args.validated
        else root / "neon-avu-eval" / "results" / "validated.json"
    )
    consensus = consensus_from_validated(validated) if validated.exists() else None
    provider, _p, ols, policy, store = _make_collaborators(cfg, args)

    def make(planner_kind: str) -> Annotator:
        from mesa_anyjev.service import build_collaborators

        _prov, planner, _o, _pol, _s = build_collaborators(
            cfg, planner_kind=planner_kind, store=store
        )
        return Annotator(
            provider=provider,
            planner=planner,
            ols=ols,
            policy=policy,
            store=store,
            cfg=cfg,
            actor=args.actor,
        )

    try:
        result = run_e2e(
            make, cards, planners=args.planners.split(","), reps=args.reps, consensus=consensus
        )
    finally:
        store.close()
    date = args.date or datetime.now(tz=UTC).strftime("%Y-%m-%d")
    out = (
        Path(args.out) / date / f"{_model_for(cfg).replace('/', '__')}.{cfg.backend.kind}.e2e.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    for planner, r in result["planners"].items():
        print(
            f"{planner:8s} rep_agreement={r['rep_agreement_mean']} "
            f"consensus_all_recall={r['consensus_all_recall_mean']} "
            f"consensus_majority_recall={r['consensus_majority_recall_mean']}"
        )
    print(f"wrote {out}")
    return EXIT_OK


def _cmd_bench(args: argparse.Namespace, cfg: Config) -> int:
    if args.verb == "e2e":
        return _cmd_bench_e2e(args, cfg)
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
    slug = _model_for(cfg).replace("/", "__")
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
        inherited = store.current()
        if inherited is not None:
            m = inherited.manifest
            print(
                f"no bundle under {store.dir}; inheriting {inherited.path} (D24: every question key still exists)"
            )
            print(f"    fitted_on={m.get('fitted_on')} questions={list(m.get('per_question', {}))}")
        else:
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
    prov.add_argument("--run-id", help="run to export or reconcile")
    prov.add_argument("--out", default=".mesa/anyjev", help="export directory (export)")
    prov.add_argument("--local-ducklake", help="DuckLake catalog DSN (reconcile)")
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
    an.add_argument(
        "--second-opinion",
        action="store_true",
        help="ask Claude the term question over each proposed group (recorded; disagreement escalates)",
    )
    an.set_defaults(func=_cmd_annotate)

    dc = sub.add_parser(
        "datacite", parents=[common], help="DataCite vocabulary decisions for a card or a text"
    )
    dc.add_argument("--card", help="dataset card: resource type and description type")
    dc.add_argument(
        "--vocabulary",
        choices=[
            "ResourceTypeGeneral",
            "ContributorType",
            "RelationType",
            "DateType",
            "DescriptionType",
        ],
    )
    dc.add_argument("--text", help="the text to classify with --vocabulary")
    dc.add_argument(
        "--backend", choices=["fake", "gateway", "hf", "composite"], help="override backend.kind"
    )
    dc.set_defaults(func=_cmd_datacite)

    fb = sub.add_parser("feedback", parents=[common], help="record a curator's pick on a group")
    fb.add_argument("--group-id", required=True)
    fb.add_argument("--decision-id", help="the chosen candidate; omit with --action reject")
    fb.add_argument("--action", choices=["pick", "reject", "decline"], default="pick")
    fb.set_defaults(func=_cmd_feedback)

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
    be.add_argument("verb", choices=["run", "table", "e2e"])
    be.add_argument("--cards", help="comma-separated card paths (e2e; default: the fixture cards)")
    be.add_argument("--planners", default="static", help="comma-separated planner kinds (e2e)")
    be.add_argument("--reps", type=int, default=2, help="repetitions per card (e2e)")
    be.add_argument("--validated", help="neon-avu-eval validated.json for consensus recall (e2e)")
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

    ho = sub.add_parser(
        "hosted",
        parents=[common],
        help="hosted Jev on MotherDuck (policy-gated; data leaves the host)",
    )
    ho.add_argument("verb", choices=["score"])
    ho.add_argument("--question", default="term.fits", help="comma-separated question ids")
    ho.add_argument("--source", choices=["labels", "run"], default="labels")
    ho.add_argument("--run-id", help="sidecar run to re-score (--source run)")
    ho.add_argument("--limit", type=int, help="at most this many states per question")
    ho.add_argument("--out", default="bench/results", help="results directory")
    ho.add_argument("--date", help="results directory name (default: today, UTC)")
    ho.add_argument(
        "--dry-run",
        action="store_true",
        help="estimate rows, requests and input size; send nothing",
    )
    ho.set_defaults(func=_cmd_hosted)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    if getattr(args, "backend", None) and args.cmd in (
        "annotate",
        "learn",
        "bench",
        "artifacts",
        "datacite",
    ):
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

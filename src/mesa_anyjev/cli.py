"""The ``mesa-anyjev`` console script. Milestone M0 verbs: ``questions``, ``provenance
migrate``, ``doctor`` (config and lock checks only until M1 adds the backends)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

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


def _cmd_provenance(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_anyjev.provenance.migrate import apply_migrations

    if args.verb == "migrate":
        version = apply_migrations(args.dsn or cfg.provenance.dsn, target=args.target)
        print(f"provenance schema version {version} at {args.dsn or cfg.provenance.dsn}")
        return EXIT_OK
    print(f"provenance {args.verb}: not implemented until a later milestone", file=sys.stderr)
    return EXIT_FAIL


def _cmd_doctor(_args: argparse.Namespace, cfg: Config) -> int:
    from mesa_anyjev import questions

    ok = True
    drift = questions.lock_drift()
    print(f"[{'ok' if not drift else 'FAIL'}] questions lock ({questions.lock_sha()[:12]})")
    ok &= not drift
    print(
        f"[ok] backend.kind={cfg.backend.kind} planner.kind={cfg.planner.kind} policy.profile={cfg.policy.profile}"
    )
    print(f"[ok] provenance.dsn={cfg.provenance.dsn}")
    print("[--] backend, gateway, artifacts and hosted checks arrive with milestone M1+")
    return EXIT_OK if ok else EXIT_FAIL


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
    d.set_defaults(func=_cmd_doctor)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    result: int = args.func(args, cfg)
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

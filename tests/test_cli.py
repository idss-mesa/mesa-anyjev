from __future__ import annotations

from pathlib import Path

from mesa_anyjev.cli import main


def test_questions_check_and_doctor(capsys: object) -> None:
    assert main(["questions", "--check"]) == 0
    assert main(["doctor"]) == 0


def test_provenance_migrate_duckdb(tmp_path: Path) -> None:
    assert main(["provenance", "migrate", "--dsn", f"duckdb:///{tmp_path / 'p.duckdb'}"]) == 0


def test_bad_config_is_exit_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not a mapping\n", encoding="utf-8")
    assert main(["--config", str(bad), "doctor"]) == 2


def test_learn_bench_artifacts_verbs(tmp_path: Path) -> None:
    from pathlib import Path as _P

    root = _P(__file__).resolve().parent
    eval_root = _P("/home/tswetnam/github/idss-mesa/neon-avu-eval")
    if not (eval_root / "results" / "validated.json").exists() or not any(
        (root / "fixtures" / "ols").glob("*.json")
    ):
        import pytest

        pytest.skip("needs the neon-avu-eval checkout and recorded OLS fixtures")
    dsn = f"duckdb:///{tmp_path / 'labels.duckdb'}"
    env = {
        "MESA_ANYJEV_OLS__FIXTURES": "replay",
        "MESA_ANYJEV_ARTIFACTS__DIR": str(tmp_path / "art"),
        "MESA_ANYJEV_POLICY__PROFILE": "dev",
    }
    import os

    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        assert main(["learn", "ingest", "--eval-root", str(eval_root), "--provenance", dsn]) == 0
        assert (
            main(
                [
                    "bench",
                    "run",
                    "--backend",
                    "fake",
                    "--tasks",
                    "neon_annotate",
                    "--levels",
                    "raw,L0",
                    "--no-loco",
                    "--out",
                    str(tmp_path / "res"),
                    "--date",
                    "2026-01-01",
                    "--provenance",
                    dsn,
                ]
            )
            == 0
        )
        assert (tmp_path / "res" / "2026-01-01" / "fake.fake.json").exists()
        assert (
            main(
                [
                    "bench",
                    "table",
                    "--results",
                    str(tmp_path / "res" / "2026-01-01" / "fake.fake.json"),
                ]
            )
            == 0
        )
        assert (
            main(["learn", "fit", "--question", "term.fits", "--level", "L1", "--provenance", dsn])
            == 0
        )
        assert main(["learn", "promote", "--question", "term.fits", "--version", "1"]) == 0
        assert main(["artifacts"]) == 0
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

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

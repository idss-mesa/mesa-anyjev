"""bench e2e on the fake backend: rep agreement and consensus recall per planner."""

from __future__ import annotations

from pathlib import Path

from mesa_anyjev.bench.e2e import consensus_from_validated, jaccard, run_e2e
from mesa_anyjev.cli import main

ROOT = Path(__file__).resolve().parent


def test_consensus_and_e2e(tmp_path: Path) -> None:
    validated = ROOT / "fixtures" / "neon-avu-eval" / "results" / "validated.json"
    cons = consensus_from_validated(validated)
    assert "DP1.10022.001.bet_sorting" in cons and cons["DP1.10022.001.bet_sorting"]["majority"]
    assert cons["DP1.10022.001.bet_sorting"]["all"] <= cons["DP1.10022.001.bet_sorting"]["majority"]
    assert jaccard(set(), set()) == 1.0 and jaccard({1}, {2}) == 0.0
    dsn = f"duckdb:///{tmp_path / 'p.duckdb'}"
    card = str(ROOT / "fixtures" / "cards" / "DP1.10022.001.bet_sorting.md")
    assert (
        main(
            [
                "bench",
                "e2e",
                "--provenance",
                dsn,
                "--cards",
                card,
                "--reps",
                "2",
                "--planners",
                "static",
                "--out",
                str(tmp_path / "res"),
                "--date",
                "2026-09-25",
            ]
        )
        == 0
    )
    import json

    out = json.loads(next((tmp_path / "res" / "2026-09-25").glob("*.e2e.json")).read_text())
    entry = out["planners"]["static"]["cards"]["DP1.10022.001.bet_sorting"]
    assert 0.0 <= entry["rep_agreement"] <= 1.0 and len(entry["n_proposals"]) == 2
    assert entry["consensus_majority_recall"] is not None and "audit" in entry
    assert run_e2e.__doc__

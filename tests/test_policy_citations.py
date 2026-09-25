"""DESIGN D6 / amendment A2: numeric auto thresholds cite a LOCO bench cell with >= 30 negatives."""

from __future__ import annotations

import json
import re
from pathlib import Path

from mesa_anyjev.policy import DEFAULTS_PATH, load_policy
from mesa_anyjev.questions import QUESTIONS

ROOT = DEFAULTS_PATH.parent
CITE = re.compile(r"^(bench/results/\d{4}-\d{2}-\d{2}/[\w.\-]+\.json)#(.+)$")
MIN_NEGATIVES = 30


def test_every_question_has_thresholds_with_the_right_stat() -> None:
    policy = load_policy()
    for qid, spec in QUESTIONS.items():
        t = policy.thresholds(qid)
        expected = "p_true" if spec.question.kind == "noul" else "confidence"
        assert t.stat == expected, f"{qid}: noul thresholds read p_true, never confidence (A7)"
        assert 0.0 <= t.propose <= 1.0


def test_numeric_auto_thresholds_cite_a_loco_cell_with_negatives() -> None:
    policy = load_policy()
    for qid, t in policy.questions.items():
        if t.auto is None:
            continue
        m = CITE.match(t.cite)
        assert m, f"{qid}: auto={t.auto} must cite bench/results/<date>/<file>.json#<cell> (D6)"
        path = ROOT / m.group(1)
        assert path.exists(), f"{qid}: cited results file {path} does not exist"
        results = json.loads(path.read_text(encoding="utf-8"))
        cell = results
        for part in m.group(2).split("."):
            cell = cell[part]
        assert cell.get("loco") is True, (
            f"{qid}: cited cell must come from a leave-one-card-out run"
        )
        assert cell.get("masked", True) is True
        assert int(cell.get("n_neg", 0)) >= MIN_NEGATIVES, (
            f"{qid}: cited cell needs >= {MIN_NEGATIVES} negatives (A2)"
        )


def test_profiles() -> None:
    policy = load_policy()
    prod, dev = policy.profile("prod"), policy.profile("dev")
    assert prod.min_level_write == "L1" and dev.min_level_write == "L0"
    assert prod.auto_calibrations == {"anyjev"}
    assert prod.allow_hosted_write is False


def test_defaults_file_ships_proposed_only() -> None:
    assert all(t.auto is None for t in load_policy().questions.values())
    assert Path(DEFAULTS_PATH).name == "policy_defaults.yaml"

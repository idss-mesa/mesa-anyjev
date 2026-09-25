"""The level / calibration contract (DESIGN D3) and the outcome rules (D6, D11, A4, A7)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from mesa_anyjev.policy import Profile, Thresholds, masked, outcome
from mesa_anyjev.providers.base import DecisionRecord, meets_level, rank_probs

STATE = {"card": {"dataset": "x"}, "column": {"name": "c"}}
SHA = "0" * 64


def _rec(**over: object) -> DecisionRecord:
    base: dict[str, object] = {
        "question_id": "term.fits",
        "question_key": "k" * 16,
        "kind": "noul",
        "options": ["Yes", "No"],
        "state": STATE,
        "state_sha256": SHA,
        "provider": "anyjev",
        "method": "anyjev",
        "model": "m",
        "level": "L1",
        "calibration": "anyjev",
        "probs": [0.8, 0.2],
        "answer_index": 0,
        "answer": "Yes",
        "confidence": 0.8,
        "p_true": 0.8,
        "margin": 0.6,
    }
    base.update(over)
    return DecisionRecord.model_validate(base)


PROD = Profile("prod", "L1", frozenset({"anyjev"}), False)
DEV = Profile("dev", "L0", frozenset({"anyjev"}), False)
T_CITED = Thresholds(
    "p_true", 0.7, 0.5, 0.15, "L1", 0.5, "bench/results/2026-01-01/x.json#term_fits.L1"
)
T_NULL = Thresholds("p_true", None, 0.5, 0.15, "L1", 0.5, "bootstrap")


def test_probs_none_iff_calibration_none() -> None:
    _rec(
        level="none",
        calibration="none",
        probs=None,
        confidence=None,
        p_true=None,
        margin=None,
        provider="claude",
        method="claude:structured_output",
    )
    with pytest.raises(ValidationError):
        _rec(level="none", calibration="none")  # probs present
    with pytest.raises(ValidationError):
        _rec(
            level="L0", calibration="none", probs=None, p_true=None
        )  # a level without AnyJev probs
    with pytest.raises(ValidationError):
        _rec(level="L1", calibration="typesafe", provider="motherduck", method="hosted:prompt_jev")


def test_no_one_hot_and_distribution_checks() -> None:
    with pytest.raises(ValidationError):
        _rec(probs=[1.0, 0.2])
    with pytest.raises(ValidationError):
        _rec(probs=[0.8, 0.2, 0.0])


def test_noul_thresholds_read_p_true_not_confidence() -> None:
    confident_no = _rec(
        probs=[0.05, 0.95], answer_index=1, answer="No", confidence=0.95, p_true=0.05, margin=0.9
    )
    assert outcome(confident_no, T_CITED, PROD) == "abstain"
    yes = _rec()
    assert outcome(yes, T_CITED, PROD) == "auto"


def test_none_calibration_never_auto_even_in_dev() -> None:
    claude = _rec(
        level="none",
        calibration="none",
        probs=None,
        confidence=None,
        p_true=None,
        margin=None,
        provider="claude",
        method="claude:structured_output",
    )
    assert outcome(claude, T_CITED, DEV) == "abstain"


def test_hosted_typesafe_can_propose_but_not_auto() -> None:
    hosted = _rec(
        level="none", calibration="typesafe", provider="motherduck", method="hosted:prompt_jev"
    )
    assert outcome(hosted, T_CITED, PROD) == "proposed"
    assert outcome(hosted, T_CITED, DEV) == "proposed"


def test_level_floor_and_null_threshold() -> None:
    l0 = _rec(level="L0")
    assert outcome(l0, T_CITED, PROD) == "proposed"  # below the prod floor
    t_l0 = Thresholds("p_true", 0.7, 0.5, 0.15, "L0", 0.5, "bench/results/2026-01-01/x.json#c")
    assert outcome(l0, t_l0, DEV) == "auto"
    assert outcome(l0, t_l0, PROD) == "proposed"  # the profile floor still applies
    assert outcome(_rec(), T_NULL, PROD) == "proposed"  # auto: null -> proposed-only


def test_missing_label_caps_the_outcome() -> None:
    assert outcome(_rec(), T_CITED, PROD, missing_labels=1) == "abstain"


def test_margin_floor() -> None:
    tight = _rec(probs=[0.55, 0.45], confidence=0.55, p_true=0.55, margin=0.1)
    assert outcome(tight, Thresholds("p_true", 0.5, 0.5, 0.15, "L1", 0.5, "c"), PROD) == "proposed"


def test_masked_renormalises_and_logs_mass() -> None:
    rec = _rec(
        kind="choice",
        options=["a", "b", "c"],
        probs=[0.5, 0.3, 0.2],
        answer_index=0,
        answer="a",
        confidence=0.5,
        p_true=None,
        margin=0.2,
    )
    out = masked(rec, np.array([False, True, True]))
    assert out.answer == "b"
    assert abs(sum(out.probs or []) - 1.0) < 1e-9
    assert out.diagnostics["masked_mass"] == 0.5
    empty = masked(rec, np.array([False, False, False]))
    assert empty.answer_index == -1 and empty.diagnostics["mask_empty"] is True


def test_rank_and_meets_level() -> None:
    idx, conf, margin = rank_probs([0.2, 0.7, 0.1])
    assert (idx, conf) == (1, 0.7) and margin == pytest.approx(0.5)
    assert meets_level(_rec(level="L2"), "L1")
    assert not meets_level(_rec(level="L0"), "L1")

"""Every production Question.key is pinned; wording or option edits must rotate the lock."""

from __future__ import annotations

from anyjev import Question

from mesa_anyjev import questions
from mesa_anyjev.registry import ONTOLOGY_OPTIONS


def test_lock_matches_code() -> None:
    assert questions.lock_drift() == []


def test_lock_sha_is_stable() -> None:
    assert questions.lock_sha() == questions.read_lock()["lock_sha"]


def test_every_question_has_a_spec_and_unique_key() -> None:
    keys = [spec.question.key for spec in questions.QUESTIONS.values()]
    assert len(keys) == len(set(keys))
    for qid, spec in questions.QUESTIONS.items():
        assert spec.question.name == qid


def test_twins_are_nouls_and_choices_over_the_gateway_limit_have_one() -> None:
    for qid, spec in questions.QUESTIONS.items():
        if spec.question.kind == "choice" and spec.question.k > 8:
            assert spec.twin_id, f"{qid}: K={spec.question.k} needs a noul twin (DESIGN D11)"
            assert questions.twin_for(qid) is not None
            assert questions.twin_for(qid).kind == "noul"  # type: ignore[union-attr]


def test_registry_options_are_the_ontology_question_options() -> None:
    assert questions.Q_COLUMN_ONTOLOGY.options == ONTOLOGY_OPTIONS
    assert questions.Q_COLUMN_ONTOLOGY.k == 12


def test_key_excludes_name_but_includes_wording() -> None:
    same = Question.noul(questions.Q_TERM_FITS.text, name="renamed")
    assert same.key == questions.Q_TERM_FITS.key
    reworded = Question.noul(questions.Q_TERM_FITS.text + " Really?", name="term.fits")
    assert reworded.key != questions.Q_TERM_FITS.key

"""M5: the Claude second-opinion provider (stubbed client), the frozen DataCite vocabularies
against mesa-mcp's enums, DataCite classification on the fake backend, and per-key artifact
validity across lock changes (D24)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from anyjev import Question

from mesa_anyjev.artifacts import ArtifactStore
from mesa_anyjev.backends.factory import make_backend
from mesa_anyjev.cards import load_card
from mesa_anyjev.config import load_config
from mesa_anyjev.datacite import card_record, classify
from mesa_anyjev.policy import load_policy
from mesa_anyjev.providers.anyjev_provider import AnyJevProvider
from mesa_anyjev.providers.claude_provider import (
    ClaudeStructuredProvider,
    answer_model,
    render_prompt,
)
from mesa_anyjev.questions import Q_COLUMN_ASPECT, Q_TERM_FITS, QUESTIONS, lock_sha
from mesa_anyjev.registry import DATACITE_VOCABULARIES

ROOT = Path(__file__).resolve().parent
CARD = ROOT / "fixtures" / "cards" / "DP1.10022.001.bet_sorting.md"


class _Messages:
    def __init__(self, answers: list[Any]) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kw: Any) -> Any:
        self.calls.append(kw)
        nxt = self.answers.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if nxt == "refusal":
            return SimpleNamespace(
                stop_reason="refusal", parsed_output=None, usage=None, content=[]
            )
        model = kw["output_format"]
        parsed = (
            model(answer=nxt)
            if nxt in model.model_fields["answer"].annotation.__args__
            else SimpleNamespace(answer=nxt)
        )
        return SimpleNamespace(
            stop_reason="end_turn",
            parsed_output=parsed,
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=[],
        )


def test_claude_provider_records_level_none_and_never_probabilities() -> None:
    cfg = load_config(env={})
    msgs = _Messages(["Yes", "refusal", RuntimeError("boom"), "not an option"])
    prov = ClaudeStructuredProvider(cfg.claude, client=SimpleNamespace(messages=msgs))
    state = {"card": {"dataset": "d"}, "candidate": {"label": "x"}}
    recs = prov.decide_batch([state] * 4, Q_TERM_FITS)
    assert [r.answer_index for r in recs] == [0, -1, -1, -1]
    assert {r.level for r in recs} == {"none"} and {r.calibration for r in recs} == {"none"}
    assert all(
        r.probs is None and r.provider == "claude" and r.method == "claude:structured_output"
        for r in recs
    )
    assert recs[1].diagnostics["refusal"] and "boom" in str(recs[2].diagnostics["error"])
    assert recs[3].diagnostics["unparsed"] == "not an option"
    call = msgs.calls[0]
    assert (
        call["thinking"] == {"type": "adaptive"}
        and "tool_choice" not in call
        and call["model"] == "claude-opus-5"
    )
    assert "State:" in call["messages"][0]["content"] and "A. Yes" in call["messages"][0]["content"]
    assert render_prompt(state, Q_COLUMN_ASPECT).count("\n" + "H. ") == 1
    assert set(answer_model(Q_TERM_FITS).model_fields) == {"answer"}


def test_datacite_vocabularies_match_mesa_mcp() -> None:
    schema = pytest.importorskip("mesa_mcp.datacite.schema")
    for name, members in DATACITE_VOCABULARIES.items():
        assert tuple(m.value for m in getattr(schema, name)) == members, name
    assert len(DATACITE_VOCABULARIES["ResourceTypeGeneral"]) == 28  # > 26: twin only
    assert QUESTIONS["datacite.contributor_type"].twin_id == "datacite.contributor_type_fits"
    assert QUESTIONS["datacite.resource_type_fits"].question.kind == "noul"


def test_datacite_classification_on_the_fake_backend() -> None:
    cfg = load_config(env={"MESA_ANYJEV_POLICY__PROFILE": "dev"})
    provider = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    policy = load_policy()
    profile = policy.profile("dev")
    card = load_card(CARD)
    c = classify(
        provider,
        "ResourceTypeGeneral",
        "A tabular dataset of beetle counts",
        policy=policy,
        profile=profile,
        card=card,
    )
    assert (
        c.question_id == "datacite.resource_type_fits" and len(c.records) == 28 and c.level == "L0"
    )
    assert c.ranked[0][0] in DATACITE_VOCABULARIES["ResourceTypeGeneral"]
    d = classify(
        provider,
        "DescriptionType",
        "We sampled beetles with pitfall traps.",
        policy=policy,
        profile=profile,
    )
    assert (
        d.question_id == "datacite.description_type" and len(d.records) == 1 and len(d.ranked) == 6
    )
    # the fake backend reads every option: 21 contributor types are one choice
    e = classify(
        provider,
        "ContributorType",
        "Jane Doe collected the samples",
        policy=policy,
        profile=profile,
    )
    assert e.question_id == "datacite.contributor_type" and len(e.records) == 1
    # under the gateway's cap (8) the same call asks the yes/no twin per member
    capped = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    capped.capabilities = capped.capabilities.__class__(8, True, 20)
    f = classify(
        capped, "ContributorType", "Jane Doe collected the samples", policy=policy, profile=profile
    )
    assert f.question_id == "datacite.contributor_type_fits" and len(f.records) == 21
    assert f.ranked[0][0] in DATACITE_VOCABULARIES["ContributorType"]
    with pytest.raises(ValueError):
        classify(provider, "Nope", "x", policy=policy, profile=profile)
    rec = card_record(provider, card, policy=policy, profile=profile)
    assert rec["record"]["titles"][0]["title"] == card.product_title
    assert len(rec["classifications"]) == 2


def test_artifacts_stay_valid_across_lock_changes_when_keys_survive(tmp_path: Path) -> None:
    from anyjev import Decider

    cfg = load_config(env={})
    backend = make_backend(cfg.backend)
    dec = Decider(backend, level="L0", adaptive_shifts=False)
    q = Q_TERM_FITS
    states = [{"card": {"dataset": "d"}, "candidate": {"label": f"x{i}"}} for i in range(40)]
    dec.calibrate(q, states, [i % 2 for i in range(40)])
    old = ArtifactStore(tmp_path / "art", "fake", "0" * 64)  # an older lock sha
    old.save(
        dec, {"fitted_on": {"backend_kind": "fake"}, "per_question": {"term.fits": {"key": q.key}}}
    )
    old.promote(1)
    new = ArtifactStore(tmp_path / "art", "fake", lock_sha())
    assert new.dir != old.dir and new.current() is not None  # inherited from the sibling lock
    fresh = Decider(backend, level="L0")
    assert new.load_into(fresh, backend_kind="fake") == 1
    # a bundle whose key no longer exists is refused
    gone = ArtifactStore(tmp_path / "art2", "fake", "1" * 64)
    gone.save(
        dec, {"fitted_on": {"backend_kind": "fake"}, "per_question": {"old.q": {"key": "deadbeef"}}}
    )
    gone.promote(1)
    later = ArtifactStore(tmp_path / "art2", "fake", lock_sha())
    assert later.current() is None
    with pytest.raises(ValueError, match="no longer exist"):
        later.load_into(fresh, version=gone.versions()[0], backend_kind="fake")
    assert Question.noul("x", name="x").key != q.key

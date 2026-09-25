"""M3 on the fake backend: the composite backend, L2 fitting with leave-one-card-out, the
bench's L2 cell, the provider's level resolution with a fitted head, and bundle loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from anyjev import Question
from anyjev.backends.fake import FakeBackend

from mesa_anyjev.artifacts import ArtifactStore
from mesa_anyjev.backends.composite import CompositeBackend
from mesa_anyjev.backends.factory import default_fake_content, make_backend
from mesa_anyjev.bench.run import run_task
from mesa_anyjev.bench.tasks.neon import tasks_from_store
from mesa_anyjev.config import load_config
from mesa_anyjev.learn.fit import fit_question, promote
from mesa_anyjev.learn.labels import TermResolver, ingest_neon_eval
from mesa_anyjev.ols import RecordingOLS
from mesa_anyjev.policy import load_policy
from mesa_anyjev.provenance.store import DuckDBStore
from mesa_anyjev.providers.anyjev_provider import AnyJevProvider
from mesa_anyjev.providers.base import CapabilityError, LevelUnavailable
from mesa_anyjev.questions import Q_COLUMN_ONTOLOGY, Q_TERM_FITS, QUESTIONS, lock_sha

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures" / "ols"
EVAL_ROOT = Path(
    os.environ.get("MESA_ANYJEV_EVAL_ROOT", "/home/tswetnam/github/idss-mesa/neon-avu-eval")
)
HAVE_LABELS = (EVAL_ROOT / "results" / "validated.json").exists() and any(FIXTURES.glob("*.json"))


def test_composite_routes_and_checks_label_parity() -> None:
    a = FakeBackend(content=default_fake_content)
    b = FakeBackend(content=default_fake_content)
    comp = CompositeBackend(a, b, max_choice_k=8)
    assert comp.name == "fake" and comp.n_layers == 4 and comp.capabilities.hidden_states
    ids = [a.tokenizer.encode("Yes")[0], a.tokenizer.encode("No")[0]]
    lp = comp.next_token_logprobs(["State:\nx\n\nQuestion: q\nAnswer Yes or No."], [ids])
    assert a.calls == 1 and b.calls == 0
    feats, _, _ = comp.hidden_states(["State:\nx\n\nQuestion: q\nAnswer Yes or No."], layers=[4])
    assert feats.shape[-1] == 64 and b.hidden_calls == 1
    assert len(lp) == 1

    class Tok:
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1, 2]

    c = FakeBackend(content=default_fake_content)
    c.tokenizer = Tok()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="label tokens"):
        CompositeBackend(a, c)


def test_count_prompts_wraps_every_prefill() -> None:
    from mesa_anyjev.backends.factory import count_prompts

    class Bare:
        def next_token_logprobs(self, prompts: list[str], ids: list[list[int]]) -> list[int]:
            return [0] * len(prompts)

        def hidden_states(self, prompts: list[str], layers: list[int]) -> int:
            return len(prompts)

    b = count_prompts(Bare())
    b.next_token_logprobs(["a", "b"], [[1], [1]])
    b.hidden_states(["c"], [1])
    assert b.prompts_seen == 3
    fake = make_backend(load_config(env={}).backend)
    assert count_prompts(fake) is fake  # FakeBackend already counts


def test_gate_skipped_at_l2_but_l2_needs_a_head() -> None:
    cfg = load_config(env={})
    provider = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    provider.capabilities = provider.capabilities.__class__(8, True, None)
    state = {"card": {"dataset": "d"}, "column": {"name": "c"}}
    with pytest.raises(CapabilityError):
        provider.decide_batch([state], Q_COLUMN_ONTOLOGY, level="L0")
    with pytest.raises(LevelUnavailable):
        provider.decide_batch([state], Q_COLUMN_ONTOLOGY, level="L2")


@pytest.mark.skipif(
    not HAVE_LABELS, reason="needs the neon-avu-eval checkout and recorded OLS fixtures"
)
def test_l2_fit_bench_and_bundle(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "labels.duckdb")
    store.ensure_schema()
    ingest_neon_eval(store, EVAL_ROOT, TermResolver(RecordingOLS(None, FIXTURES, "replay")))
    cfg = load_config(env={})
    provider = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    assert provider.capabilities.hidden_states  # FakeBackend plants hidden states
    artifacts = ArtifactStore(tmp_path / "art", "fake", lock_sha())
    rep = fit_question(
        provider,
        store,
        artifacts,
        "term.fits",
        policy=load_policy(),
        level="L2",
        backend_kind="fake",
    )
    assert rep.status == "fitted", rep.detail
    assert rep.loco["n_folds"] >= 1 and all(f["levels"] == ["L2"] for f in rep.folds.values())
    assert all(f["blocks_executed"] for f in rep.folds.values())
    manifest = artifacts.versions()[-1].manifest
    assert manifest["per_question"]["term.fits"]["method"].startswith("head:")
    assert manifest["per_question"]["term.fits"]["layer_abs"] is not None
    ok, _ = promote(artifacts, rep.version or 0, "term.fits")
    assert ok
    # a fresh provider loads the bundle and now resolves term.fits at L2 (exact key only)
    fresh = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    assert fresh.load_bundle(artifacts, backend_kind="fake") >= 1
    assert fresh.resolve_level(Q_TERM_FITS, "auto") == "L2"
    assert (
        fresh.resolve_level(QUESTIONS["avu.keep"].question, "auto") == "L0"
    )  # no routing between nouls (A1)
    with pytest.raises(ValueError, match="strict"):
        AnyJevProvider(make_backend(cfg.backend), cfg.decider).load_bundle(
            artifacts, backend_kind="gateway"
        )
    # the bench's L2 cell on the same labels
    tasks = tasks_from_store(store)
    res = run_task(provider, tasks["neon_term_fits"], ("L1", "L2"), loco=True, flip_probe=False)
    assert res["cells"]["L2"]["levels"] == ["L2"] and res["cells"]["L2"]["loco"]
    assert res["cells"]["L1"]["levels"] == ["L1"]
    # D18: the fit report and the bench cell are the same numbers on the same labels
    for k in ("acc", "ece", "brier", "cov@5%", "cov@10%", "n"):
        assert abs(float(rep.loco[k]) - float(res["cells"]["L2"][k])) < 1e-9, k
    assert "_probs" not in next(iter(manifest["validation"]["term.fits"]["folds"].values()))
    store.close()


def test_noul_head_serves_only_its_own_key() -> None:
    """A1: a fitted noul head must not be routed to another noul question at level auto."""
    cfg = load_config(env={})
    provider = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    q_other = Question.noul("Something else entirely?", name="other")
    states = [
        {"card": {"dataset": "d"}, "candidate": {"label": f"x{i}"}, "column": {"name": "x0"}}
        for i in range(40)
    ]
    labels = [i % 2 for i in range(40)]
    provider.fit_head(Q_TERM_FITS, states, labels, layers=[4])
    assert provider.resolve_level(Q_TERM_FITS, "auto") == "L2"
    assert provider.resolve_level(q_other, "auto") == "L0"
    recs = provider.decide_batch(states[:3], q_other, level="auto")
    assert {r.level for r in recs} == {"L0"}

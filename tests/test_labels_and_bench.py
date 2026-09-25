"""Label ingestion from neon-avu-eval (recorded get_term fixtures), the task loaders, the
bench runner on the fake backend, and the L1 fitter with its guards and promote gate."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mesa_anyjev.artifacts import ArtifactStore
from mesa_anyjev.backends.factory import make_backend
from mesa_anyjev.bench.run import run_task, suggest_policy, write_results
from mesa_anyjev.bench.tasks.neon import tasks_from_store
from mesa_anyjev.config import load_config
from mesa_anyjev.learn.fit import fit_question, promote
from mesa_anyjev.learn.labels import TermResolver, ingest_neon_eval, labelled_states
from mesa_anyjev.ols import RecordingOLS
from mesa_anyjev.policy import load_policy
from mesa_anyjev.provenance.store import DuckDBStore
from mesa_anyjev.providers.anyjev_provider import AnyJevProvider
from mesa_anyjev.questions import QUESTIONS

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures" / "ols"
EVAL_ROOT = Path(
    os.environ.get("MESA_ANYJEV_EVAL_ROOT", "/home/tswetnam/github/idss-mesa/neon-avu-eval")
)

pytestmark = pytest.mark.skipif(
    not (EVAL_ROOT / "results" / "validated.json").exists() or not any(FIXTURES.glob("*.json")),
    reason="needs the neon-avu-eval checkout and recorded OLS fixtures",
)


@pytest.fixture(scope="module")
def labelled(tmp_path_factory: pytest.TempPathFactory) -> DuckDBStore:
    store = DuckDBStore(tmp_path_factory.mktemp("labels") / "prov.duckdb")
    store.ensure_schema()
    report = ingest_neon_eval(
        store, EVAL_ROOT, TermResolver(RecordingOLS(None, FIXTURES, "replay"))
    )
    assert report.inserted > 0
    return store


def test_ingest_counts_match_research_md(labelled: DuckDBStore) -> None:
    ls = labelled_states(labelled, QUESTIONS["term.fits"].question, min_weight=0.5)
    counts = ls.class_counts()
    assert counts[0] >= 85 and counts[1] >= 190, (
        counts
    )  # 92 consensus positives, 211 single-model negatives (minus unresolved)
    assert len(set(ls.cards)) == 7
    strict = labelled_states(labelled, QUESTIONS["term.fits"].question, min_weight=0.6)
    assert set(strict.labels) == {0}  # positives-only at 0.6: the A2 trap the min_weight avoids
    keep = labelled_states(labelled, QUESTIONS["avu.keep"].question, min_weight=0.5)
    assert len(keep) > 100 and 0 in keep.labels
    vk = labelled_states(labelled, QUESTIONS["avu.value_kind"].question, min_weight=0.5)
    assert len(set(vk.labels)) >= 3


def test_ingest_is_idempotent(labelled: DuckDBStore) -> None:
    report = ingest_neon_eval(
        labelled, EVAL_ROOT, TermResolver(RecordingOLS(None, FIXTURES, "replay"))
    )
    assert report.inserted == 0 and report.skipped_existing > 0


def test_tasks_and_bench_on_fake(labelled: DuckDBStore, tmp_path: Path) -> None:
    tasks = tasks_from_store(labelled)
    assert {"neon_term_fits", "neon_keep_avu", "neon_aspect", "neon_term_choice26"} <= set(tasks)
    assert tasks["neon_term_choice26"].levels_supported == ("raw", "L0")
    cfg = load_config(env={})
    provider = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    results = [
        run_task(provider, tasks["neon_term_fits"], ("raw", "L0", "L1", "L2"), loco=True),
        run_task(
            provider, tasks["neon_term_choice26"], ("raw", "L0", "L1"), loco=True, flip_probe=False
        ),
        run_task(provider, tasks["neon_aspect"], ("raw", "L0"), loco=False),
    ]
    tf = results[0]["cells"]
    assert set(tf["L0"]) >= {
        "acc",
        "ece",
        "brier",
        "cov@5%",
        "cov@10%",
        "n_neg",
        "prompts",
        "ms_per_decision",
    }
    assert tf["L0"]["n_neg"] >= 190
    assert tf["L1"]["loco"] is True and tf["L1"]["n_folds"] >= 1 and tf["L1"]["levels"] == ["L1"]
    assert tf["L2"]["loco"] is True and tf["L2"]["levels"] == ["L2"]  # fake hidden states
    assert results[1]["cells"]["L1"]["not_applicable"]
    assert results[2]["cells"]["L0"]["flip"] is not None  # reversed-options probe for a choice
    path = write_results(results, tmp_path, "fake", "fake", {"model": "fake"}, date="2026-01-01")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tasks"]["neon_term_fits"]["cells"]["L1"]["loco"]
    assert (tmp_path / "2026-01-01" / "fake.fake.md").exists()
    sugg = suggest_policy(results)
    assert "term.fits" not in sugg or "L1" in sugg["term.fits"]


def test_fit_guards_and_promote(labelled: DuckDBStore, tmp_path: Path) -> None:
    cfg = load_config(env={})
    provider = AnyJevProvider(make_backend(cfg.backend), cfg.decider)
    policy = load_policy()
    artifacts = ArtifactStore(tmp_path / "art", "fake", "l" * 64)
    rep = fit_question(
        provider, labelled, artifacts, "term.fits", policy=policy, level="L1", backend_kind="fake"
    )
    assert rep.status == "fitted", rep.detail
    assert rep.loco["n_folds"] >= 1 and rep.loco["n_neg"] >= 5
    assert all(f["prior_frozen"] for f in rep.folds.values())
    assert rep.version == 1
    ok, _msg = promote(artifacts, 1, "term.fits")
    assert ok and artifacts.current().version == 1  # type: ignore[union-attr]
    rep2 = fit_question(
        provider, labelled, artifacts, "term.fits", policy=policy, level="L1", backend_kind="fake"
    )
    assert rep2.version == 2
    ok2, _ = promote(artifacts, 2, "term.fits")
    assert ok2  # same data, same fit: no regression
    # a question with too few labels
    rep3 = fit_question(
        provider,
        labelled,
        artifacts,
        "column.ontology",
        policy=policy,
        level="L1",
        backend_kind="fake",
    )
    assert rep3.status in ("insufficient_labels", "fitted")


def test_claude_planner_stub(card) -> None:  # type: ignore[no-untyped-def]
    from mesa_anyjev.config import PlannerConfig
    from mesa_anyjev.planner.base import Plan
    from mesa_anyjev.planner.claude_planner import ClaudePlanner

    class Resp:
        stop_reason = "end_turn"
        parsed_output = Plan(ontologies=["envo", "uo"], taxon_queries=["Aves"])
        usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()
        content: list[object] = []

    class Client:
        class messages:
            calls: list[dict[str, object]] = []

            @classmethod
            def parse(cls, **kw: object) -> Resp:
                cls.calls.append(kw)
                return Resp()

    planner = ClaudePlanner(PlannerConfig(kind="claude"), client=Client())
    result = planner.plan(card)
    assert (
        result.planner == "claude"
        and not result.fallback
        and result.plan.ontologies == ["envo", "uo"]
    )
    call = Client.messages.calls[-1]
    assert (
        call["output_format"] is Plan
        and call["thinking"] == {"type": "adaptive"}
        and "tool_choice" not in call
    )

    class Refusal(Resp):
        stop_reason = "refusal"

    class RefusingClient:
        class messages:
            @staticmethod
            def parse(**kw: object) -> Refusal:
                return Refusal()

    fb = ClaudePlanner(PlannerConfig(kind="claude"), client=RefusingClient()).plan(card)
    assert fb.fallback and fb.planner == "claude" and len(fb.plan.ontologies) == 12

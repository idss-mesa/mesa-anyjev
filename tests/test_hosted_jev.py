"""Hosted Jev (MotherDuck prompt_jev) without a token: a local runner stands in for the SQL
function so the provider's rendering, parsing, policy gate, sidecar run and results file are
covered in CI. The live path (`hosted` marker) needs MESA_ANYJEV_MOTHERDUCK=1 and a token."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from anyjev import Question

from mesa_anyjev.cli import main
from mesa_anyjev.config import load_config
from mesa_anyjev.learn.hosted import batch_from_labels, batch_from_run, score_batch
from mesa_anyjev.provenance.store import DuckDBStore
from mesa_anyjev.providers.motherduck_provider import (
    RESULT_KEYS,
    HostedJevProvider,
    parse_result,
    sql_for,
)
from mesa_anyjev.questions import (
    LOCK_PATH,
    Q_COLUMN_ASPECT,
    Q_TERM_FITS,
    QUESTIONS,
    lock_payload,
)
from mesa_anyjev.service import HostedDisabled, hosted_allowed, require_hosted

ROOT = Path(__file__).resolve().parent


class FakeRunner:
    """Answers like prompt_jev would: a DOUBLE for noul, a STRUCT for choice, from the text."""

    name = "fake-motherduck"

    def __init__(self, *, null_every: int = 0, bad_shape: bool = False) -> None:
        self.calls = 0
        self.rows = 0
        self.null_every = null_every
        self.bad_shape = bad_shape

    def score(self, texts: list[str], question: Question) -> list[Any]:
        self.calls += 1
        out: list[Any] = []
        for text in texts:
            self.rows += 1
            if self.null_every and self.rows % self.null_every == 0:
                out.append(None)
                continue
            h = sum(text.encode()) % 100 / 100.0
            if question.kind == "noul":
                out.append(h)
            elif self.bad_shape:
                out.append({"choice": question.options[0]})
            else:
                k = len(question.options)
                weights = [(h * 7 + j) % 1.0 + 0.05 for j in range(k)]
                tot = sum(weights)
                probs = [
                    {"value": str(o), "probability": w / tot}
                    for o, w in zip(question.options, weights, strict=True)
                ]
                best = max(probs, key=lambda p: p["probability"])
                out.append(
                    {
                        "choice": best["value"],
                        "probabilities": probs,
                        "confidence": best["probability"],
                    }
                )
        return out


def test_sql_is_pinned_in_the_lock_and_uses_constants() -> None:
    sql = sql_for(Q_TERM_FITS)
    assert sql.startswith("prompt_jev(state_text, '") and sql.endswith(", noul := TRUE)")
    choice_sql = sql_for(Q_COLUMN_ASPECT)
    assert "choice := ['" in choice_sql and all(
        f"'{o}'" in choice_sql for o in Q_COLUMN_ASPECT.options
    )
    assert sql_for(Question.score("How sure?", levels=["low", "high"], name="s")).endswith(
        "score := ['low', 'high'])"
    )
    locked = json.loads(LOCK_PATH.read_text())
    assert locked["hosted_sql"] == {qid: sql_for(spec.question) for qid, spec in QUESTIONS.items()}
    assert lock_payload()["lock_sha"] == locked["lock_sha"]  # the SQL does not enter the sha
    assert "''" in sql_for(Question.noul("Is it 'quoted'?", name="q"))  # escaped


def test_parse_result_pins_the_shape_and_reorders() -> None:
    q = Q_COLUMN_ASPECT
    opts = list(q.options)
    shuffled = [
        {"value": o, "probability": p}
        for o, p in zip(reversed(opts), [0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01], strict=True)
    ]
    probs, diag = parse_result(
        q, {"choice": opts[-1], "probabilities": shuffled, "confidence": 0.5}
    )
    assert (
        probs is not None
        and probs[-1] == 0.5
        and abs(sum(probs) - 1) < 1e-9
        and diag["hosted_confidence"] == 0.5
    )
    assert parse_result(q, None) == (None, {"hosted_null": True, "unavailable": True})
    assert parse_result(q, {"choice": "x"})[0] is None  # missing keys
    assert (
        "hosted_unmatched"
        in parse_result(
            q,
            {
                "choice": "x",
                "probabilities": [{"value": "nope", "probability": 1.0}],
                "confidence": 1,
            },
        )[1]
    )
    assert (
        "hosted_sum"
        in parse_result(
            q,
            {
                "choice": opts[0],
                "probabilities": [{"value": opts[0], "probability": 0.5}],
                "confidence": 0.5,
            },
        )[1]
    )
    assert parse_result(Q_TERM_FITS, 0.8) == (
        [0.8, 0.19999999999999996],
        {"hosted_confidence": 0.8},
    )
    assert parse_result(Q_TERM_FITS, 1.5)[0] is None
    s = Question.score("How sure?", levels=["low", "mid", "high"], name="s")
    probs, diag = parse_result(s, 1.25)
    assert probs == [0.0, 0.75, 0.25] and diag["score_value"] == 1.25
    assert RESULT_KEYS == ("choice", "probabilities", "confidence")


def test_provider_records_are_typesafe_level_none() -> None:
    cfg = load_config(env={})
    runner = FakeRunner(null_every=3)
    cfg.motherduck.batch_size = 2
    prov = HostedJevProvider(cfg.motherduck, runner)
    states = [{"card": {"dataset": "d"}, "candidate": {"label": f"x{i}"}} for i in range(5)]
    recs = prov.decide_batch(states, Q_TERM_FITS)
    assert runner.calls == 3 and prov.rows_sent == 5  # 32-row packing configurable
    assert [r.level for r in recs] == ["none"] * 5 and {r.provider for r in recs} == {"motherduck"}
    live = [r for r in recs if r.probs is not None]
    assert live and all(r.calibration == "typesafe" and r.p_true is not None for r in live)
    nulls = [r for r in recs if r.probs is None]
    assert nulls and all(
        r.calibration == "none" and r.answer_index == -1 and r.diagnostics["hosted_null"]
        for r in nulls
    )
    assert all(r.diagnostics["egress_region"] == "us-east-1" for r in recs)
    bad = HostedJevProvider(cfg.motherduck, FakeRunner(bad_shape=True)).decide_batch(
        states[:1], Q_COLUMN_ASPECT
    )
    assert bad[0].probs is None and "hosted_shape" in bad[0].diagnostics


def test_policy_gate_off_by_default_and_allowlist() -> None:
    cfg = load_config(env={})
    assert hosted_allowed(cfg, local_source=True)[0] is False
    with pytest.raises(HostedDisabled):
        require_hosted(cfg, local_source=True)
    cfg = load_config(env={"MESA_ANYJEV_POLICY__HOSTED_PROVIDERS": "allowlist"})
    assert hosted_allowed(cfg, local_source=True)[0] is True
    assert hosted_allowed(cfg, project_root="/iplant/home/x/proj")[0] is False
    assert hosted_allowed(
        cfg, project_root="/iplant/home/x/proj", project_avus={"mesa.hosted_inference": "allow"}
    )[0]
    cfg.policy.hosted_allow_project_roots = ["/iplant/home/x"]
    assert hosted_allowed(cfg, project_root="/iplant/home/x/proj/file.csv")[0] is True
    cfg.policy.hosted_allow_local_sources = False
    assert hosted_allowed(cfg, local_source=True)[0] is False
    # prod never auto-writes a typesafe record
    from mesa_anyjev.policy import load_policy

    assert "typesafe" not in load_policy().profile("prod").auto_calibrations


@pytest.mark.skipif(
    not any((ROOT / "fixtures" / "ols").glob("*.json")), reason="OLS fixtures not recorded"
)
def test_score_batch_records_a_hosted_run_and_results_file(tmp_path: Path) -> None:
    from mesa_anyjev.learn.labels import TermResolver, ingest_neon_eval
    from mesa_anyjev.ols import RecordingOLS

    eval_root = ROOT / "fixtures" / "neon-avu-eval"
    store = DuckDBStore(tmp_path / "labels.duckdb")
    store.ensure_schema()
    ingest_neon_eval(
        store, eval_root, TermResolver(RecordingOLS(None, ROOT / "fixtures" / "ols", "replay"))
    )
    cfg = load_config(env={"MESA_ANYJEV_POLICY__HOSTED_PROVIDERS": "allowlist"})
    prov = HostedJevProvider(cfg.motherduck, FakeRunner(null_every=7))
    batch = batch_from_labels(store, "term.fits", min_weight=0.5, limit=40)
    dry = score_batch(prov, batch, store=store, cfg=cfg, actor="t", dry_run=True)
    assert (
        dry.dry_run
        and dry.estimate["rows"] == 40
        and dry.estimate["requests"] == 2
        and dry.estimate["input_tokens_approx"] > 0
    )
    rep = score_batch(
        prov, batch, store=store, cfg=cfg, actor="t", out_dir=tmp_path / "res", date="2026-09-25"
    )
    assert rep.run_id is not None and rep.n == 40 and rep.n_null == 5
    run = store.run(rep.run_id)
    assert (
        run
        and run["backend_kind"] == "motherduck"
        and run["data_left_host"] is True
        and run["egress_region"] == "us-east-1"
    )
    rows = store.decisions(rep.run_id)
    assert len(rows) == 40 and {r["level"] for r in rows} == {"none"}
    assert {r["calibration"] for r in rows} == {"typesafe", "none"}
    assert sum(1 for r in rows if r["outcome"] == "decider_unavailable") == 5
    assert rep.cell["n"] == 35 and "ece" in rep.cell and rep.cell["hosted"] is True
    payload = json.loads(Path(rep.results_path).read_text())
    assert payload["backend"] == "motherduck" and "hosted_term_fits" in payload["tasks"]
    assert Path(rep.results_path).name == "motherduck__prompt_jev.motherduck.json"
    # re-score the decisions of that run: same states, same shas
    again = batch_from_run(store, rep.run_id, "term.fits", limit=3)
    assert len(again.states) == 3 and again.labels == [None] * 3
    store.close()


def test_cli_refuses_hosted_when_off(tmp_path: Path) -> None:
    dsn = f"duckdb:///{tmp_path / 'p.duckdb'}"
    assert main(["hosted", "score", "--provenance", dsn, "--dry-run"]) == 2
    os.environ["MESA_ANYJEV_POLICY__HOSTED_PROVIDERS"] = "allowlist"
    try:
        # allowed, but no labels for the question -> a zero-row estimate, nothing sent
        assert main(["hosted", "score", "--provenance", dsn, "--dry-run"]) == 0
    finally:
        del os.environ["MESA_ANYJEV_POLICY__HOSTED_PROVIDERS"]

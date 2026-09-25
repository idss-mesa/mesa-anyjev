"""Engine tier: local weights on this host (MESA_ANYJEV_ENGINE=hf, the `hf` extra, a CUDA
device). Loads Qwen/Qwen3-8B once per module (~2 min from disk) and checks the three things M3
needs: honest raw/L0 readout, hidden states for a head, and a fitted head that serves only its
own key at level auto."""

from __future__ import annotations

import os

import pytest
from anyjev import Question

from mesa_anyjev.backends.factory import make_backend
from mesa_anyjev.cards import load_card
from mesa_anyjev.config import load_config
from mesa_anyjev.providers.anyjev_provider import AnyJevProvider
from mesa_anyjev.questions import Q_COLUMN_ANNOTATE, Q_COLUMN_ONTOLOGY, Q_TERM_FITS
from mesa_anyjev.states import candidate_state, column_state

pytestmark = pytest.mark.engine

if os.environ.get("MESA_ANYJEV_ENGINE") != "hf":
    pytest.skip("MESA_ANYJEV_ENGINE=hf not set", allow_module_level=True)
pytest.importorskip("torch")


@pytest.fixture(scope="module")
def provider():  # type: ignore[no-untyped-def]
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cfg = load_config(env={**os.environ, "MESA_ANYJEV_BACKEND__KIND": "hf"})
    backend = make_backend(cfg.backend)
    assert backend.n_layers == 36 and backend.hidden_size == 4096  # Qwen/Qwen3-8B
    return AnyJevProvider(backend, cfg.decider)


@pytest.fixture(scope="module")
def card():  # type: ignore[no-untyped-def]
    from pathlib import Path

    return load_card(
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "cards"
        / "DP1.10003.001.brd_countdata.md"
    )


def test_raw_and_l0_read_the_full_vocab(provider, card) -> None:  # type: ignore[no-untyped-def]
    states = [column_state(card, c) for c in card.columns]
    recs = provider.decide_batch(states, Q_COLUMN_ANNOTATE, level="L0")
    assert all(r.level == "L0" and r.calibration == "anyjev" for r in recs)
    assert provider.last_missing_labels == 0  # local logits: every label token is read
    assert all(float(r.diagnostics.get("answer_mass", 0.0)) > 0.9 for r in recs)
    # K=12 is fine locally: the full vocabulary is read, no top-20 cap
    assert provider.capabilities.max_choice_k >= 12
    wide = provider.decide_batch(states[:3], Q_COLUMN_ONTOLOGY, level="L0")
    assert all(r.probs is not None and abs(sum(r.probs) - 1.0) < 1e-6 for r in wide)


def test_fitted_head_serves_only_its_key(provider, card) -> None:  # type: ignore[no-untyped-def]
    col = card.column("observerDistance")
    cands = [
        {
            "label": "distance",
            "curie": "PATO:0000040",
            "ontology_id": "pato",
            "description": "spatial extent",
        },
        {
            "label": "colour",
            "curie": "PATO:0000014",
            "ontology_id": "pato",
            "description": "a quality of light",
        },
    ]
    states = [
        candidate_state(card, "column", col, "measurement", c, i + 1)
        for i, c in enumerate(cands * 20)
    ]
    labels = [0, 1] * 20
    provider.fit_head(Q_TERM_FITS, states, labels, layers=None)
    assert provider.resolve_level(Q_TERM_FITS, "auto") == "L2"
    recs = provider.decide_batch(states[:2], Q_TERM_FITS, level="auto")
    assert [r.level for r in recs] == ["L2", "L2"]
    assert (
        int(recs[0].diagnostics.get("blocks_executed", 99)) < provider.backend.n_layers
    )  # early stop
    other = Question.noul("Is this a unit?", name="probe.other")
    assert provider.resolve_level(other, "auto") == "L0"  # A1: no noul routing

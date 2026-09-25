"""ArtifactStore round trip on the AnyJev FakeBackend (no GPU, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from anyjev import Decider, Question
from anyjev.backends.fake import FakeBackend

from mesa_anyjev.artifacts import ArtifactStore, model_slug


def _decider() -> Decider:
    backend = FakeBackend(content=lambda state, option: 1.0 if option in state else -1.0, position_bias=[0.5, 0.0])
    return Decider(backend, level="L0", adaptive_shifts=False)


def test_slug() -> None:
    assert model_slug("Qwen/Qwen3-8B") == "Qwen__Qwen3-8B"


def test_save_promote_load(tmp_path: Path) -> None:
    q = Question.noul("Does the state mention Yes?", name="t")
    dec = _decider()
    states = [f"state {i} Yes" if i % 2 else f"state {i} No" for i in range(40)]
    labels = [0 if i % 2 else 1 for i in range(40)]
    dec.calibrate(q, states, labels)  # an L1 artifact
    store = ArtifactStore(tmp_path, "fake", "lock" * 16)
    assert store.versions() == [] and store.current() is None
    v1 = store.save(dec, {"fitted_on": {"backend_kind": "fake"}, "per_question": {"t": {"level": "L1"}}})
    assert v1.version == 1 and (v1.path / "artifacts.json").exists() and (v1.path / "observations.json").exists()
    assert store.current() is None
    store.promote(1)
    assert store.current() is not None and store.current().version == 1  # type: ignore[union-attr]
    fresh = _decider()
    assert store.load_into(fresh, backend_kind="fake") == 1
    assert fresh.decide("state 3 Yes", [q], level="L1")["t"].level == "L1"
    with pytest.raises(ValueError, match="strict"):
        store.load_into(_decider(), backend_kind="gateway")
    with pytest.raises(FileNotFoundError):
        store.promote(9)
    v2 = store.save(dec, {})
    assert v2.version == 2 and [v.version for v in store.versions()] == [1, 2]
    assert len(v1.sha256) == 64


def test_load_refuses_model_and_lock_mismatch(tmp_path: Path) -> None:
    dec = _decider()
    store = ArtifactStore(tmp_path, "fake", "lock" * 16)
    store.save(dec, {})
    store.promote(1)
    other_lock = ArtifactStore(tmp_path, "fake", "lock" * 16)
    other_lock.lock_sha = "x" * 64
    with pytest.raises(ValueError, match="lock"):
        other_lock.load_into(_decider())

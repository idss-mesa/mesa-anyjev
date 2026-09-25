"""The vendored AnyJev metrics are byte-identical to upstream (THIRD_PARTY.md) and work."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from mesa_anyjev.bench import _metrics

UPSTREAM_SHA256 = "df0af62c248f27cc806c90266ae62c28b47df8dd8aa9645de815dedc50d7bbbe"


def test_sha256_matches_third_party_md() -> None:
    path = Path(_metrics.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == UPSTREAM_SHA256
    third_party = (path.parents[3] / "THIRD_PARTY.md").read_text(encoding="utf-8")
    assert UPSTREAM_SHA256 in third_party


def test_summarize_runs() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]] * 5)
    labels = [0, 1, 0, 1] * 5
    out = _metrics.summarize(probs, labels)
    assert out["acc"] == 1.0
    assert 0.0 <= out["ece"] <= 1.0

"""Versioned AnyJev artifact bundles (DESIGN D5, D15).

Layout: ``<dir>/<model_slug>/<lock_sha8>/v<N>/{artifacts.json, observations.json, manifest.json}``
plus a ``CURRENT`` file naming the promoted version. ``artifacts.json`` is
``Decider.export_artifacts()``; observations are exported separately because
``save_artifacts`` drops them. Versions are immutable; ``promote`` moves ``CURRENT`` only.
Loading refuses a model, lock or ``fitted_on`` mismatch when ``strict``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_VERSION = re.compile(r"^v(\d+)$")


def model_slug(model: str) -> str:
    """AnyJev's ``anyjev-heads`` convention: ``/`` -> ``__``."""
    return model.replace("/", "__")


@dataclass(frozen=True)
class ArtifactVersion:
    version: int
    path: Path

    @property
    def manifest(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256((self.path / "artifacts.json").read_bytes()).hexdigest()


class ArtifactStore:
    def __init__(self, root: str | Path, model: str, lock_sha: str, *, strict: bool = True) -> None:
        self.root = Path(root).expanduser()
        self.model = model
        self.lock_sha = lock_sha
        self.strict = strict
        self.dir = self.root / model_slug(model) / lock_sha[:8]

    # -- listing -----------------------------------------------------------------------------
    def versions(self) -> list[ArtifactVersion]:
        if not self.dir.exists():
            return []
        out = []
        for child in self.dir.iterdir():
            if child.is_dir() and (m := _VERSION.match(child.name)):
                out.append(ArtifactVersion(int(m.group(1)), child))
        return sorted(out, key=lambda v: v.version)

    def current(self) -> ArtifactVersion | None:
        pointer = self.dir / "CURRENT"
        if not pointer.exists():
            return self._inherited()
        name = pointer.read_text(encoding="utf-8").strip()
        for v in self.versions():
            if v.path.name == name:
                return v
        return None

    def _inherited(self) -> ArtifactVersion | None:
        """No bundle under the current lock: the newest promoted bundle of a sibling lock
        directory whose question keys all still exist (D24), else ``None``."""
        parent = self.dir.parent
        if not parent.exists():
            return None
        candidates: list[tuple[float, ArtifactVersion]] = []
        for sibling in parent.iterdir():
            if not sibling.is_dir() or sibling == self.dir:
                continue
            pointer = sibling / "CURRENT"
            if not pointer.exists():
                continue
            path = sibling / pointer.read_text(encoding="utf-8").strip()
            if not (path / "manifest.json").exists():
                continue
            m = _VERSION.match(path.name)
            version = ArtifactVersion(int(m.group(1)) if m else 0, path)
            if not self.stale_keys(version.manifest):
                candidates.append((path.stat().st_mtime, version))
        if not candidates:
            return None
        return max(candidates, key=lambda c: c[0])[1]

    @staticmethod
    def stale_keys(manifest: dict[str, Any]) -> list[str]:
        """Question keys in the manifest that the current questions no longer produce."""
        from mesa_anyjev.questions import QUESTIONS

        live = {spec.question.key for spec in QUESTIONS.values()}
        per_question = manifest.get("per_question") or {}
        return sorted(
            f"{qid}:{str(entry.get('key'))[:12]}"
            for qid, entry in per_question.items()
            if entry.get("key") not in live
        )

    # -- writes ---------------------------------------------------------------------------------
    def save(self, decider: Any, manifest_extra: dict[str, Any]) -> ArtifactVersion:
        """Write a new immutable version from a live ``anyjev.Decider``."""
        n = (self.versions()[-1].version + 1) if self.versions() else 1
        target = self.dir / f"v{n}"
        tmp = self.dir / f".v{n}.tmp"
        tmp.mkdir(parents=True, exist_ok=False)
        full = decider.export_artifacts(include_observations=True)
        observations = full.pop("observations", {})
        _write_json(tmp / "artifacts.json", full)
        _write_json(tmp / "observations.json", observations)
        manifest = {
            "model": self.model,
            "questions_lock_sha": self.lock_sha,
            "version": n,
            **manifest_extra,
        }
        _write_json(tmp / "manifest.json", manifest)
        os.replace(tmp, target)
        return ArtifactVersion(n, target)

    def promote(self, version: int) -> None:
        if not any(v.version == version for v in self.versions()):
            raise FileNotFoundError(f"no version v{version} under {self.dir}")
        tmp = self.dir / ".CURRENT.tmp"
        tmp.write_text(f"v{version}\n", encoding="utf-8")
        os.replace(tmp, self.dir / "CURRENT")

    # -- loading ---------------------------------------------------------------------------------
    def load_into(
        self,
        decider: Any,
        *,
        version: ArtifactVersion | None = None,
        backend_kind: str | None = None,
    ) -> int:
        chosen = version or self.current()
        if chosen is None:
            return 0
        manifest = chosen.manifest
        if manifest.get("model") != self.model:
            raise ValueError(f"artifact model {manifest.get('model')!r} != {self.model!r}")
        if manifest.get("questions_lock_sha") != self.lock_sha:
            # D24: validity is per question key. A bundle fit under an older lock stays usable
            # when every key it carries still exists unchanged; a reworded or removed question
            # (its key gone) refuses.
            missing = self.stale_keys(manifest)
            if missing:
                raise ValueError(
                    "artifact was fit under a different questions.lock.json and these question "
                    f"keys no longer exist: {', '.join(missing)} (DESIGN D7/D24)"
                )
        fitted_on = manifest.get("fitted_on") or {}
        # composite reads its hidden states from the same local weights hf does, so an
        # hf-fitted bundle (heads) is valid there; nothing else crosses kinds (D5)
        compatible = {backend_kind, "hf"} if backend_kind == "composite" else {backend_kind}
        if (
            self.strict
            and backend_kind
            and fitted_on.get("backend_kind") not in (None, *compatible)
        ):
            raise ValueError(
                f"artifact fit on backend {fitted_on.get('backend_kind')!r} refused for {backend_kind!r} "
                "(strict; DESIGN D5)"
            )
        loaded: int = decider.load_artifacts(str(chosen.path / "artifacts.json"))
        return loaded


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")

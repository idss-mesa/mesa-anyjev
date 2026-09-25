"""``ElicitationChooser``: answers mesa-mcp's own ``term_choice`` elicitation (the eight-candidate
picker inside ``mesa_avu_apply_term``) with a calibrated ``term.fits.chooser`` decision, over the
real MCP protocol and with zero change to mesa-mcp (plan amendment B13).

The picker carries only (ontology id, value, candidates), so the chooser asks its own fixed-key
question over that shape; ``term.fits`` artifacts are never reused. It returns the best candidate's
content when p(fits) clears the ``propose`` threshold and ``None`` (decline) otherwise, and it
keeps a log so a harness can verify what was chosen and why."""

from __future__ import annotations

import re
from typing import Any

from mesa_anyjev.policy import Policy
from mesa_anyjev.questions import Q_TERM_FITS_CHOOSER
from mesa_anyjev.states import chooser_state

_MESSAGE_RE = re.compile(r"Which (?P<ont>[A-Za-z0-9_]+) term describes (?P<value>.+)\?\s*$")


def parse_picker(message: str, schema: dict[str, Any]) -> tuple[str, str, list[dict[str, str]]]:
    """(ontology_id, value, candidates[{iri, label, curie}]) from mesa-mcp's picker."""
    m = _MESSAGE_RE.search(message or "")
    ontology_id = m.group("ont").lower() if m else ""
    raw_value = m.group("value") if m else ""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    prop = (schema.get("properties") or {}).get("iri") or {}
    iris = list(prop.get("enum", []))
    names = list(prop.get("enumNames", iris))
    cands = []
    for iri, name in zip(iris, names, strict=False):
        label, curie = str(name), ""
        if name.endswith(")") and " (" in name:
            label, curie = name.rsplit(" (", 1)
            curie = curie[:-1]
        cands.append({"iri": str(iri), "label": label, "curie": curie})
    return ontology_id, value, cands


class ElicitationChooser:
    def __init__(self, provider: Any, policy: Policy, *, question_id: str = "term.fits.chooser"):
        self.provider = provider
        self.thresholds = policy.thresholds(question_id)
        self.log: list[dict[str, Any]] = []

    def choose(self, message: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        ontology_id, value, cands = parse_picker(message, schema)
        if not cands:
            self.log.append({"message": message, "reason": "no candidates", "content": None})
            return None
        states = [chooser_state(ontology_id, value, c, len(cands)) for c in cands]
        recs = self.provider.decide_batch(states, Q_TERM_FITS_CHOOSER)
        ranked = sorted(
            zip(cands, recs, strict=True), key=lambda x: x[1].p_true or 0.0, reverse=True
        )
        best, rec = ranked[0]
        p = rec.p_true or 0.0
        content = {"iri": best["iri"]} if p >= self.thresholds.propose else None
        self.log.append(
            {
                "message": message,
                "ontology_id": ontology_id,
                "value": value,
                "ranked": [
                    {"iri": c["iri"], "curie": c["curie"], "p_true": r.p_true, "level": r.level}
                    for c, r in ranked
                ],
                "level": rec.level,
                "p_true": p,
                "threshold": self.thresholds.propose,
                "content": content,
            }
        )
        return content

    async def __call__(self, message: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        """The ``Chooser`` protocol of mesa-ducklake's llm_e2e harness."""
        return self.choose(message, schema)

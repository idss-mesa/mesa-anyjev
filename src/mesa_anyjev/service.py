"""``DecisionService``: one provider, one planner, one OLS layer, one policy and one store
behind a lock, shared by the CLI and the ``mesa_decide_*`` MCP tools (DESIGN D14).

* The AnyJev Decider is not thread-safe, so every decision goes through ``self.lock``; a
  caller that cannot take it within ``policy.max_wait_s`` gets :class:`DeciderBusy` and the
  tool degrades to outcome ``decider_unavailable`` instead of queueing forever.
* Human picks are authoritative (outcome ``human``): they become ``human_overrides`` rows,
  ``labels`` rows (curator 1.0 for the pick, curator_implicit 0.7 for the other offered
  candidates, curator 1.0 negatives for an explicit "none of these", plan amendment A3) and
  link status changes. Serving never fits (D15): labels wait for ``learn fit``.
* On a resumed elicitation every label, probability and level comes from the sidecar rows
  (``candidates_for_group``), never from the client-controlled request state (D8).
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from mesa_anyjev.apply import ApplyResult, Mode, apply
from mesa_anyjev.avu import build_avu, value_for
from mesa_anyjev.cards import DatasetCard
from mesa_anyjev.config import Config
from mesa_anyjev.ols import Candidate
from mesa_anyjev.pipeline import AnnotationRun, Annotator
from mesa_anyjev.policy import DEFAULTS_PATH
from mesa_anyjev.provenance.models import AvuLinkRow, HumanOverrideRow, LabelRow
from mesa_anyjev.provenance.store import ProvenanceStore, open_store

PICK_WEIGHT = 1.0
IMPLICIT_WEIGHT = 0.7


class DeciderBusy(RuntimeError):
    """The decider lock could not be taken within ``max_wait_s``."""


def build_collaborators(
    cfg: Config,
    *,
    planner_kind: str | None = None,
    provenance_dsn: str | None = None,
    store: ProvenanceStore | None = None,
    log: Any = None,
) -> tuple[Any, Any, Any, Any, ProvenanceStore]:
    """(provider, planner, ols_layer, policy, store) from the configuration."""
    from mesa_anyjev.artifacts import ArtifactStore
    from mesa_anyjev.backends.factory import make_backend
    from mesa_anyjev.ols import OLSLayer, RecordingOLS
    from mesa_anyjev.planner.static_planner import StaticPlanner
    from mesa_anyjev.policy import load_policy
    from mesa_anyjev.providers.anyjev_provider import AnyJevProvider
    from mesa_anyjev.questions import lock_sha

    log = log or (lambda msg: print(msg, file=sys.stderr))
    backend = make_backend(cfg.backend)
    provider = AnyJevProvider(
        backend,
        cfg.decider,
        served_model=cfg.backend.served_model
        if cfg.backend.kind in ("gateway", "composite")
        else None,
    )
    if cfg.backend.kind != "fake":
        artifacts = ArtifactStore(
            cfg.artifacts.dir, model_for(cfg), lock_sha(), strict=cfg.artifacts.strict
        )
        try:
            loaded = provider.load_bundle(artifacts, backend_kind=cfg.backend.kind)
        except ValueError as exc:
            log(f"artifact bundle refused: {exc}")
        else:
            if loaded:
                log(f"loaded {loaded} artifact(s) from the promoted bundle")
    planner: Any
    kind = planner_kind or cfg.planner.kind
    if kind == "gateway":
        from mesa_anyjev.planner.gateway_planner import GatewayPlanner

        planner = GatewayPlanner(
            cfg.backend.gateway_base_url,
            cfg.backend.gateway_api_key,
            cfg.planner.gateway_model,
            cfg.planner.timeout,
        )
    elif kind == "claude":
        from mesa_anyjev.planner.claude_planner import ClaudePlanner

        planner = ClaudePlanner(cfg.planner)
    else:
        planner = StaticPlanner()
    inner: Any = None
    if cfg.ols.fixtures != "replay":  # replay is strictly offline; auto/record/off reach EBI OLS
        from mesa_mcp.ols.client import OLSClient

        inner = OLSClient(cfg.ols.base_url)
    client: Any = (
        RecordingOLS(inner, cfg.ols.fixtures_dir, cfg.ols.fixtures)
        if cfg.ols.fixtures != "off"
        else inner
    )
    ols = OLSLayer(client, max_candidates=cfg.policy.max_candidates)
    policy = load_policy(
        cfg.policy.defaults_file if Path(cfg.policy.defaults_file).is_absolute() else DEFAULTS_PATH
    )
    if store is None:
        store = open_store(provenance_dsn or cfg.provenance.dsn)
    return provider, planner, ols, policy, store


def model_for(cfg: Config) -> str:
    """The artifact/results model name per backend kind (D5): the served repo id on the
    gateway, the local repo id for hf and composite, ``fake`` otherwise."""
    return {
        "gateway": cfg.backend.canonical_model,
        "hf": cfg.backend.hf_model,
        "composite": cfg.backend.hf_model,
    }.get(cfg.backend.kind, "fake")


class DecisionService:
    def __init__(
        self,
        cfg: Config,
        *,
        store: ProvenanceStore | None = None,
        planner_kind: str | None = None,
        provenance_dsn: str | None = None,
        collaborators: tuple[Any, Any, Any, Any, ProvenanceStore] | None = None,
    ) -> None:
        self.cfg = cfg
        self.lock = threading.Lock()
        self.max_wait_s = float(cfg.policy.max_wait_s)
        self.provider, self.planner, self.ols, self.policy, self.store = (
            collaborators
            or build_collaborators(
                cfg, planner_kind=planner_kind, provenance_dsn=provenance_dsn, store=store
            )
        )

    # -- lock ------------------------------------------------------------------------------------
    def _acquire(self) -> None:
        if not self.lock.acquire(timeout=self.max_wait_s):
            raise DeciderBusy(f"decider busy for more than {self.max_wait_s}s")

    # -- decide phase -----------------------------------------------------------------------------
    def annotate(self, card: DatasetCard, *, actor: str) -> AnnotationRun:
        self._acquire()
        try:
            return Annotator(
                provider=self.provider,
                planner=self.planner,
                ols=self.ols,
                policy=self.policy,
                store=self.store,
                cfg=self.cfg,
                actor=actor,
            ).annotate(card)
        finally:
            self.lock.release()

    # -- reads ------------------------------------------------------------------------------------
    def run_summary(self, run_id: UUID) -> dict[str, Any]:
        run = self.store.run(run_id)
        if run is None:
            raise KeyError(f"run {run_id} not found")
        links = self.store.links(run_id)
        groups = self.store.groups(run_id)
        return {
            "run": run,
            "links": links,
            "groups": groups,
            "pending_groups": [g["group_id"] for g in groups if g["outcome"] == "proposed"],
        }

    def candidates_for_group(self, group_id: UUID) -> list[dict[str, Any]]:
        """The offered candidates with their sidecar probabilities and levels, best first."""
        rows = [d for d in self._group_decisions(group_id) if d["question_id"] == "term.fits"]
        links = {
            str(link["decision_id"]): link
            for link in self._links_for_group(group_id)
            if link.get("decision_id")
        }
        out = []
        for d in rows:
            cand = (d.get("state_json") or {}).get("candidate") or {}
            link = links.get(str(d["decision_id"]))
            out.append(
                {
                    "decision_id": str(d["decision_id"]),
                    "link_id": str(link["link_id"]) if link else None,
                    "curie": cand.get("curie"),
                    "label": cand.get("label"),
                    "ontology_id": cand.get("ontology_id"),
                    "iri": (link or {}).get("term_iri"),
                    "p_true": d.get("p_true"),
                    "level": d.get("level"),
                    "calibration": d.get("calibration"),
                    "outcome": d.get("outcome"),
                    "write_status": (link or {}).get("write_status"),
                }
            )
        out.sort(key=lambda c: -float(c["p_true"] or 0.0))
        return out

    def _group_decisions(self, group_id: UUID) -> list[dict[str, Any]]:
        run_id = self._run_of_group(group_id)
        return [d for d in self.store.decisions(run_id) if str(d.get("group_id")) == str(group_id)]

    def _links_for_group(self, group_id: UUID) -> list[dict[str, Any]]:
        run_id = self._run_of_group(group_id)
        return [
            link for link in self.store.links(run_id) if str(link.get("group_id")) == str(group_id)
        ]

    def _run_of_group(self, group_id: UUID) -> UUID:
        group = self.store.group(group_id)
        if group is None:
            raise KeyError(f"group {group_id} not found")
        return UUID(str(group["run_id"]))

    # -- human feedback ------------------------------------------------------------------------
    def record_human_pick(
        self,
        group_id: UUID,
        *,
        actor: str,
        chosen_decision_id: UUID | None,
        action: str = "pick",
        elicitation_key: str | None = None,
    ) -> dict[str, Any]:
        """A pick (``chosen_decision_id`` among the group's candidates), an explicit ``none``
        (``chosen_decision_id=None``) or a ``reject`` of the winner. Returns the override id,
        the labels written and the link that is now accepted (if any)."""
        cands = self.candidates_for_group(group_id)
        if not cands:
            raise KeyError(f"group {group_id} has no candidates")
        by_id = {c["decision_id"]: c for c in cands}
        chosen = str(chosen_decision_id) if chosen_decision_id is not None else None
        if chosen is not None and chosen not in by_id:
            raise ValueError("chosen decision is not among the offered candidates")
        if action not in ("pick", "reject", "decline"):
            raise ValueError(f"unknown feedback action {action!r}")
        decisions = {str(d["decision_id"]): d for d in self._group_decisions(group_id)}
        run_id = self._run_of_group(group_id)
        run = self.store.run(run_id) or {}
        labels: list[LabelRow] = []
        for c in cands:
            d = decisions[c["decision_id"]]
            is_pick = chosen is not None and c["decision_id"] == chosen
            if action == "decline":
                continue  # a decline says nothing about the candidates
            explicit_none = action == "reject" or (action == "pick" and chosen is None)
            labels.append(
                LabelRow(
                    question_id="term.fits",
                    question_key=str(d["question_key"]),
                    state_sha256=str(d["state_sha256"]),
                    state_json=dict(d["state_json"] or {}),
                    label_index=0 if is_pick else 1,
                    label_source="curator" if (is_pick or explicit_none) else "curator_implicit",
                    weight=PICK_WEIGHT if (is_pick or explicit_none) else IMPLICIT_WEIGHT,
                    source_ref=f"override:{group_id}",
                    card=str(run.get("card_name") or "") or None,
                    decision_id=UUID(c["decision_id"]),
                )
            )
        n_labels = self.store.insert_labels(labels)
        override = HumanOverrideRow(
            group_id=group_id,
            decision_id=UUID(chosen) if chosen else None,
            link_id=UUID(by_id[chosen]["link_id"]) if chosen and by_id[chosen]["link_id"] else None,
            actor=actor,
            action="pick"
            if action == "pick" and chosen
            else ("reject" if action != "decline" else "decline"),
            chosen_index=next((i for i, c in enumerate(cands) if c["decision_id"] == chosen), None),
            chosen_curie=by_id[chosen]["curie"] if chosen else None,
            elicitation_key=elicitation_key,
            offered=[
                {"decision_id": c["decision_id"], "curie": c["curie"], "p_true": c["p_true"]}
                for c in cands
            ],
        )
        self.store.insert_override(override)
        accepted_link: str | None = None
        links = self._links_for_group(group_id)
        if action == "decline":
            pass
        elif chosen is None:
            self.store.set_link_status(
                [
                    UUID(str(link["link_id"]))
                    for link in links
                    if link["write_status"] == "proposed"
                ],
                "rejected",
            )
            self.store.update_group(group_id, outcome="rejected")
        else:
            accepted_link = self._accept_candidate(group_id, by_id[chosen], links, actor)
            self.store.update_group(group_id, outcome="human", winner_decision_id=UUID(chosen))
        return {
            "override_id": str(override.override_id),
            "labels_written": n_labels,
            "accepted_link_id": accepted_link,
            "outcome": "human" if chosen else ("rejected" if action != "decline" else "declined"),
        }

    def _accept_candidate(
        self,
        group_id: UUID,
        cand: dict[str, Any],
        links: list[dict[str, Any]],
        actor: str,
    ) -> str:
        """Accept the picked candidate's link, or build one when the pick was not the winner."""
        others = [
            UUID(str(link["link_id"]))
            for link in links
            if link["write_status"] == "proposed" and str(link["link_id"]) != cand["link_id"]
        ]
        if others:
            self.store.set_link_status(others, "rejected")
        if cand["link_id"]:
            self.store.set_link_status([UUID(cand["link_id"])], "accepted")
            return str(cand["link_id"])
        winner = next((link for link in links if link.get("decision_id")), None) or (
            links[0] if links else None
        )
        if winner is None:
            raise KeyError(f"group {group_id} has no link to derive the AVU shape from")
        term = Candidate(
            label=str(cand["label"] or ""),
            curie=str(cand["curie"] or ""),
            iri=str(cand.get("iri") or ""),
            ontology_id=str(cand["ontology_id"] or ""),
        )
        kind = str(winner.get("value_kind") or "the term label")
        value = value_for(
            kind,
            term_label=term.label,
            column_name=winner.get("column_name"),
            site_code=winner.get("site_code"),
            top_value=str(winner["value"]) if kind == "the top profile value" else None,
        )
        avu = build_avu(term, value)
        row = AvuLinkRow(
            run_id=UUID(str(winner["run_id"])),
            group_id=group_id,
            decision_id=UUID(cand["decision_id"]),
            irods_path=winner.get("irods_path"),
            target_type=str(winner.get("target_type") or "data_object"),
            attribute=avu["attribute"],
            value=avu["value"],
            unit=avu["unit"],
            term_curie=term.curie,
            term_iri=term.iri or None,
            term_label=term.label,
            ontology_id=term.ontology_id,
            aspect=winner.get("aspect"),
            column_name=winner.get("column_name"),
            site_code=winner.get("site_code"),
            value_kind=kind,
            source=f"mesa-anyjev:feedback:{actor}",
            write_status="accepted",
        )
        self.store.insert_links([row])
        return str(row.link_id)

    # -- write phase ------------------------------------------------------------------------------
    def apply(
        self,
        run_id: UUID,
        *,
        actor: str,
        irods_path: str,
        accept: Any = "auto",
        mode: Mode = "local",
        ducklake: Any | None = None,
        zone: str = "local",
        target_type: str = "data_object",
        dry_run: bool = False,
        session: Any | None = None,
        auth_value: Any | None = None,
        outcome_tag: str | None = None,
    ) -> ApplyResult:
        return apply(
            run_id,
            store=self.store,
            actor=actor,
            irods_path=irods_path,
            accept=accept,
            mode=mode,
            ducklake=ducklake,
            zone=zone,
            target_type=target_type,
            dry_run=dry_run,
            session=session,
            auth_value=auth_value,
            outcome_tag=outcome_tag,
        )

    def accepted_link_ids(self, run_id: UUID) -> list[str]:
        return [
            str(link["link_id"])
            for link in self.store.links(run_id)
            if link["write_status"] == "accepted"
        ]

    def close(self) -> None:
        self.store.close()


def state_matches(cands: Sequence[dict[str, Any]], decision_id: str) -> bool:
    return any(c["decision_id"] == decision_id for c in cands)

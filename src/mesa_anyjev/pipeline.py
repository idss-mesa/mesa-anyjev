"""The decide phase: ``annotate(card) -> AnnotationRun`` (DESIGN D10; amendments A6, A7).

No writes to iRODS here. The card is *staged*: one ``decide_batch`` per question across all
columns (so the batch prior sees every column at once and the run is a function of the card
and the artifacts), then one batch per candidate group for ``term.fits``, then one batch over
every assembled AVU for ``avu.keep``. Every decision, group and proposal is written to the
provenance sidecar before the function returns.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from anyjev import Question

from mesa_anyjev import __version__
from mesa_anyjev.avu import (
    Avu,
    build_avu,
    pre_rule_value_kind,
    top_profile_value,
    triple,
    value_for,
)
from mesa_anyjev.cards import ColumnInfo, DatasetCard, SiteInfo, is_identifier
from mesa_anyjev.config import Config, config_sha256
from mesa_anyjev.ols import Candidate, OLSLayer
from mesa_anyjev.planner.base import Planner, PlanResult
from mesa_anyjev.policy import Outcome, Policy, Profile, masked, outcome
from mesa_anyjev.provenance.models import (
    AvuLinkRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    RunRow,
)
from mesa_anyjev.provenance.store import ProvenanceStore
from mesa_anyjev.providers.base import CapabilityError, DecisionRecord
from mesa_anyjev.questions import (
    Q_COLUMN_ANNOTATE,
    Q_COLUMN_ASPECT,
    Q_COLUMN_ONTOLOGY,
    Q_COLUMN_ONTOLOGY_FITS,
    Q_KEEP_AVU,
    Q_TERM_FITS,
    Q_VALUE_KIND,
    lock_sha,
)
from mesa_anyjev.registry import (
    ASPECTS,
    ONTOLOGY_REGISTRY,
    allowed_for_aspect,
    entry,
    mask_for_aspect,
)
from mesa_anyjev.states import (
    avu_state,
    candidate_state,
    column_state,
    ontology_state,
    state_sha256,
    value_kind_state,
)

ANYJEV_COMMIT = "795a4970b47218b7c0686cd579d691fc2cf8df2f"
MAX_TAXON_AVUS = 2


@dataclass
class Proposal:
    link_id: UUID
    group_id: UUID | None
    decision_id: UUID | None
    avu: Avu
    candidate: Candidate
    scope: str
    column_name: str | None
    site_code: str | None
    aspect: str
    value_kind: str
    p: float | None
    level: str
    outcome: Outcome
    keep_p: float | None = None
    rationale: str = ""


@dataclass
class AnnotationRun:
    run_id: UUID
    card: DatasetCard
    plan: PlanResult
    proposals: list[Proposal]
    rejected: list[Proposal]
    n_decisions: int
    n_prompts: int
    missing_labels: int
    seconds: float
    outcomes: dict[str, int] = field(default_factory=dict)

    def to_eval_result(self) -> dict[str, Any]:
        """The neon-avu-eval result shape, so its scoring scripts run unchanged."""
        return {
            "model": self.plan.model or self.plan.planner,
            "family": "mesa-anyjev",
            "card": self.card.name,
            "run_id": str(self.run_id),
            "parse_ok": True,
            "avus": [
                {
                    "attribute": p.avu["attribute"],
                    "value": p.avu["value"],
                    "unit": p.avu["unit"],
                    "ontology_id": p.candidate.ontology_id,
                    "curie": p.candidate.curie,
                    "iri": p.candidate.iri,
                    "label": p.candidate.label,
                    "aspect": p.aspect,
                    "column": p.column_name,
                    "rationale": p.rationale,
                    "p": p.p,
                    "level": p.level,
                    "outcome": p.outcome,
                    "link_id": str(p.link_id),
                }
                for p in self.proposals
            ],
            "n_decisions": self.n_decisions,
            "n_prompts": self.n_prompts,
            "missing_labels": self.missing_labels,
            "seconds": self.seconds,
            "outcomes": self.outcomes,
        }


class _Recorder:
    """Writes decisions and groups as they happen; keeps the sequence number."""

    def __init__(
        self, store: ProvenanceStore, run_id: UUID, policy: Policy, profile: Profile
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.policy = policy
        self.profile = profile
        self.seq = 0
        self.outcomes: dict[str, int] = {}
        self.rows: list[DecisionRow] = []

    def record(
        self,
        rec: DecisionRecord,
        *,
        scope: str,
        column_name: str | None = None,
        site_code: str | None = None,
        group_id: UUID | None = None,
        missing_labels: int = 0,
        outcome_override: Outcome | None = None,
        options_meta: Sequence[Candidate] | None = None,
    ) -> tuple[DecisionRow, Outcome]:
        self.seq += 1
        t = (
            self.policy.thresholds(rec.question_id)
            if rec.question_id in self.policy.questions
            else None
        )
        out: Outcome = outcome_override or (
            outcome(rec, t, self.profile, missing_labels=missing_labels) if t else "rule"
        )
        row = DecisionRow(
            run_id=self.run_id,
            seq=self.seq,
            group_id=group_id,
            question_id=rec.question_id,
            question_key=rec.question_key,
            kind=rec.kind,
            k=len(rec.options),
            scope=scope,  # type: ignore[arg-type]
            column_name=column_name,
            site_code=site_code,
            state_sha256=rec.state_sha256,
            state_json=rec.state,
            prompt_sha256=rec.prompt_sha256,
            provider=rec.provider,
            method=rec.method,
            model=rec.model,
            served_model=rec.served_model,
            level=rec.level,
            calibration=rec.calibration,
            probs=rec.probs,
            answer_index=rec.answer_index,
            answer=rec.answer,
            confidence=rec.confidence,
            p_true=rec.p_true,
            margin=rec.margin,
            score_value=rec.score_value,
            answer_mass=_f(rec.diagnostics.get("answer_mass")),
            prior_method=_s(rec.diagnostics.get("prior_method")),
            order_flip_l0=_f(rec.diagnostics.get("order_flip_l0")),
            permutations=_i(rec.diagnostics.get("permutations")),
            temperature=_f(rec.diagnostics.get("temperature")),
            n_calib=_i(rec.diagnostics.get("n_calib")),
            routed_from=_s(rec.diagnostics.get("routed_from")),
            adapted=_b(rec.diagnostics.get("adapted")),
            blocks_executed=_i(rec.diagnostics.get("blocks_executed")),
            masked_mass=_f(rec.diagnostics.get("masked_mass")),
            threshold_auto=t.auto if t else None,
            threshold_propose=t.propose if t else None,
            outcome=out,
        )
        options = []
        if rec.kind == "choice" and rec.probs is not None:
            order = np.argsort(rec.probs)[::-1].tolist()
            for i, text in enumerate(rec.options):
                options.append(
                    DecisionOptionRow(
                        decision_id=row.decision_id,
                        option_index=i,
                        option_text=text,
                        prob=rec.probs[i],
                        rank=order.index(i),
                    )
                )
        elif options_meta:
            c = options_meta[0]
            options.append(
                DecisionOptionRow(
                    decision_id=row.decision_id,
                    option_index=0,
                    option_text=c.label,
                    curie=c.curie,
                    iri=c.iri,
                    ontology_id=c.ontology_id,
                    prob=rec.p_true,
                )
            )
        self.store.insert_decisions([row], options)
        self.rows.append(row)
        self.outcomes[out] = self.outcomes.get(out, 0) + 1
        return row, out

    def rule(
        self,
        question: Question,
        state: dict[str, Any],
        answer_index: int,
        *,
        scope: str,
        column_name: str | None = None,
        site_code: str | None = None,
        model: str = "n/a",
    ) -> DecisionRow:
        rec = DecisionRecord(
            question_id=question.id,
            question_key=question.key,
            kind=question.kind,
            options=list(question.options),
            state=state,
            state_sha256=state_sha256(state),
            provider="rule",
            method="rule",
            model=model,
            level="none",
            calibration="none",
            probs=None,
            answer_index=answer_index,
            answer=question.options[answer_index] if answer_index >= 0 else "",
        )
        row, _ = self.record(
            rec, scope=scope, column_name=column_name, site_code=site_code, outcome_override="rule"
        )
        return row

    def group(self, **kw: Any) -> DecisionGroupRow:
        row = DecisionGroupRow(run_id=self.run_id, **kw)
        self.store.insert_group(row)
        return row


def _f(v: Any) -> float | None:
    return None if v is None else float(v)


def _i(v: Any) -> int | None:
    return None if v is None else int(v)


def _s(v: Any) -> str | None:
    return None if v is None else str(v)


def _b(v: Any) -> bool | None:
    return None if v is None else bool(v)


class Annotator:
    """Holds the collaborators; ``annotate(card)`` runs one card."""

    def __init__(
        self,
        *,
        provider: Any,
        planner: Planner,
        ols: OLSLayer,
        policy: Policy,
        store: ProvenanceStore,
        cfg: Config,
        actor: str,
    ) -> None:
        self.provider = provider
        self.planner = planner
        self.ols = ols
        self.policy = policy
        self.profile = policy.profile(cfg.policy.profile)
        self.store = store
        self.cfg = cfg
        self.actor = actor

    # -- helpers ---------------------------------------------------------------------------------
    def _decide(
        self, states: list[dict[str, Any]], question: Question
    ) -> tuple[list[DecisionRecord], int]:
        if not states:
            return [], 0
        recs = self.provider.decide_batch(states, question)
        return recs, int(getattr(self.provider, "last_missing_labels", 0))

    def _t(self, qid: str) -> Any:
        return self.policy.thresholds(qid)

    # -- the run ---------------------------------------------------------------------------------
    def annotate(self, card: DatasetCard) -> AnnotationRun:
        started = time.monotonic()
        backend = self.provider.backend if hasattr(self.provider, "backend") else None
        prompts_before = _prompts(backend)
        if hasattr(self.provider, "reset_running_prior"):
            self.provider.reset_running_prior()
        plan_result = self.planner.plan(card)
        plan = plan_result.plan
        run = RunRow(
            actor=self.actor,
            card_name=card.name,
            card_sha256=card.sha256,
            planner=plan_result.planner,
            planner_model=plan_result.model,
            planner_prompt_sha256=plan_result.prompt_sha256,
            plan_json=plan.model_dump(),
            planner_fallback=plan_result.fallback,
            backend_kind=self.cfg.backend.kind,
            canonical_model=self.provider.model,
            served_model=getattr(self.provider, "served_model", None),
            served_revision=self.cfg.backend.tokenizer_revision
            if self.cfg.backend.kind == "gateway"
            else None,
            tokenizer=self.cfg.backend.tokenizer if self.cfg.backend.kind == "gateway" else None,
            tokenizer_revision=self.cfg.backend.tokenizer_revision
            if self.cfg.backend.kind == "gateway"
            else None,
            anyjev_commit=ANYJEV_COMMIT,
            mesa_anyjev_version=__version__,
            questions_lock_sha=lock_sha(),
            policy_profile=self.profile.name,
            config_sha256=config_sha256(self.cfg),
        )
        self.store.begin_run(run)
        rec = _Recorder(self.store, run.run_id, self.policy, self.profile)
        missing_total = 0
        in_play = (
            frozenset(plan.ontologies)
            if plan.ontologies
            else frozenset(e.id for e in ONTOLOGY_REGISTRY)
        )

        # Q1 column.annotate, staged over every non-identifier column ------------------------------
        live: list[ColumnInfo] = []
        for col in card.columns:
            hint = plan.columns.get(col.name)
            if is_identifier(col) and not (hint and hint.annotate is True):
                rec.rule(
                    Q_COLUMN_ANNOTATE,
                    column_state(card, col),
                    1,
                    scope="column",
                    column_name=col.name,
                )
                continue
            if hint and hint.annotate is False:
                rec.rule(
                    Q_COLUMN_ANNOTATE,
                    column_state(card, col),
                    1,
                    scope="column",
                    column_name=col.name,
                    model="planner",
                )
                continue
            live.append(col)
        recs, miss = self._decide([column_state(card, c) for c in live], Q_COLUMN_ANNOTATE)
        missing_total += miss
        annotate_cols: list[ColumnInfo] = []
        for col, r in zip(live, recs, strict=True):
            _, out = rec.record(r, scope="column", column_name=col.name, missing_labels=miss)
            if out in ("auto", "proposed"):
                annotate_cols.append(col)

        # Q2 column.aspect ------------------------------------------------------------------------------
        recs, miss = self._decide([column_state(card, c) for c in annotate_cols], Q_COLUMN_ASPECT)
        missing_total += miss
        col_aspects: dict[str, list[str]] = {}
        for col, r in zip(annotate_cols, recs, strict=True):
            _, out = rec.record(r, scope="column", column_name=col.name, missing_labels=miss)
            probs = r.probs or []
            order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            top = [ASPECTS[i] for i in order[:2]]
            chosen = (
                [top[0]] if out in ("auto", "proposed") else top
            )  # fan-out below threshold (rule)
            if (hint := plan.columns.get(col.name)) and hint.aspect and hint.aspect not in chosen:
                chosen.append(hint.aspect)
            col_aspects[col.name] = [a for a in chosen if a != "other"] or []

        # Q3 column.ontology (choice on capable backends, else the noul twin) -----------------------------
        col_ontologies: dict[str, list[str]] = {}
        use_choice = self.provider.capabilities.max_choice_k >= Q_COLUMN_ONTOLOGY.k
        for col in annotate_cols:
            aspects = col_aspects.get(col.name, [])
            if not aspects:
                continue
            chosen_onts: list[str] = []
            for aspect in aspects:
                allowed = allowed_for_aspect(aspect) & in_play
                if aspect == "unit":
                    chosen_onts.append("uo")
                    continue
                if not allowed:
                    continue
                if use_choice:
                    try:
                        (r,), miss = self._decide(
                            [{**column_state(card, col), "aspect": aspect}], Q_COLUMN_ONTOLOGY
                        )
                    except CapabilityError:
                        use_choice = False
                        r = None  # type: ignore[assignment]
                    if r is not None:
                        missing_total += miss
                        r = masked(r, mask_for_aspect(aspect, in_play))
                        _, out = rec.record(
                            r, scope="column", column_name=col.name, missing_labels=miss
                        )
                        probs = r.probs or []
                        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
                        picks = [
                            ONTOLOGY_REGISTRY[i].id
                            for i in order[: (1 if out in ("auto", "proposed") else 2)]
                            if probs[i] > 0
                        ]
                        chosen_onts.extend(o for o in picks if o not in chosen_onts)
                        continue
                entries = [entry(o) for o in sorted(allowed)]
                states = [
                    ontology_state(card, col, aspect, e.id, e.option_text, sorted(e.aspects))
                    for e in entries
                ]
                recs, miss = self._decide(states, Q_COLUMN_ONTOLOGY_FITS)
                missing_total += miss
                scored: list[tuple[float, str]] = []
                for e, r in zip(entries, recs, strict=True):
                    _, out = rec.record(
                        r, scope="column", column_name=col.name, missing_labels=miss
                    )
                    scored.append((r.p_true or 0.0, e.id))
                scored.sort(reverse=True)
                propose = self._t("column.ontology_fits").propose
                picks = [o for p, o in scored if p >= propose][:2] or [o for _, o in scored[:2]]
                chosen_onts.extend(o for o in picks if o not in chosen_onts)
            if (
                (hint := plan.columns.get(col.name))
                and hint.ontology
                and hint.ontology in in_play
                and hint.ontology not in chosen_onts
            ):
                chosen_onts.append(hint.ontology)
            col_ontologies[col.name] = chosen_onts

        # S + Q4: candidates and term.fits per (column, ontology) group ------------------------------------
        proposals: list[Proposal] = []
        rejected: list[Proposal] = []
        for col in annotate_cols:
            aspect = (col_aspects.get(col.name) or ["other"])[0]
            hint = plan.columns.get(col.name)
            queries = list(hint.queries) if hint and hint.queries else []
            from mesa_anyjev.planner.static_planner import queries_for_column

            for q in queries_for_column(col):
                if q not in queries:
                    queries.append(q)
            for ont in col_ontologies.get(col.name, []):
                if ont == "uo":
                    unit = self.ols.unit_candidate(col.unit) if col.unit else None
                    cands, log = (
                        ([unit], {"ontology_id": "uo", "table": True})
                        if unit
                        else self.ols.search_candidates([col.unit or "unit"], "uo")
                    )
                else:
                    cands, log = self.ols.search_candidates(queries[:3], ont)
                p = self._rank_group(
                    rec, card, "column", col, aspect, ont, cands, log, missing_total_ref=None
                )
                missing_total += p[1]
                if p[0] is not None:
                    (proposals if p[0].outcome in ("auto", "proposed") else rejected).append(p[0])

        # Q5 site environment ------------------------------------------------------------------------------
        for site in card.sites:
            hint_s = plan.sites.get(site.code)
            queries = (
                list(hint_s.environment_queries) if hint_s and hint_s.environment_queries else []
            )
            if not queries:
                from mesa_anyjev.planner.static_planner import habitat_queries

                queries = habitat_queries(site)
            cands, log = self.ols.biome_candidates(queries)
            p = self._rank_group(
                rec, card, "site", site, "environment", "envo", cands, log, missing_total_ref=None
            )
            missing_total += p[1]
            if p[0] is not None:
                (proposals if p[0].outcome in ("auto", "proposed") else rejected).append(p[0])

        # Q6 dataset taxon -----------------------------------------------------------------------------------
        taxon_q = list(plan.taxon_queries)
        if not taxon_q:
            from mesa_anyjev.planner.static_planner import taxon_queries

            taxon_q = taxon_queries(card)
        if taxon_q and "ncbitaxon" in in_play:
            cands, log = self.ols.search_candidates(taxon_q[:3], "ncbitaxon")
            p = self._rank_group(
                rec,
                card,
                "dataset",
                None,
                "taxon",
                "ncbitaxon",
                cands,
                log,
                missing_total_ref=None,
                keep_top=MAX_TAXON_AVUS,
            )
            missing_total += p[1]
            for prop in p[2]:
                (proposals if prop.outcome in ("auto", "proposed") else rejected).append(prop)

        # Q7 value kind + AVU assembly ---------------------------------------------------------------------
        for prop in proposals:
            pcol = card.column(prop.column_name) if prop.column_name else None
            pre = pre_rule_value_kind(prop.aspect, prop.scope)
            if pre is None and pcol is not None:
                (r,), miss = self._decide(
                    [value_kind_state(card, pcol, prop.candidate.as_state(), prop.aspect)],
                    Q_VALUE_KIND,
                )
                missing_total += miss
                _, out = rec.record(
                    r,
                    scope="avu",
                    column_name=col.name,
                    group_id=prop.group_id,
                    missing_labels=miss,
                )
                kind = r.answer if out in ("auto", "proposed") else "the term label"
            else:
                kind = pre or "the term label"
            value = value_for(
                kind,
                term_label=prop.candidate.label,
                column_name=prop.column_name,
                site_code=prop.site_code,
                top_value=top_profile_value(pcol.profile) if pcol else None,
            )
            prop.value_kind = kind
            prop.avu = build_avu(prop.candidate, value)

        # dedup on the exact triple (rule), then Q8 avu.keep over every proposal ----------------------------
        seen: dict[tuple[str, str, str], Proposal] = {}
        unique: list[Proposal] = []
        for prop in proposals:
            key = triple(prop.avu)
            if key in seen:
                prop.outcome = "rejected"
                prop.rationale = f"duplicate of {seen[key].link_id}"
                rejected.append(prop)
                continue
            seen[key] = prop
            unique.append(prop)
        siblings = [p.avu for p in unique]
        states = [
            avu_state(
                card,
                p.avu,
                p.candidate.as_state(),
                p.scope,
                p.column_name or p.site_code,
                [s for s in siblings if s is not p.avu],
            )
            for p in unique
        ]
        recs, miss = self._decide(states, Q_KEEP_AVU)
        missing_total += miss
        kept: list[Proposal] = []
        for prop, r in zip(unique, recs, strict=True):
            row, out = rec.record(
                r,
                scope="avu",
                column_name=prop.column_name,
                site_code=prop.site_code,
                group_id=prop.group_id,
                missing_labels=miss,
            )
            prop.keep_p = r.p_true
            prop.decision_id = row.decision_id
            if out in ("auto", "proposed"):
                kept.append(prop)
            else:
                prop.outcome = "rejected"
                rejected.append(prop)
        kept.sort(key=lambda p: p.keep_p or 0.0, reverse=True)
        for prop in kept[self.cfg.policy.max_avus :]:
            prop.outcome = "rejected"
            prop.rationale = "over max_avus"
            rejected.append(prop)
        kept = kept[: self.cfg.policy.max_avus]

        # links ------------------------------------------------------------------------------------------
        links = []
        for prop in kept + rejected:
            if not prop.avu:
                continue
            links.append(
                AvuLinkRow(
                    link_id=prop.link_id,
                    run_id=run.run_id,
                    group_id=prop.group_id,
                    decision_id=prop.decision_id,
                    attribute=prop.avu["attribute"],
                    value=prop.avu["value"],
                    unit=prop.avu["unit"],
                    term_curie=prop.candidate.curie,
                    term_iri=prop.candidate.iri,
                    term_label=prop.candidate.label,
                    ontology_id=prop.candidate.ontology_id,
                    aspect=prop.aspect,
                    column_name=prop.column_name,
                    site_code=prop.site_code,
                    value_kind=prop.value_kind,
                    write_status="proposed" if prop in kept else "rejected",
                )
            )
        self.store.insert_links(links)
        seconds = time.monotonic() - started
        n_prompts = _prompts(backend) - prompts_before
        self.store.finish_run(
            run.run_id,
            "decided",
            n_prompts=n_prompts,
            seconds=seconds,
            missing_labels=missing_total,
        )
        return AnnotationRun(
            run.run_id,
            card,
            plan_result,
            kept,
            rejected,
            rec.seq,
            n_prompts,
            missing_total,
            seconds,
            dict(rec.outcomes),
        )

    def _rank_group(
        self,
        rec: _Recorder,
        card: DatasetCard,
        scope: str,
        target: ColumnInfo | SiteInfo | None,
        aspect: str,
        ont: str,
        cands: list[Candidate],
        log: dict[str, Any],
        *,
        missing_total_ref: Any,
        keep_top: int = 1,
    ) -> tuple[Proposal | None, int, list[Proposal]]:
        column_name = target.name if isinstance(target, ColumnInfo) else None
        site_code = target.code if isinstance(target, SiteInfo) else None
        group = rec.group(
            question_id="term.fits",
            scope=scope,
            column_name=column_name,
            site_code=site_code,
            aspect=aspect,
            ontology_id=ont,
            search_json=log,
            n_candidates=len(cands),
            outcome="abstain",
        )
        if not cands:
            return None, 0, []
        states = [
            candidate_state(card, scope, target, aspect, c.as_state(), len(cands)) for c in cands
        ]
        recs, miss = self._decide(states, Q_TERM_FITS)
        rows = [
            rec.record(
                r,
                scope=scope,
                column_name=column_name,
                site_code=site_code,
                group_id=group.group_id,
                missing_labels=miss,
                options_meta=[c],
            )
            for r, c in zip(recs, cands, strict=True)
        ]
        ranked = sorted(
            zip(cands, recs, rows, strict=True), key=lambda x: x[1].p_true or 0.0, reverse=True
        )
        t = self._t("term.fits")
        out_props: list[Proposal] = []
        for i, (cand, r, (row, out)) in enumerate(ranked[:keep_top]):
            runner_up = ranked[i + 1][1].p_true if i + 1 < len(ranked) else 0.0
            gmargin = (r.p_true or 0.0) - (runner_up or 0.0)
            if out in ("auto", "proposed") and gmargin < t.margin and out == "auto":
                out = "proposed"
            prop = Proposal(
                link_id=uuid4(),
                group_id=group.group_id,
                decision_id=row.decision_id,
                avu={},
                candidate=cand,
                scope=scope,
                column_name=column_name,
                site_code=site_code,
                aspect=aspect,
                value_kind="the term label",
                p=r.p_true,
                level=r.level,
                outcome=out,
                rationale=f"p(fits)={(r.p_true or 0.0):.2f} {r.level}",
            )
            out_props.append(prop)
        winner = out_props[0]
        top_p = winner.p or 0.0
        second = ranked[1][1].p_true if len(ranked) > 1 else 0.0
        self.store.update_group(
            group.group_id,
            winner_decision_id=winner.decision_id,
            top_p=top_p,
            group_margin=top_p - (second or 0.0),
            level=winner.level,
            outcome=winner.outcome,
            missing_labels=miss,
        )
        return winner, miss, out_props


def _prompts(backend: Any) -> int:
    if backend is None:
        return 0
    for attr in ("prompts_seen", "requests"):
        if hasattr(backend, attr):
            return int(getattr(backend, attr))
    return 0

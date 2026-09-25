"""The reasoning model on the CARC gateway (carc-tools) as the default planner (DESIGN D2).

A port of neon-avu-eval's streamed chat loop: streaming keeps bytes flowing so the gateway's
read timeout fires only on real stalls; ``chat_template_kwargs.enable_thinking=false`` where
the model honours it; JSON-only instruction, ``parse_json`` strips ``<think>`` blocks and
fences; up to two nudges; any failure falls back to :class:`StaticPlanner` and marks the run
``planner_fallback=true``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

import httpx
from pydantic import ValidationError

from mesa_anyjev.backends.gateway import assert_loopback
from mesa_anyjev.cards import DatasetCard
from mesa_anyjev.planner.base import Plan, PlanResult
from mesa_anyjev.planner.static_planner import StaticPlanner
from mesa_anyjev.registry import ASPECT_OPTIONS, ONTOLOGY_REGISTRY

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a careful scientific metadata curator planning how to annotate a dataset with OBO "
    "Foundry ontology terms. You only PLAN: which ontologies apply, which columns deserve an "
    "annotation and with which aspect and ontology, and short search queries. You never invent "
    "identifiers. Reply with ONLY a JSON object, no prose, no code fence."
)


def parse_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def plan_prompt(card: DatasetCard) -> str:
    registry = "\n".join(f"- {e.id}: {e.option_text}" for e in ONTOLOGY_REGISTRY)
    aspects = "\n".join(f"- {a}" for a in ASPECT_OPTIONS)
    columns = "\n".join(
        f"- {c.name} | {c.description} | {c.dtype} | {c.unit or '-'} | {c.profile[:160]}"
        for c in card.columns
    )
    sites = "; ".join(f"{s.code} ({s.name}, {s.domain}; {s.habitat})" for s in card.sites)
    return (
        f"Dataset: {card.name}\nProduct: {card.product_title}. {card.product_description}\n"
        f"Sites: {sites}\nRows: {card.rows}; months {card.months_from} to {card.months_to}\n\n"
        f"Columns (name | description | type | unit | profile):\n{columns}\n\n"
        f"Ontology registry (use only these ids):\n{registry}\n\nAspects:\n{aspects}\n\n"
        "Return JSON of this exact shape:\n"
        '{"ontologies": ["envo", ...], "columns": {"<column name>": {"annotate": true|false|null, '
        '"aspect": "<aspect id or null>", "ontology": "<registry id or null>", "queries": ["<=3 short OLS queries"]}}, '
        '"sites": {"<SITE>": {"environment_queries": ["biome query", ...]}}, "taxon_queries": ["taxon name", ...], '
        '"notes": "one sentence"}\n'
        "Mark record identifiers, internal codes and timestamps annotate=false. Keep queries short (1-4 words)."
    )


class GatewayPlanner:
    name = "gateway"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str = "carc-tools",
        timeout: float = 600.0,
        *,
        client: httpx.Client | None = None,
        think: bool = False,
    ) -> None:
        self.base_url = assert_loopback(base_url)
        self.model = model
        self.timeout = timeout
        self.think = think
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=5.0),
            trust_env=False,
        )
        self._static = StaticPlanner()

    def _chat(self, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, dict[str, Any]]:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": self.think},
        }
        content: list[str] = []
        usage: dict[str, Any] = {}
        with self._client.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.read()[:300]!r}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        content.append(delta)
        return "".join(content), usage

    def plan(self, card: DatasetCard) -> PlanResult:
        prompt = plan_prompt(card)
        sha = hashlib.sha256((SYSTEM + prompt).encode("utf-8")).hexdigest()
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        usage_total: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
        raw = ""
        try:
            for nudge in range(3):
                raw, usage = self._chat(messages, max_tokens=4096)
                for k in ("prompt_tokens", "completion_tokens"):
                    usage_total[k] += int(usage.get(k, 0))
                usage_total["llm_calls"] += 1
                parsed = parse_json(raw)
                if parsed is not None:
                    try:
                        plan = Plan.model_validate(_coerce(parsed))
                    except ValidationError as exc:
                        logger.warning(
                            "gateway planner: plan rejected (%s); nudging", exc.error_count()
                        )
                        parsed = None
                    else:
                        return PlanResult(
                            plan=plan,
                            planner=self.name,
                            model=self.model,
                            prompt_sha256=sha,
                            raw_text=raw,
                            usage=usage_total,
                        )
                if nudge < 2:
                    messages += [
                        {"role": "assistant", "content": raw or "(no answer)"},
                        {
                            "role": "user",
                            "content": "Reply now with ONLY the JSON object described, nothing else.",
                        },
                    ]
        except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as exc:
            logger.warning("gateway planner failed (%s); falling back to static rules", exc)
        fallback = self._static.plan(card)
        return fallback.model_copy(
            update={
                "planner": self.name,
                "model": self.model,
                "prompt_sha256": sha,
                "raw_text": raw or None,
                "usage": usage_total,
                "fallback": True,
            }
        )


def _coerce(parsed: dict[str, Any]) -> dict[str, Any]:
    """Tolerate the small shape deviations local models make (a list of column objects, etc.)."""
    cols = parsed.get("columns")
    if isinstance(cols, list):
        parsed["columns"] = {
            str(c.get("name", i)): {k: v for k, v in c.items() if k != "name"}
            for i, c in enumerate(cols)
            if isinstance(c, dict)
        }
    for key in ("columns", "sites"):
        if not isinstance(parsed.get(key), dict):
            parsed[key] = {}
    for key in ("ontologies", "taxon_queries"):
        if not isinstance(parsed.get(key), list):
            parsed[key] = []
    if not isinstance(parsed.get("notes"), str):
        parsed["notes"] = ""
    for name, hint in list(parsed["columns"].items()):
        if isinstance(hint, dict):
            hint.setdefault("queries", [])
            if not isinstance(hint["queries"], list):
                hint["queries"] = [str(hint["queries"])]
            parsed["columns"][name] = {
                k: hint.get(k) for k in ("annotate", "aspect", "ontology", "queries")
            }
        else:
            parsed["columns"].pop(name)
    for code, hint in list(parsed["sites"].items()):
        if isinstance(hint, dict):
            eq = hint.get("environment_queries", [])
            parsed["sites"][code] = {
                "environment_queries": eq if isinstance(eq, list) else [str(eq)]
            }
        else:
            parsed["sites"].pop(code)
    return parsed

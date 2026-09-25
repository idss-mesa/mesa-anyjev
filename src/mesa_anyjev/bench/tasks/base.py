"""The AnyJev bench ``Task`` pattern (bench/tasks/base.py at 795a497, Apache-2.0), with two
additions: ``levels_supported`` (a control task whose key changes per item can only run raw
and L0; the runner writes an explicit "not applicable" cell) and leave-one-card-out folds.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from anyjev import Question

TASKS: dict[str, Callable[[], Task]] = {}

Item = tuple[Any, int]


@dataclass
class Task:
    name: str
    question: Question
    items: list[Item]  # (state, label index into question.options)
    license: str
    source: str
    notes: str = ""
    levels_supported: tuple[str, ...] = ("raw", "L0", "L1", "L2")
    cards: list[str] = field(default_factory=list)  # meta.card per item, for LOCO
    weights: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def split(self, n_test: int, n_calib: int, seed: int = 0) -> tuple[list[Item], list[Item]]:
        rng = random.Random(seed)  # noqa: S311 - reproducible exploratory split, not security
        idx = list(range(len(self.items)))
        rng.shuffle(idx)
        test = [self.items[i] for i in idx[:n_test]]
        calib = [self.items[i] for i in idx[n_test : n_test + n_calib]]
        return test, calib

    def leave_one_card_out(self) -> Iterator[tuple[str, list[Item], list[Item]]]:
        """``(held_out_card, test_items, train_items)`` per distinct card. The only split that
        may set a threshold or promote an artifact (DESIGN D6)."""
        if len(self.cards) != len(self.items):
            raise ValueError("leave_one_card_out needs one card per item")
        for card in sorted(set(self.cards)):
            test = [it for it, c in zip(self.items, self.cards, strict=True) if c == card]
            train = [it for it, c in zip(self.items, self.cards, strict=True) if c != card]
            yield card, test, train

    def class_counts(self, items: Sequence[Item] | None = None) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, label in items if items is not None else self.items:
            counts[label] = counts.get(label, 0) + 1
        return counts


def register(name: str) -> Callable[[Callable[[], Task]], Callable[[], Task]]:
    def deco(fn: Callable[[], Task]) -> Callable[[], Task]:
        TASKS[name] = fn
        return fn

    return deco


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]()

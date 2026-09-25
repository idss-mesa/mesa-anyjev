from __future__ import annotations

from anyjev import Question

from mesa_anyjev.bench.tasks.base import TASKS, Task, get_task, register


def test_loco_and_class_counts() -> None:
    q = Question.noul("x?")
    task = Task(
        "t",
        q,
        [({"i": i}, i % 2) for i in range(6)],
        "CC0",
        "test",
        cards=["A", "A", "B", "B", "C", "C"],
    )
    folds = list(task.leave_one_card_out())
    assert [c for c, _, _ in folds] == ["A", "B", "C"]
    assert all(len(test) == 2 and len(train) == 4 for _, test, train in folds)
    assert task.class_counts() == {0: 3, 1: 3}
    assert task.levels_supported == ("raw", "L0", "L1", "L2")


def test_register_and_get() -> None:
    @register("unit_test_task")
    def load() -> Task:
        return Task(
            "unit_test_task", Question.noul("y?"), [], "CC0", "test", levels_supported=("raw", "L0")
        )

    assert get_task("unit_test_task").levels_supported == ("raw", "L0")
    TASKS.pop("unit_test_task")

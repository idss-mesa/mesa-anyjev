"""Bench tasks. Importing a task module registers it (M2 adds the neon loaders)."""

from mesa_anyjev.bench.tasks.base import TASKS, Task, get_task, register

__all__ = ["TASKS", "Task", "get_task", "register"]

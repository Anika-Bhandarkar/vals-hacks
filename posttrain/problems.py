"""Training problem pool: Codeforces `verifiable` train split (prepared on the Modal volume).

    uv run python scripts/prepare_codeforces.py --split train --min-rating 1200 --max-rating 2600 \
        --no-require-editorial --max-problems 2000

Every problem has hidden tests on the volume, so any SFT/RL source can be filtered or rewarded by
the same grader. Eval problems (LiveCodeBench-CPP, Codeforces test split) are removed by
`posttrain.decontam`.
"""

from __future__ import annotations

import random

from cppgrade.agent_tasks import AgentTask, task_from_codeforces
from cppgrade.codeforces import load_index, statement
from posttrain.decontam import is_contaminated

SINGLE_PROMPT = """Solve the following competitive programming problem.

{statement}

Write a complete C++20 program that reads the input from standard input and writes the answer to standard output. Do not print anything else. Put your final solution in a single ```cpp code block."""


def load_pool(
    split: str = "train",
    min_rating: int = 0,
    max_rating: int = 10_000,
    limit: int | None = None,
    seed: int = 0,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Codeforces index rows usable for training, decontaminated, in a seeded nested order
    (growing `limit` only adds problems)."""
    if split == "test":
        raise ValueError("The Codeforces test split is Test A's eval set; don't train on it.")
    rows = [
        p for p in load_index(split)
        if min_rating <= (p.get("rating") or 0) <= max_rating
        and p["id"] not in (exclude_ids or set())
        and not is_contaminated(p["id"], f"{p['description']} {p['input_format']} {p['output_format']}")
    ]
    rows.sort(key=lambda p: p["id"])
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def single_prompt(p: dict) -> str:
    return SINGLE_PROMPT.format(statement=statement(p))


def agent_task(p: dict) -> AgentTask:
    return task_from_codeforces(p)


def limits(p: dict) -> tuple[float, int]:
    """(time limit seconds, memory MB) for grading."""
    return float(p["time_limit"]), max(256, int(p["memory_limit"]))

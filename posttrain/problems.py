"""Training problem pool: Codeforces `verifiable` train split (prepared on the Modal volume).

    uv run python scripts/prepare_codeforces.py --split train --min-rating 1200 --max-rating 2600 \
        --no-require-editorial --max-problems 2000

Every problem has hidden tests on the volume, so any SFT/RL source can be filtered or rewarded by
the same grader. Eval problems (LiveCodeBench-CPP, Codeforces test split) are removed by
`posttrain.decontam`.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict

from cppgrade.agent_tasks import AgentTask, task_from_codeforces
from cppgrade.codeforces import load_index, statement
from posttrain.decontam import is_contaminated

SINGLE_PROMPT = """Solve the following competitive programming problem.

{statement}

Write a complete C++20 program that reads the input from standard input and writes the answer to standard output. Do not print anything else. Put your final solution in a single ```cpp code block."""


# Disjoint parts of the training pool, so RL never trains on problems distillation already pushed
# toward the teacher's answer, and a holdout neither touches (memorization check). Assigned per
# GROUP (a contest round plus any problem re-posted in another round, e.g. Div. 1 / Div. 2), so one
# problem can't sit on both sides under two IDs.
PARTS = (("distill", 0.40), ("rl", 0.50), ("holdout", 0.10))
_DUP_N, _DUP_OVERLAP = 13, 0.5  # word 13-grams; related problems if half its 13-grams match -> same group
_REPOST_OVERLAP = 0.9           # and a re-post (dropped) if 90% match in a different contest


def _text(p: dict) -> str:
    return f"{p['description']} {p['input_format']} {p['output_format']}"


def _grams(text: str) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i : i + _DUP_N]) for i in range(len(w) - _DUP_N + 1)}


def group_and_dedupe(rows: list[dict]) -> tuple[list[dict], dict[str, str], int]:
    """Drop duplicate statements (keep one copy) and group problems by round + shared statements.
    Returns (kept rows, problem id -> group key, number of duplicates dropped)."""
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    grams = {p["id"]: _grams(_text(p)) for p in rows}
    owners: dict[tuple, list[str]] = defaultdict(list)
    for pid, gs in grams.items():
        for g in gs:
            owners[g].append(pid)
    for p in rows:  # same round -> same group
        union(p["id"], f"contest:{p['contest_id']}")
    by_id = {p["id"]: p for p in rows}
    dropped: set[str] = set()
    for pid in sorted(grams):
        hits = Counter(o for g in grams[pid] for o in owners[g] if o != pid and len(owners[g]) <= 3)
        for other, k in hits.items():
            if k < _DUP_OVERLAP * max(1, len(grams[pid])):
                continue
            # Similar statements always share a group (so they land in the same part). Within one contest
            # that's usually an easy/hard version pair: different problems, keep both. Across contests a
            # near-identical statement is a re-post (Div. 1 / Div. 2): keep one copy.
            union(pid, other)
            if (by_id[pid]["contest_id"] != by_id[other]["contest_id"] and k >= _REPOST_OVERLAP * len(grams[pid])
                    and pid not in dropped and other not in dropped):
                dropped.add(max(pid, other))
    kept = [p for p in rows if p["id"] not in dropped]
    return kept, {p["id"]: find(p["id"]) for p in rows}, len(dropped)


def part_of(group_key: str) -> str:
    """Deterministic: the same group always lands in the same part, whatever else is in the pool."""
    u = int(hashlib.sha256(group_key.encode()).hexdigest()[:12], 16) / 16**12
    acc = 0.0
    for name, frac in PARTS:
        acc += frac
        if u < acc:
            return name
    return PARTS[-1][0]


def load_pool(
    split: str = "train",
    min_rating: int = 0,
    max_rating: int = 10_000,
    limit: int | None = None,
    seed: int = 0,
    exclude_ids: set[str] | None = None,
    part: str | None = None,
) -> list[dict]:
    """Codeforces index rows usable for training, decontaminated and deduplicated, in a seeded nested
    order (growing `limit` only adds problems). `part`: "distill" | "rl" | "holdout" | None (all)."""
    if split == "test":
        raise ValueError("The Codeforces test split is Test A's eval set; don't train on it.")
    if part not in (None, *(n for n, _ in PARTS)):
        raise ValueError(f"part must be one of {[n for n, _ in PARTS]} or None")
    clean = [p for p in load_index(split)
             if p["id"] not in (exclude_ids or set()) and not is_contaminated(p["id"], _text(p))]
    kept, group, _ = group_and_dedupe(clean)  # grouping uses the whole pool, before rating filters
    rows = [p for p in kept
            if min_rating <= (p.get("rating") or 0) <= max_rating
            and (part is None or part_of(group[p["id"]]) == part)]
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

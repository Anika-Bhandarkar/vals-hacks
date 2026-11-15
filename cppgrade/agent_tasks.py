"""Competition-format tasks built from LiveCodeBench-CPP problems on the Modal Volume.

Mirrors the hackathon setup as described by the organizers:
  - one hard problem per task, the statement as a text file plus sample tests as files
  - limits: 2 s and 1 GiB per test; the model writes C++20
  - tools: compile+run on any input, and submit (reports sample-test results); up to 50 submits
  - the LAST submission that compiles is graded on hidden tests: small tests 40 pts, large 60
  - the task ends on EXIT, on running out of turns, or on repeated malformed tool calls
  - ties broken by fewest overall turns

Things we don't know yet (the organizers' system prompt, tool names/schemas, turn and context
limits) are guesses here: they're all in this file so they're easy to change once we know.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from cppgrade.lcb import load_index

TIME_LIMIT_S = 2.0
MEMORY_MB = 1024
MAX_SUBMISSIONS = 50


@dataclass(frozen=True)
class AgentTask:
    question_id: str
    title: str
    statement: str
    difficulty: str
    contest_date: str
    samples: tuple[tuple[str, str], ...]  # (input, output) pairs
    tests_file: str  # path on the tests volume (hidden tests = public + private)
    n_tests: int
    time_limit_s: float = TIME_LIMIT_S
    memory_mb: int = MEMORY_MB

    @property
    def sample_tests(self) -> list[dict[str, str]]:
        return [{"input": i, "output": o} for i, o in self.samples]


def load_agent_tasks(
    split: str = "v6_2408_2505",
    difficulties: tuple[str, ...] = ("medium", "hard"),
    limit: int | None = None,
    seed: int = 0,
    after: str | None = None,
) -> list[AgentTask]:
    """AtCoder (stdin/stdout) problems with sample tests. `limit` takes a seeded random subset that
    is stable and nested (the first N of one fixed shuffle), so growing --limit only adds tasks."""
    tasks = []
    for p in load_index(split):
        if p["platform"] != "atcoder" or not p["n_tests"] or p["difficulty"] not in difficulties:
            continue
        if after and p["contest_date"] < after:
            continue
        samples = tuple((t["input"], t["output"]) for t in p["public_tests"] if t.get("testtype") == "stdin")
        if not samples:
            continue
        tasks.append(AgentTask(
            question_id=p["question_id"], title=p["question_title"], statement=p["question_content"].strip(),
            difficulty=p["difficulty"], contest_date=p["contest_date"], samples=samples,
            tests_file=p["tests_file"], n_tests=p["n_tests"],
        ))
    tasks.sort(key=lambda t: t.question_id)
    random.Random(seed).shuffle(tasks)
    return tasks[:limit] if limit else tasks


# ---------------------------------------------------------------- prompts (guesses until we see theirs)

SYSTEM_PROMPT = f"""You are solving one competitive programming problem in C++20.

You have two tools:
- run_cpp(code, stdin): compile your C++20 program and run it on any input you choose. Use it to test samples, your own edge cases, brute-force comparisons and timing on large generated inputs.
- submit(code): submit a solution. It reports how the solution does on the sample tests. You may submit up to {MAX_SUBMISSIONS} times.

Scoring: the LAST submission that compiles is graded on hidden tests. Small hidden tests are worth 40 points and large ones 60. A slower solution that is correct on small tests still earns those 40 points, but a later submission that compiles and is wrong replaces it.
Limits per test: {TIME_LIMIT_S:g} seconds and {MEMORY_MB // 1024} GiB of memory.
Ties are broken by fewer total turns, so don't waste turns.

When you are finished, reply with exactly: EXIT"""


def _sample_files(task: AgentTask) -> str:
    parts = []
    for k, (inp, out) in enumerate(task.samples, 1):
        parts.append(f"--- sample{k}.in ---\n{inp.rstrip()}\n--- sample{k}.out ---\n{out.rstrip()}")
    return "\n\n".join(parts)


def render_task(task: AgentTask) -> str:
    """The user message: problem.txt contents, the sample files, and the limits."""
    return (
        f"--- problem.txt ---\n{task.title.upper()}\n\n{task.statement}\n\n"
        f"Sample test files:\n\n{_sample_files(task)}\n\n"
        f"Limits: {task.time_limit_s:g} seconds and {task.memory_mb // 1024} GiB of memory per test."
    )

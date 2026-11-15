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

SYSTEM_PROMPT_TEMPLATE = """You are solving one competitive programming problem in C++20.

You have two tools:
- run_cpp(code, stdin): compile your C++20 program and run it on any input you choose. Use it to test samples, your own edge cases, brute-force comparisons and timing on large generated inputs.
- submit(code): submit a solution. It reports how the solution does on the sample tests. You may submit up to {max_submissions} times.

Scoring: the LAST submission that compiles is graded on hidden tests. Small hidden tests are worth 40 points and large ones 60. A slower solution that is correct on small tests still earns those 40 points, but a later submission that compiles and is wrong replaces it.
Limits per test: {limits}.
Ties are broken by fewer total turns, so don't waste turns.

When you are finished, reply with exactly: EXIT"""


def format_limits(time_limit_s: float, memory_mb: int) -> str:
    mem = f"{memory_mb // 1024} GiB" if memory_mb % 1024 == 0 else f"{memory_mb} MB"
    unit = "second" if time_limit_s == 1 else "seconds"
    return f"{time_limit_s:g} {unit} and {mem} of memory"


def system_prompt(task: "AgentTask | None" = None) -> str:
    """The system prompt with this task's limits (identical text to the baseline for 2 s / 1 GiB tasks)."""
    t, m = (task.time_limit_s, task.memory_mb) if task else (TIME_LIMIT_S, MEMORY_MB)
    return SYSTEM_PROMPT_TEMPLATE.format(max_submissions=MAX_SUBMISSIONS, limits=format_limits(t, m))


SYSTEM_PROMPT = system_prompt()  # default limits; kept for backwards compatibility


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
        f"Limits: {format_limits(task.time_limit_s, task.memory_mb)} per test."
    )


def task_from_codeforces(p: dict) -> AgentTask:
    """A Codeforces problem (cppgrade.codeforces index row) in the competition task format."""
    from cppgrade.codeforces import statement_body

    return AgentTask(
        question_id=p["id"], title=p["title"], statement=statement_body(p),
        difficulty=str(p.get("rating") or "?"), contest_date=str(p.get("contest_start", "")),
        samples=tuple((e["input"], e["output"]) for e in p["examples"]),
        tests_file=p["tests_file"], n_tests=p["n_tests"],
        time_limit_s=float(p["time_limit"]), memory_mb=max(256, int(p["memory_limit"])),
    )


# ---------------------------------------------------------------- the competition harness's prompt
# Verbatim from github.com/vals-ai/hackathon-post-training-agent (src/post_training_agent/utils.py).
# The harness sends it as the only (user) message, with no system prompt besides the tool definitions.
VALS_INSTRUCTIONS_PROMPT = """
You will solve a competitive programming problem.

Write your solution as a complete C++20 program that reads standard input and writes standard output. Enclose code in three backticks, for example:

```
[your code here]
```

You have two tools. cpp_executor compiles a program and runs it on stdin you choose, for experiments, brute forces and stress tests. submission saves a solution as your answer and runs it on the sample tests from the statement. Your last submission that compiles is graded on hidden tests, under the time limit, with partial credit for each test it passes.

You may make at most 50 submissions in at most 100 turns. When you are done and do not want to submit again, respond with 'EXIT'.

The problem statement follows:
{question}
"""


def render_task_vals(task: AgentTask) -> str:
    """The harness's user message: its instructions + the problem statement file (title, statement with
    examples, limits)."""
    question = f"{task.title}\n\n{task.statement}"
    if "time limit" not in task.statement.lower():
        question += f"\n\nLimits: {format_limits(task.time_limit_s, task.memory_mb)} per test."
    return VALS_INSTRUCTIONS_PROMPT.format(question=question)

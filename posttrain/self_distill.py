"""Self-distillation with privileged context (on-policy; cookbook SDFT loop).

The same model plays both roles:
  student  sees only the problem, generates a solution (on-policy, its own reasoning)
  teacher  frozen base model, sees the problem PLUS a privileged context, scores the student's
           tokens; the student is trained toward the teacher's top-K token distribution
So the student learns "what I would have written if I had known X", without needing X at test
time. No external teacher, tokenizer matches by construction, everything runs on Tinker.

Context types (built by scripts/build_sd_data.py):
  editorial  sketch of the intended solution (code stripped); open-r1/codeforces `editorial`
  tags       the problem's topic tags only (weak hint)
  reference  a correct solution (from rft-single accepted answers, or any accepted code)
  feedback   one of the model's own failed attempts + its verdict (what went wrong)

Data JSONL, one row per problem:
  {"problem_id", "question", "context", "context_type", "rating", "pass_rate"?}

Gate before spending: Test A measures whether a hint actually helps the model (hint uplift). If the
hinted model isn't much better than the plain one, the teacher has nothing to teach.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from functools import partial

from tinker_cookbook import renderers
from tinker_cookbook.distillation.datasets import PromptOnlyEnv
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder

from posttrain.sft_data import read_jsonl

# The teacher's user message. {question} is the student's exact prompt, {golden_answer} the context.
TEACHER_TEMPLATE = (
    "{question}\n\n"
    "---\n{golden_answer}\n---\n\n"
    "Use this only to guide your own reasoning. Solve the problem as if you had worked it out yourself, "
    "without mentioning the hint, and put your complete C++20 solution in a single ```cpp code block."
)

MAX_CONTEXT_CHARS = 8000  # ~2-3K tokens; leaves room for 24K+ tokens of student reasoning in 32K


def _cap(text: str, n: int = MAX_CONTEXT_CHARS) -> str:
    return text if len(text) <= n else text[:n] + "\n[...truncated]"


def context_editorial(p: dict) -> str | None:
    from cppgrade.codeforces import strip_code

    ed = strip_code(p.get("editorial") or "")
    return f"Hint: a sketch of the intended solution, from the official editorial:\n{_cap(ed)}" if ed else None


def context_tags(p: dict) -> str | None:
    tags = [t for t in (p.get("tags") or []) if not t.startswith("*")]
    return f"Hint: this problem's topics are: {', '.join(tags)}." if tags else None


def context_reference(code: str) -> str:
    return f"Hint: a correct solution (it passes all hidden tests):\n```cpp\n{_cap(code.strip())}\n```"


def context_feedback(fail: dict) -> str:
    verdict = fail["verdict"].replace("_", " ") + (f" ({fail['subtype']})" if fail.get("subtype") else "")
    detail = fail.get("detail", "").strip()
    progress = f" It passed {fail['passed']} of {fail['total']} tests." if fail.get("total") else ""
    return (f"Hint: a previous attempt at this problem, and how it was judged:\n```cpp\n{_cap(fail['code'].strip(), 6000)}\n```\n"
            f"Verdict: {verdict}.{progress}" + (f"\n{_cap(detail, 1500)}" if detail else "") +
            "\nFind and avoid its mistake.")


class SelfDistillDataset:
    """SDFTBatchProvider over our JSONL: (builders, student prompts, privileged contexts) per batch."""

    def __init__(self, rows: list[dict], batch_size: int, group_size: int, renderer: renderers.Renderer,
                 seed: int = 0, epochs: int = 1):
        rows = list(rows)
        order: list[dict] = []
        for e in range(epochs):
            r = rows[:]
            random.Random(seed + e).shuffle(r)
            order += r
        self.rows, self.batch_size, self.group_size, self.renderer = order, batch_size, group_size, renderer

    @classmethod
    def from_files(cls, files: Sequence[str], **kw) -> "SelfDistillDataset":
        return cls([r for f in files for r in read_jsonl(f)], **kw)

    def get_batch(self, index: int) -> tuple[Sequence[EnvGroupBuilder], list[str], list[str]]:
        batch = self.rows[index * self.batch_size : (index + 1) * self.batch_size]
        builders = [
            ProblemGroupBuilder(env_thunk=partial(PromptOnlyEnv, r["question"], self.renderer),
                                num_envs=self.group_size, dataset_name=f"sd_{r['context_type']}")
            for r in batch
        ]
        return builders, [r["question"] for r in batch], [r["context"] for r in batch]

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

"""On-policy distillation (cookbook SDFT loop): from the student's own base model with a privileged
context (self-distillation), or from a different, bigger teacher model.

Self-distillation (the default): the same model plays both roles:
  student  sees only the problem, generates a solution (on-policy, its own reasoning)
  teacher  frozen base model, sees the problem PLUS a privileged context, scores the student's
           tokens; the student is trained toward the teacher's top-K token distribution
So the student learns "what I would have written if I had known X", without needing X at test
time. No external teacher, tokenizer matches by construction, everything runs on Tinker.

Context types (built by scripts/build_distill_data.py; `none` for a different teacher on plain prompts):
  editorial  sketch of the intended solution (code stripped); open-r1/codeforces `editorial`
  tags       the problem's topic tags only (weak hint)
  reference  a correct solution (from rft-single accepted answers, or any accepted code)
  feedback   one of the model's own failed attempts + its verdict (what went wrong)

Data JSONL, one row per problem:
  {"problem_id", "question", "context", "context_type", "rating", "pass_rate"?}

Gate before spending: Test A measures whether a hint actually helps the model (hint uplift). If the
hinted model isn't much better than the plain one, the teacher has nothing to teach.

Teacher model: by default the student's own base model (self-distillation). With a different
teacher (e.g. Qwen3.8-27B -> Qwen3.5-9B; they share a tokenizer) this is ordinary on-policy
distillation: the student samples, the bigger model scores those same tokens. Teacher prompt:
  same  the student's exact prompt (pure on-policy distillation; the default for a different teacher)
  hint  the student's prompt + the privileged context (TEACHER_TEMPLATE; the default for self-distillation)
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import threading
from collections.abc import Sequence
from functools import partial
from pathlib import Path

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

# Teacher sees exactly the student's prompt (build_sdft_teacher_prompt renders the same single user message).
SAME_PROMPT_TEMPLATE = "{question}"

TEACHER_PROMPTS = {"hint": TEACHER_TEMPLATE, "same": SAME_PROMPT_TEMPLATE}

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


# ---------------------------------------------------------------- different teacher model


@contextlib.contextmanager
def teacher_redirect(student_model: str, teacher_model: str, teacher_checkpoint: str | None = None):
    """Make the cookbook SDFT loop use a different teacher.

    sdft.main creates its teacher with exactly one call,
    `service_client.create_sampling_client(base_model=cfg.model_name)`. While this context is active,
    a call with base_model == student model and no model_path is redirected to the teacher. Student
    sampling clients come from the training client and are not affected. Raises if the redirect
    didn't happen exactly once, so a cookbook change can't silently fall back to self-distillation.
    """
    import tinker

    if teacher_model == student_model and teacher_checkpoint is None:
        yield {"redirects": 0}
        return
    orig = tinker.ServiceClient.create_sampling_client
    state = {"redirects": 0}

    def patched(self, model_path=None, base_model=None, *a, **kw):
        if base_model == student_model and model_path is None:
            state["redirects"] += 1
            print(f"[teacher_redirect] teacher sampling client: {teacher_model}"
                  + (f" @ {teacher_checkpoint}" if teacher_checkpoint else "")
                  + " (the cookbook's next log line names the student; ignore it)", flush=True)
            return orig(self, model_path=teacher_checkpoint, base_model=teacher_model, *a, **kw)
        return orig(self, model_path=model_path, base_model=base_model, *a, **kw)

    tinker.ServiceClient.create_sampling_client = patched
    try:
        yield state
    finally:
        tinker.ServiceClient.create_sampling_client = orig
    if state["redirects"] != 1:
        raise RuntimeError(f"teacher redirect fired {state['redirects']} times (expected 1): "
                           "the cookbook SDFT loop changed; check how it creates its teacher")


def check_same_tokenizer(student_model: str, teacher_model: str) -> None:
    """Distillation scores the student's token ids with the teacher: the vocabularies must be identical."""
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    s, t = get_tokenizer(student_model), get_tokenizer(teacher_model)
    probe = "#include <bits/stdc++.h>\nint main(){ std::vector<long long> v; } <think> </think> 1e18 \u2264 \u0394"
    if len(s) != len(t) or s.encode(probe) != t.encode(probe) or s.eos_token_id != t.eos_token_id:
        raise SystemExit(f"{student_model} and {teacher_model} don't share a tokenizer "
                         f"(vocab {len(s)} vs {len(t)}); token-level distillation needs identical ids.")


# ---------------------------------------------------------------- student answer lengths


class _LengthLog:
    """Thread-safe append-only JSONL of student answer lengths (the loop only logs means)."""

    def __init__(self, path: str | None, stop_ids: Sequence[int]):
        self.path, self.stop_ids, self.lock = path, set(stop_ids), threading.Lock()
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def add(self, batch: int, problem_id: str, tokens: list[int]) -> None:
        if not self.path:
            return
        rec = {"batch": batch, "problem_id": problem_id, "n_tokens": len(tokens),
               "cut_off": not tokens or tokens[-1] not in self.stop_ids}
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


class _LoggedPromptOnlyEnv(PromptOnlyEnv):
    def __init__(self, prompt, renderer, *, log: _LengthLog, batch: int, problem_id: str):
        super().__init__(prompt, renderer)
        self._log, self._batch, self._pid = log, batch, problem_id

    async def step(self, action, *, extra=None):
        self._log.add(self._batch, self._pid, list(action))
        return await super().step(action, extra=extra)


class DistillDataset:
    """SDFTBatchProvider over our JSONL: (builders, student prompts, privileged contexts) per batch."""

    def __init__(self, rows: list[dict], batch_size: int, group_size: int, renderer: renderers.Renderer,
                 seed: int = 0, epochs: int = 1, length_log_path: str | None = None):
        rows = list(rows)
        order: list[dict] = []
        for e in range(epochs):
            r = rows[:]
            random.Random(seed + e).shuffle(r)
            order += r
        self.rows, self.batch_size, self.group_size, self.renderer = order, batch_size, group_size, renderer
        self.log = _LengthLog(length_log_path, renderer.get_stop_sequences())

    @classmethod
    def from_files(cls, files: Sequence[str], **kw) -> "DistillDataset":
        return cls([r for f in files for r in read_jsonl(f)], **kw)

    def get_batch(self, index: int) -> tuple[Sequence[EnvGroupBuilder], list[str], list[str]]:
        batch = self.rows[index * self.batch_size : (index + 1) * self.batch_size]
        builders = [
            ProblemGroupBuilder(env_thunk=partial(_LoggedPromptOnlyEnv, r["question"], self.renderer, log=self.log,
                                                  batch=index, problem_id=r["problem_id"]),
                                num_envs=self.group_size, dataset_name=f"sd_{r['context_type']}")
            for r in batch
        ]
        return builders, [r["question"] for r in batch], [r["context"] for r in batch]

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

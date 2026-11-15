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

    def add(self, batch: int, problem_id: str, tokens: list[int], **extra) -> None:
        if not self.path:
            return
        rec = {"batch": batch, "problem_id": problem_id, "n_tokens": len(tokens),
               "cut_off": not tokens or tokens[-1] not in self.stop_ids, **extra}
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


# Forced stop (--cutoff force): a student answer cut off while still thinking gets "\n</think>\n\n"
# appended (as observation tokens: not trained) and one more sampling call to write the solution, so
# the teacher scores and the student trains on a complete answer. Answers still cut off are dropped.
FORCED_STOP_TEXT = "\n</think>\n\n"


class _LoggedPromptOnlyEnv(PromptOnlyEnv):
    def __init__(self, prompt, renderer, *, log: _LengthLog, batch: int, problem_id: str,
                 force_stop: list[int] | None = None, think_end: int | None = None):
        super().__init__(prompt, renderer)
        self._log, self._batch, self._pid = log, batch, problem_id
        self._force, self._think_end, self._ob, self._turn = force_stop, think_end, None, 0

    async def initial_observation(self):
        ob, stop = await super().initial_observation()
        self._ob = ob
        return ob, stop

    async def step(self, action, *, extra=None):
        from tinker_cookbook.rl.types import StepResult

        tokens = list(action)
        cut = not tokens or tokens[-1] not in self._log.stop_ids
        if self._turn == 0:
            force = bool(self._force and cut and self._think_end not in tokens)
            self._log.add(self._batch, self._pid, tokens, forced=force)
            if force:
                import tinker

                self._turn = 1
                nxt = tinker.ModelInput.from_ints(self._ob.to_ints() + tokens + list(self._force))
                return StepResult(reward=0.0, episode_done=False, next_observation=nxt,
                                  next_stop_condition=self.stop_condition, metrics={"forced_stop": 1.0})
        else:
            self._log.add(self._batch, self._pid, tokens, kind="forced_answer")
        return await super().step(action, extra=extra)


class DistillDataset:
    """SDFTBatchProvider over our JSONL: (builders, student prompts, privileged contexts) per batch.
    cutoff: keep | drop | force (see FORCED_STOP_TEXT; drop/force also need `training_controls`)."""

    def __init__(self, rows: list[dict], batch_size: int, group_size: int, renderer: renderers.Renderer,
                 seed: int = 0, epochs: int = 1, length_log_path: str | None = None, cutoff: str = "keep"):
        rows = list(rows)
        order: list[dict] = []
        for e in range(epochs):
            r = rows[:]
            random.Random(seed + e).shuffle(r)
            order += r
        self.rows, self.batch_size, self.group_size, self.renderer = order, batch_size, group_size, renderer
        self.log = _LengthLog(length_log_path, renderer.get_stop_sequences())
        self.force_stop = self.think_end = None
        if cutoff == "force":
            tok = renderer.tokenizer
            self.force_stop = tok.encode(FORCED_STOP_TEXT, add_special_tokens=False)
            (self.think_end,) = tok.encode("</think>", add_special_tokens=False)

    @classmethod
    def from_files(cls, files: Sequence[str], **kw) -> "DistillDataset":
        return cls([r for f in files for r in read_jsonl(f)], **kw)

    def get_batch(self, index: int) -> tuple[Sequence[EnvGroupBuilder], list[str], list[str]]:
        batch = self.rows[index * self.batch_size : (index + 1) * self.batch_size]
        builders = [
            ProblemGroupBuilder(env_thunk=partial(_LoggedPromptOnlyEnv, r["question"], self.renderer, log=self.log,
                                                  batch=index, problem_id=r["problem_id"],
                                                  force_stop=self.force_stop, think_end=self.think_end),
                                num_envs=self.group_size, dataset_name=f"sd_{r['context_type']}")
            for r in batch
        ]
        return builders, [r["question"] for r in batch], [r["context"] for r in batch]

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)


# ---------------------------------------------------------------- training controls (patches on sdft)


def lr_at(step: int, total: int, base: float, schedule: str = "constant", warmup: int = 0,
          min_frac: float = 0.0) -> float:
    """Linear warmup over `warmup` steps, then constant / linear / cosine decay to min_frac * base."""
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    if schedule == "constant":
        return base
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    f = 1 - p if schedule == "linear" else 0.5 * (1 + math.cos(math.pi * p))
    return base * (min_frac + (1 - min_frac) * f)


class TrainState:
    """Shared between the patches and the eval guard: the step the next eval belongs to."""

    def __init__(self, step: int = 0):
        self.step = step
        self.cutoff_stats: dict[str, float] = {}


@contextlib.contextmanager
def training_controls(state: TrainState, lr_fn, cutoff: str, stop_ids: Sequence[int]):
    """Per-step learning rate and cut-off dropping for the cookbook SDFT loop, which has neither.

    sdft.main calls `train_step` / `_train_step_reverse_kl` (module globals) with a constant
    cfg.learning_rate and a metrics dict holding "progress/batch"; the wrappers swap in lr_fn(step).
    With cutoff drop/force, `assemble_training_data` drops datums whose last token isn't a stop
    token (never finished), before the teacher scores them, so they cost sampling only. A step with
    nothing left skips the optimizer step (logged as distill/skipped_step)."""
    from tinker_cookbook.distillation import sdft

    stop = set(stop_ids)
    orig_train, orig_rev, orig_assemble = sdft.train_step, sdft._train_step_reverse_kl, sdft.assemble_training_data

    def assemble(*a, **kw):
        data_D, metadata_D = orig_assemble(*a, **kw)
        if cutoff == "keep":
            return data_D, metadata_D
        keep = [i for i, d in enumerate(data_D) if int(d.loss_fn_inputs["target_tokens"].data[-1]) in stop]
        state.cutoff_stats = {"distill/datums_total": float(len(data_D)), "distill/datums_kept": float(len(keep)),
                              "distill/cutoff_dropped_frac": 1 - len(keep) / max(1, len(data_D))}
        return [data_D[i] for i in keep], [metadata_D[i] for i in keep]

    def wrap(orig):
        async def run(*a, learning_rate: float, metrics: dict | None = None, **kw):
            step = int((metrics or {}).get("progress/batch", state.step))
            lr = lr_fn(step)
            if metrics is not None:
                metrics["optim/lr"] = lr
                metrics.update(state.cutoff_stats)
            data = kw.get("data_D", a[0] if a else None)
            try:
                if not data:
                    if metrics is not None:
                        metrics["distill/skipped_step"] = 1.0
                    return []
                return await orig(*a, learning_rate=lr, metrics=metrics, **kw)
            finally:
                state.step = step + 1
                state.cutoff_stats = {}
        return run

    sdft.train_step, sdft._train_step_reverse_kl = wrap(orig_train), wrap(orig_rev)
    sdft.assemble_training_data = assemble
    try:
        yield state
    finally:
        sdft.train_step, sdft._train_step_reverse_kl, sdft.assemble_training_data = orig_train, orig_rev, orig_assemble


class EarlyStop(Exception):
    pass


class GuardedEvaluator:
    """Wraps an in-training evaluator: appends every result to <log_path>/evals.jsonl and, if
    `max_drop` is set, stops the run (EarlyStop, after writing early_stop.json) once `metric` falls
    more than `max_drop` below its first recorded value (step 0)."""

    def __init__(self, inner, state: TrainState, log_path: str, metric: str | None = None,
                 max_drop: float | None = None):
        self.inner, self.state, self.metric, self.max_drop = inner, state, metric, max_drop
        self.path = Path(log_path) / "evals.jsonl"

    def history(self) -> list[dict]:
        return [json.loads(line) for line in open(self.path)] if self.path.exists() else []

    async def __call__(self, sampling_client, final: bool = False) -> dict[str, float]:
        m = await self.inner(sampling_client)
        rec = {"step": self.state.step, "final": final, **m}
        past = [r for r in self.history() if self.metric in r]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        val = m.get(self.metric) if self.metric else None
        n = next((v for k, v in m.items() if k.endswith("/n")), None)
        if self.max_drop is None or val is None or not n:
            return m
        base = past[0] if past else rec
        if base[self.metric] - val > self.max_drop:
            best = max(past + [rec], key=lambda r: r[self.metric])
            info = {"step": rec["step"], "metric": self.metric, "value": val, "baseline": base[self.metric],
                    "baseline_step": base["step"], "max_drop": self.max_drop,
                    "best_step": best["step"], "best_value": best[self.metric]}
            (self.path.parent / "early_stop.json").write_text(json.dumps(info, indent=2))
            raise EarlyStop(f"{self.metric} {val:.3f} at step {rec['step']} is more than {self.max_drop} below "
                            f"step {base['step']}'s {base[self.metric]:.3f}; best was step {best['step']} "
                            f"({best[self.metric]:.3f})")
        return m

"""Agentic RL on the competition environment (cppgrade.agent_env), for the cookbook's rl/train.py loop.

One group = one Codeforces problem from the `rl` part of the training pool, `group_size` independent
competition sessions on it (tools run_cpp + submit; the last compiling submission is graded on hidden
tests, small 40 / large 60).

Reward (computed once per rollout in `compute_group_rewards`, however the episode ended):

    no compiling submission:   0
    score == 0:                0
    fails a statement example: 0   (require_samples, on by default: see below)
    score > 0:                 score / 100 - turn_penalty * turns
    always:                    - malformed_penalty * malformed turns
    optional (off by default): + partial_credit * bonus

Why `require_samples`: in rl-v1, 10 of the 11 rollouts that scored 60 (large tests pass, small fail)
failed the statement example: wrong programs paid by weak large-test groups (e.g. every large test with
the same answer). A graded submission that fails an example would almost surely fail the real hidden
tests, so the score only counts when it passes them all. Costs a little on problems with several valid
outputs (where an exact-match example can reject a correct answer).

`bonus` (0-1, cppgrade.grader.partial_credit): fraction of hidden tests passed beyond what the best
constant-answer program ("YES" for every case, "-1", ...) passes, per small/large group, weighted
40/60. Paid only if the graded submission passes every statement example, so cheap guesses that slip
past the constant baseline mostly get nothing. It is computed and logged for every rollout even when
`partial_credit` is 0, so a run without it shows how many dropped groups it would have rescued
(`scripts/train_rl.py --summarize`).

Why the reward is here and not in the env's `reward_fn`: the cookbook skips `reward_fn` when an episode
ends on a cut-off turn, malformed calls or context overflow, but the competition still grades the last
compiling submission in all of those cases. So the env's own step rewards are all zeroed (reward_fn,
parse penalties, overflow reward) and the full reward is added at group level.

The turn penalty applies only to a positive score: the tiebreak (fewest turns) separates equal scores,
and a penalty on zero-score rollouts would only teach the model to give up early.

Every rollout is also appended to `<log_path>/rollouts.jsonl` (dataset batch, problem, reward, score,
turns, tokens, stop reason, ...). The cookbook drops constant-reward groups before logging metrics,
so its metrics.jsonl only describes the groups that were trained on; rollouts.jsonl has all of them.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import chz
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, Trajectory

from cppgrade.agent_env import EnvSettings, build_competition_env
from cppgrade.grader import partial_credit
from posttrain.problems import agent_task


@dataclass(frozen=True)
class RewardSettings:
    turn_penalty: float = 0.01       # per assistant turn, only when score > 0
    malformed_penalty: float = 0.05  # per turn whose tool calls failed to parse (or broken framing)
    partial_credit: float = 0.0      # weight of the above-trivial partial-credit bonus (0 = off)
    max_trivial: float = 0.5         # no bonus from a group whose constant-answer baseline is above this
    require_samples: bool = True     # score counts only if the graded submission passes every statement example


_POOL = None


def _pool(max_concurrent: int):
    """One on-demand Modal pool per process, created lazily inside the running event loop (it isn't
    picklable, so it can't live on the builder)."""
    global _POOL
    if _POOL is None:
        from cppgrade.grader import make_ondemand_pool

        _POOL = make_ondemand_pool(max_concurrent=max_concurrent)
    return _POOL


@functools.cache
def _renderer(renderer_name: str, model_name: str):
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return renderers.get_renderer(renderer_name, get_tokenizer(model_name), model_name=model_name)


async def _no_env_reward(history):
    return 0.0, {}


def _is_exit_text(env) -> bool:
    last = next((m for m in reversed(env.message_env.history) if m["role"] == "assistant"), None)
    if not last or last.get("tool_calls"):
        return False
    c = last.get("content")
    text = c if isinstance(c, str) else "".join(p.get("text", "") for p in (c or []) if p.get("type") == "text")
    return text.strip().endswith("EXIT")


def rollout_stats(traj: Trajectory) -> dict:
    tr = traj.transitions
    malformed = sum(t.metrics.get("parse_error", 0.0) > 0 for t in tr)
    cut = bool(tr) and tr[-1].metrics.get("max_tokens_reached", 0.0) > 0
    return {
        "turns": len(tr),
        "gen_tokens": sum(len(t.ac.tokens) for t in tr),
        "prompt_tokens_last": tr[-1].ob.length if tr else 0,
        "malformed_turns": int(malformed),
        "cut_off": cut,
        "cut_off_first_turn": cut and len(tr) == 1,
        "stop_reason": traj.stop_reason,
    }


@dataclass
class CompetitionGroupBuilder(EnvGroupBuilder):
    problem: dict                 # posttrain.problems index row (picklable)
    model_name: str
    renderer_name: str
    group_size: int
    env: EnvSettings
    reward: RewardSettings
    batch: int = -1               # dataset batch index, for rollouts.jsonl
    log_path: str | None = None
    pool_size: int = 64

    async def make_envs(self) -> Sequence[Env]:
        task = agent_task(self.problem)
        renderer = _renderer(self.renderer_name, self.model_name)
        envs = []
        for _ in range(self.group_size):
            env, tools = build_competition_env(task, renderer, _pool(self.pool_size), self.env)
            env.message_env.reward_fn = _no_env_reward  # the reward is computed per group, below
            env.competition_tools = tools
            env.t_start = time.time()
            envs.append(env)
        return envs

    async def compute_group_rewards(self, trajectory_group: list[Trajectory], env_group: Sequence[Env]):
        t0 = time.time()
        finals = await asyncio.gather(*(e.competition_tools.final_score() for e in env_group))
        grade_s = time.time() - t0
        out, records = [], []
        for k, (traj, env, final) in enumerate(zip(trajectory_group, env_group, finals)):
            tools = env.competition_tools
            st = rollout_stats(traj)
            score = float(final.score)
            sub = tools.last_compiling
            samples_ok = bool(sub and sub.sample_verdicts and all(v == "accepted" for v in sub.sample_verdicts))
            bonus = partial_credit(final, self.reward.max_trivial) if samples_ok else 0.0
            counted = score if (samples_ok or not self.reward.require_samples) else 0.0
            base = counted / 100.0 - self.reward.turn_penalty * st["turns"] if counted > 0 else 0.0
            base -= self.reward.malformed_penalty * st["malformed_turns"]
            r = base + self.reward.partial_credit * bonus
            exit_text = _is_exit_text(env)
            m = {
                "rl/score": score,
                "rl/score_proportional": float(final.score_proportional),
                "rl/full_100": float(score == 100),
                "rl/no_compiling_submission": float(tools.last_compiling is None),
                "rl/submissions": float(len(tools.submissions)),
                "rl/run_calls": float(tools.n_run_calls),
                "rl/turns": float(st["turns"]),
                "rl/gen_tokens": float(st["gen_tokens"]),
                "rl/malformed_turns": float(st["malformed_turns"]),
                "rl/has_malformed": float(st["malformed_turns"] > 0),
                "rl/cut_off": float(st["cut_off"]),
                "rl/cut_off_first_turn": float(st["cut_off_first_turn"]),
                "rl/exit": float(exit_text),
                "rl/infra_errors": float(tools.n_infra_errors + bool(final.infra_error)),
                "rl/samples_ok": float(samples_ok),
                "rl/score_voided": float(score > 0 and counted == 0),
                "rl/partial_bonus": bonus,
                "rl/bonus_only": float(bonus > 0 and score == 0),
            }
            out.append((r, m))
            records.append({
                "batch": self.batch, "problem_id": self.problem["id"], "rating": self.problem.get("rating"),
                "rollout": k, "reward": round(r, 4), "reward_base": round(base, 4), "score": score, "score_counted": counted,
                "score_proportional": float(final.score_proportional), "bonus": round(bonus, 4),
                "samples_ok": samples_ok, "trivial_small": final.trivial_small, "trivial_large": final.trivial_large,
                "exit": exit_text,
                "submissions": len(tools.submissions), "compiling": sum(s.compiled for s in tools.submissions),
                "run_calls": tools.n_run_calls, "infra_errors": m["rl/infra_errors"],
                "rollout_s": round(t0 - env.t_start, 1), "grade_s": round(grade_s, 1), **st,
            })
        if self.log_path:
            p = Path(self.log_path) / "rollouts.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.writelines(json.dumps(x) + "\n" for x in records)
        return out

    def logging_tags(self) -> list[str]:
        return ["competition"]


@dataclass
class CompetitionDataset(RLDataset):
    problems: list[dict]
    groups_per_batch: int
    make_builder: object = field(repr=False)  # (problem, batch index) -> CompetitionGroupBuilder

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        chunk = self.problems[index * self.groups_per_batch:(index + 1) * self.groups_per_batch]
        return [self.make_builder(p, index) for p in chunk]

    def __len__(self) -> int:
        return len(self.problems) // self.groups_per_batch  # whole batches only


@chz.chz
class CompetitionDatasetBuilder(RLDatasetBuilder):
    """Training set: `posttrain.problems.load_pool(part="rl")`, rating-filtered, seeded order. No test
    set here (in-training evals are LiveCodeBench tasks, via posttrain.evaluators)."""

    model_name: str
    renderer_name: str
    groups_per_batch: int
    group_size: int
    min_rating: int = 1400
    max_rating: int = 2200
    seed: int = 0
    limit: int | None = None
    max_turns: int = 8
    context_window: int = 65536
    max_tokens_per_turn: int = 16384
    max_consecutive_parse_errors: int = 2
    turn_penalty: float = 0.01
    malformed_penalty: float = 0.05
    partial_credit: float = 0.0
    max_trivial: float = 0.5
    require_samples: bool = True
    log_path: str | None = None
    pool_size: int = 64

    def problems(self) -> list[dict]:
        from posttrain.problems import load_pool

        return load_pool(part="rl", min_rating=self.min_rating, max_rating=self.max_rating,
                         seed=self.seed, limit=self.limit)

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        env = EnvSettings(
            max_turns=self.max_turns, context_window=self.context_window,
            max_generation_tokens=self.max_tokens_per_turn,
            max_consecutive_parse_errors=self.max_consecutive_parse_errors,
            parse_error_penalty=0.0,  # the malformed penalty is applied in compute_group_rewards
        )
        reward = RewardSettings(turn_penalty=self.turn_penalty, malformed_penalty=self.malformed_penalty,
                                partial_credit=self.partial_credit, max_trivial=self.max_trivial,
                                require_samples=self.require_samples)

        def make_builder(p: dict, batch: int) -> CompetitionGroupBuilder:
            return CompetitionGroupBuilder(problem=p, model_name=self.model_name, renderer_name=self.renderer_name,
                                           group_size=self.group_size, env=env, reward=reward, batch=batch,
                                           log_path=self.log_path, pool_size=self.pool_size)

        return CompetitionDataset(self.problems(), self.groups_per_batch, make_builder), None

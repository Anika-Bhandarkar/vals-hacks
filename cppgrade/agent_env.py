"""The competition as a cookbook Env: one task, tools run_cpp + submit, graded at episode end.

Same object for evaluation and RL. The episode ends when the model replies without a tool call
(EXIT), hits `max_turns`, makes more than `max_consecutive_parse_errors` malformed tool calls in a
row, or overflows the context window.

Reward (only used for RL; evals read `CompetitionTools.final_score()` directly, because the cookbook
skips reward_fn when an episode ends on parse errors or context overflow, while the competition
still grades the last compiling submission):

    reward = score / 100  -  turn_penalty * turns

`score` is the all-or-nothing 40/60 score, or the proportional one with `proportional=True` (denser
signal, less sensitive to our small/large proxy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy
from tinker_cookbook.tool_use import build_agent_tool_env

from cppgrade.agent_tasks import SYSTEM_PROMPT, AgentTask, render_task
from cppgrade.agent_tools import CompetitionTools


@dataclass
class EnvSettings:
    max_turns: int = 15                  # competition: until EXIT or "out of turns" (number unknown)
    context_window: int = 32768          # model's context (Qwen3-8B: 32K)
    max_generation_tokens: int | None = None  # per-turn cap; None = whatever fits in the context
    max_consecutive_parse_errors: int = 2      # "repeated malformed tool calls end the task" (count unknown)
    turn_penalty: float = 0.0            # RL only: ties go to fewer turns
    proportional: bool = False           # RL only: reward from proportional score
    parse_error_penalty: float = 0.05    # RL only: per malformed turn


def _count_assistant_turns(history: list[Message]) -> int:
    return sum(m["role"] == "assistant" for m in history)


def build_competition_env(task: AgentTask, renderer: Renderer, pool: Any, settings: EnvSettings | None = None):
    """Returns (env, tools). Keep `tools` to read submissions/score after the rollout."""
    s = settings or EnvSettings()
    tools = CompetitionTools(task, pool)
    tool_list = [tools.run_cpp, tools.submit]

    async def reward_fn(history: list[Message]) -> tuple[float, dict[str, float]]:
        res = await tools.final_score()
        score = res.score_proportional if s.proportional else res.score
        turns = _count_assistant_turns(history)
        return score / 100.0 - s.turn_penalty * turns, {
            "score": res.score,
            "score_proportional": res.score_proportional,
            "turns": float(turns),
            "submissions": float(len(tools.submissions)),
            "run_calls": float(tools.n_run_calls),
        }

    initial = renderer.create_conversation_prefix_with_tools(
        tools=[t.to_spec() for t in tool_list], system_prompt=SYSTEM_PROMPT
    ) + [{"role": "user", "content": render_task(task)}]

    env = build_agent_tool_env(
        renderer=renderer,
        tools=tool_list,
        initial_messages=initial,
        reward_fn=reward_fn,
        max_turns=s.max_turns,
        max_trajectory_tokens=s.context_window,
        max_generation_tokens=s.max_generation_tokens,
        context_overflow_reward=0.0,
        parse_error_policy=ParseErrorPolicy(
            max_consecutive=s.max_consecutive_parse_errors, penalty_per_error=s.parse_error_penalty
        ),
    )
    return env, tools

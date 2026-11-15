"""Agentic RL in the competition environment (tools run_cpp + submit, last compiling submission graded
on hidden tests), with the cookbook's rl/train.py loop. Env, tasks and reward: posttrain/rl_env.py.

    # check config, task pool and a worst-case cost, without calling Tinker
    uv run python scripts/train_rl.py --name rl-v1 --dry-run

    # from a distillation checkpoint (a tinker:// *state* path from runs/train/<run>/checkpoints.jsonl)
    uv run python scripts/train_rl.py --name rl-v1 --load-checkpoint tinker://.../weights/final

    # per-step table from a run (also printed at the end of training)
    uv run python scripts/train_rl.py --summarize runs/train/rl-v1

Tasks: Codeforces `rl` part of the training pool (posttrain.problems.PARTS), rating 1400-2200 by default.
Never the `distill` or `holdout` parts.

Output in runs/train/<name>/: metrics.jsonl (cookbook: reward, rl/*, time/*, tokens/*, cost/*),
rollouts.jsonl (one line per rollout, including groups dropped for constant reward), checkpoints.jsonl,
rl_config.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

# Run folders: runs/train/<name> locally; the Modal entry point (scripts/train_rl_modal.py) points this at its Volume.
RUNS_DIR = Path(os.environ.get("VALS_RUNS_DIR", ROOT / "runs" / "train"))

# USD per million tokens. Tinker's published list (tinker-docs .../models.json, read by the cookbook's
# cost tracker); its Qwen3-8B row matches what we were actually billed. Used for the dry-run estimate.
PUBLISHED_USD_PER_M = {
    "Qwen/Qwen3-8B": {"sample": 0.60, "prefill": 0.195, "cached_prefill": 0.039, "train": 0.44},
    "Qwen/Qwen3.5-9B": {"sample": 0.807, "prefill": 0.30, "cached_prefill": 0.06, "train": 1.463},
}


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def summarize(run_dir: Path) -> None:
    rp = run_dir / "rollouts.jsonl"
    if rp.exists():
        by_batch = defaultdict(list)
        for line in open(rp):
            r = json.loads(line)
            by_batch[r["batch"]].append(r)
        print("\n== rollouts by dataset batch (all rollouts, including dropped constant-reward groups)")
        print(f"{'batch':>5} {'n':>3} {'reward':>7} {'score':>6} {'100s':>5} {'nosub':>6} {'cut1st':>6} {'cut':>5} "
              f"{'exit':>5} {'malf':>5} {'subs':>5} {'runs':>5} {'turns':>5} {'gen_tok':>8} {'roll_s':>7} {'grade_s':>7}")
        for b in sorted(by_batch):
            rs = by_batch[b]
            print(f"{b:>5} {len(rs):>3} {_mean(r['reward'] for r in rs):>7.3f} {_mean(r['score'] for r in rs):>6.1f} "
                  f"{_mean(r['score'] == 100 for r in rs):>5.0%} {_mean(r['compiling'] == 0 for r in rs):>6.0%} "
                  f"{_mean(r['cut_off_first_turn'] for r in rs):>6.0%} {_mean(r['cut_off'] for r in rs):>5.0%} "
                  f"{_mean(r['exit'] for r in rs):>5.0%} {_mean(r['malformed_turns'] > 0 for r in rs):>5.0%} "
                  f"{_mean(r['submissions'] for r in rs):>5.1f} {_mean(r['run_calls'] for r in rs):>5.1f} "
                  f"{_mean(r['turns'] for r in rs):>5.1f} {_mean(r['gen_tokens'] for r in rs):>8.0f} "
                  f"{_mean(r['rollout_s'] for r in rs):>7.0f} {_mean(r['grade_s'] for r in rs):>7.0f}")
        groups = defaultdict(list)
        for b, rs in by_batch.items():
            for r in rs:
                groups[(b, r["problem_id"])].append(r["reward"])
        const = sum(len(set(v)) == 1 for v in groups.values())
        print(f"   groups: {len(groups)}, constant reward (no signal): {const}   "
              f"stop reasons: {dict(Counter(json.loads(l)['stop_reason'] for l in open(rp)).most_common())}")
        # Partial-credit bonus (logged even when off): what it would change at weight 0.15.
        rows = [r for rs in by_batch.values() for r in rs if "bonus" in r]
        if rows:
            g2 = defaultdict(list)
            for r in rows:
                g2[(r["batch"], r["problem_id"])].append(r)
            base_const = [v for v in g2.values() if len({r["reward_base"] for r in v}) == 1]
            rescued = sum(len({round(r["reward_base"] + 0.15 * r["bonus"], 4) for r in v}) > 1 for v in base_const)
            bonus_only = [r for r in rows if r["bonus"] > 0 and r["score"] == 0]
            print(f"   partial credit: {sum(r['bonus'] > 0 for r in rows)}/{len(rows)} rollouts earn a bonus, "
                  f"{len(bonus_only)} with score 0 (mean bonus {_mean(r['bonus'] for r in bonus_only):.2f}); "
                  f"{rescued}/{len(base_const)} constant-reward groups would get signal at --partial-credit 0.15; "
                  f"samples all pass: {_mean(r['samples_ok'] for r in rows):.0%}")
    mp = run_dir / "metrics.jsonl"
    if mp.exists():
        rows = [json.loads(l) for l in open(mp)]
        keys = {"reward": "env/all/reward/total", "score": "env/all/rl/score", "sample_s_max": "time/sampling_time_max",
                "train_s": "time/train_step", "ckpt_s": "time/save_checkpoint", "sampled": "tokens/sampled",
                "prefill": "tokens/prefill_uncached", "prefill_cached": "tokens/prefill_cached",
                "trained": "tokens/trained", "usd": "cost/step_usd"}
        print("\n== training steps (cookbook metrics: trained groups only; sample_s_max = slowest rollout)")
        print("step " + " ".join(f"{k:>14}" for k in keys))
        for r in rows:
            if "cost/step_usd" not in r:
                continue
            print(f"{r.get('step', '?'):>4} " + " ".join(
                f"{r[k]:>14.3f}" if isinstance(r.get(k), (int, float)) else f"{'-':>14}" for k in keys.values()))
        last = next((r for r in reversed(rows) if "cost/cumulative_usd" in r), None)
        if last:
            print(f"   cumulative cost estimate (cookbook, published rates): ${last['cost/cumulative_usd']:.2f}")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--renderer", default="qwen3_5")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--load-checkpoint", default=None,
                    help="tinker:// training-state path (…/weights/…), e.g. a distillation checkpoint; fresh optimizer")
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--groups-per-batch", type=int, default=8, help="tasks per step")
    ap.add_argument("--group-size", type=int, default=4, help="rollouts per task")
    ap.add_argument("--max-tokens", type=int, default=16384, help="per turn")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--context-window", type=int, default=65536)
    ap.add_argument("--max-parse-errors", type=int, default=2, help="consecutive malformed calls tolerated")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--turn-penalty", type=float, default=0.01)
    ap.add_argument("--malformed-penalty", type=float, default=0.05)
    ap.add_argument("--partial-credit", type=float, default=0.0,
                    help="weight of the above-trivial partial-credit bonus (posttrain/rl_env.py); 0 = off, try 0.15")
    ap.add_argument("--no-require-samples", action="store_true",
                    help="count the hidden-test score even if the graded submission fails a statement example")
    ap.add_argument("--max-trivial", type=float, default=0.5,
                    help="no bonus from a test group a constant-answer program passes more than this fraction of")
    ap.add_argument("--min-rating", type=int, default=1400)
    ap.add_argument("--max-rating", type=int, default=2200)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N problems of the seeded order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=5, help="full checkpoints (state + sampler weights, kept 7 days)")
    ap.add_argument("--rolling-save-every", type=int, default=1,
                    help="state-only resume points (expire after 2h); a preempted run resumes from the latest")
    ap.add_argument("--max-steps-off-policy", type=int, default=1,
                    help="async: sampling may run this many steps ahead of training; -1 = synchronous loop")
    ap.add_argument("--keep-constant-groups", action="store_true",
                    help="train on groups where every rollout got the same reward (default: drop them)")
    ap.add_argument("--rollout-retries", type=int, default=2, help="retries for a rollout that raises (Modal flakes)")
    ap.add_argument("--pool", type=int, default=64, help="max concurrent Modal sandboxes")
    ap.add_argument("--eval-agentic", type=int, default=0, help="N LiveCodeBench tasks every --eval-every steps (0=off)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--if-exists", default="raise", choices=["raise", "resume", "delete", "ask"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.summarize:
        summarize(args.summarize)
        return
    if not args.name:
        ap.error("--name is required")
    if args.load_checkpoint and "sampler_weights" in args.load_checkpoint:
        ap.error("--load-checkpoint needs a training-state path (tinker://…/weights/…), not sampler_weights; "
                 "see the state path in the run's checkpoints.jsonl")
    if args.max_tokens > args.context_window:
        ap.error("--max-tokens must fit in --context-window")

    from posttrain.rl_env import CompetitionDatasetBuilder

    log_path = str(RUNS_DIR / args.name)
    builder = CompetitionDatasetBuilder(
        model_name=args.model, renderer_name=args.renderer, groups_per_batch=args.groups_per_batch,
        group_size=args.group_size, min_rating=args.min_rating, max_rating=args.max_rating, seed=args.seed,
        limit=args.limit, max_turns=args.max_turns, context_window=args.context_window,
        max_tokens_per_turn=args.max_tokens, max_consecutive_parse_errors=args.max_parse_errors,
        turn_penalty=args.turn_penalty, malformed_penalty=args.malformed_penalty,
        partial_credit=args.partial_credit, max_trivial=args.max_trivial,
        require_samples=not args.no_require_samples, log_path=log_path,
        pool_size=args.pool,
    )

    if args.dry_run:
        probs = builder.problems()
        n_batches = len(probs) // args.groups_per_batch
        rates = PUBLISHED_USD_PER_M.get(args.model)
        rollouts = args.groups_per_batch * args.group_size
        worst_tokens = rollouts * args.max_turns * args.max_tokens
        print(json.dumps({
            "model": args.model, "renderer": args.renderer, "load_checkpoint": args.load_checkpoint,
            "problems": len(probs), "part": "rl",
            "ratings": dict(sorted(Counter((p.get("rating") or 0) // 200 * 200 for p in probs).items())),
            "batches_available": n_batches, "max_steps": args.max_steps,
            "rollouts_per_step": rollouts, "partial_credit": args.partial_credit,
            "async": args.max_steps_off_policy >= 0, "drop_constant_groups": not args.keep_constant_groups,
            "worst_case_generated_tokens_per_step": worst_tokens,
            "worst_case_usd_per_step_sample_plus_train": (round(worst_tokens / 1e6 * (rates["sample"] + rates["train"]), 2)
                                                          if rates else "unknown rates"),
            "note": "prefill is extra (every turn re-reads the conversation, mostly cache hits); "
                    "the smoke test's cost/step_usd is the real number",
            "log_path": log_path,
        }, indent=2))
        if n_batches < args.max_steps:
            print(f"WARNING: only {n_batches} batches for {args.max_steps} steps (and dropped groups use more)")
        return

    from tinker_cookbook import cli_utils
    from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
    from tinker_cookbook.rl.train import AsyncConfig, Config
    from tinker_cookbook.rl.train import main as rl_main

    evaluators = []
    if args.eval_agentic:
        from posttrain.evaluators import AgenticEvaluator

        # the cookbook wants builders (it calls each one to get the evaluator), not evaluators
        evaluators.append(lambda: AgenticEvaluator(model_name=args.model, renderer_name=args.renderer,
                                                   limit=args.eval_agentic, max_turns=args.max_turns,
                                                   max_tokens=args.max_tokens, context_window=args.context_window))
    cfg = Config(
        learning_rate=args.lr, dataset_builder=builder, model_name=args.model, recipe_name="vals_hacks_rl",
        renderer_name=args.renderer, lora_rank=args.lora_rank, max_tokens=args.max_tokens,
        temperature=args.temperature, log_path=log_path, load_checkpoint_path=args.load_checkpoint,
        save_every=args.save_every, rolling_save_every=args.rolling_save_every, eval_every=args.eval_every if evaluators else 0, evaluator_builders=evaluators,
        remove_constant_reward_groups=not args.keep_constant_groups, max_steps=args.max_steps,
        async_config=(AsyncConfig(max_steps_off_policy=args.max_steps_off_policy, groups_per_batch=args.groups_per_batch)
                      if args.max_steps_off_policy >= 0 else None),
        rollout_error_tolerance=RetryOnFailure(max_retries=args.rollout_retries),
        session_name=args.name,
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=args.if_exists)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    (Path(log_path) / "rl_config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    asyncio.run(rl_main(cfg))
    summarize(Path(log_path))
    ck = Path(log_path) / "checkpoints.jsonl"
    if ck.exists():
        last = json.loads(ck.read_text().strip().splitlines()[-1])
        print(f"\nlast checkpoint: {json.dumps(last)}")
        sp = last.get("sampler_path")
        if sp:
            print(f"evaluate: uv run python scripts/eval_agentic.py --model {args.model} --model-path {sp} "
                  f"--max-tokens {args.max_tokens} --max-turns {args.max_turns} --limit 50 --name agent-{args.name}")


if __name__ == "__main__":
    main()

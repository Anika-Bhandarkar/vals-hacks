"""Self-distillation on Tinker: student = LoRA on the plain prompt, teacher = frozen base model on the
prompt + privileged context (posttrain/self_distill.py). Wraps the cookbook SDFT loop (top-K distillation).

    # check data, a rendered teacher prompt, token budget and a rough cost (no Tinker calls)
    uv run python scripts/train_self_distill.py --name sd-editorial-v1 --data data/sd/editorial.jsonl --dry-run

    # train
    uv run python scripts/train_self_distill.py --name sd-editorial-v1 --data data/sd/editorial.jsonl \
        --batch-size 16 --group-size 4 --save-every 10 --eval-single 30 --eval-every 10

    # after SFT/RL: start from that checkpoint (teacher stays the base model unless --teacher-sync-every)
    ... --load-checkpoint tinker://<run>/weights/<name>

Output: runs/train/<name>/ (metrics.jsonl, checkpoints.jsonl). Evaluate the final sampler checkpoint
with scripts/eval_agentic.py / eval_lcb.py as usual.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

USD_PER_M_SAMPLED = 0.11 / 1.63  # measured: ~$0.11 per 1.63M sampled tokens (Tinker + Modal), single-attempt eval


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="JSONL from build_sd_data.py")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--renderer", default=None)
    ap.add_argument("--load-checkpoint", default=None)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4, help="cookbook: 5e-4..1e-3 for LoRA, lower end recommended")
    ap.add_argument("--batch-size", type=int, default=16, help="problems per step")
    ap.add_argument("--group-size", type=int, default=4, help="student samples per problem")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=24000, help="student generation budget")
    ap.add_argument("--max-context-length", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=20, help="teacher top-K tokens (0 = importance-sampling fallback)")
    ap.add_argument("--reverse", action="store_true", help="reverse KL over the top-K (mode-seeking)")
    ap.add_argument("--teacher-sync-every", type=int, default=None, help="copy student into teacher every N steps")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-single", type=int, default=0, help="N held-out LCB problems, single attempt")
    ap.add_argument("--eval-agentic", type=int, default=0, help="N held-out LCB tasks, competition format")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--if-exists", default="raise", choices=["raise", "resume", "delete", "ask"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from collections import Counter

    from tinker_cookbook import renderers
    from tinker_cookbook.distillation.sdft import build_sdft_teacher_prompt
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from posttrain.self_distill import TEACHER_TEMPLATE, SelfDistillDataset
    from posttrain.sft_data import read_jsonl

    renderer_name = args.renderer or get_recommended_renderer_name(args.model)
    tok = get_tokenizer(args.model)
    renderer = renderers.get_renderer(renderer_name, tok, model_name=args.model)
    rows = [r for f in args.data for r in read_jsonl(f)]
    if not rows:
        raise SystemExit("no rows in --data")

    # Budget check: teacher prompt + student completion must fit the context, or completions get cut.
    t_lens = [build_sdft_teacher_prompt(r["question"], r["context"], renderer, demo_template=TEACHER_TEMPLATE).length for r in rows]
    s_lens = [renderer.build_generation_prompt([{"role": "user", "content": r["question"]}]).length for r in rows]
    room = [args.max_context_length - t for t in t_lens]
    too_tight = sum(x < args.max_tokens for x in room)
    steps = -(-len(rows) * args.epochs // args.batch_size)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    sampled_per_step = args.batch_size * args.group_size * 15_000  # ~15K tokens per thinking answer (measured mean)
    summary = {
        "problems": len(rows), "context_types": dict(Counter(r["context_type"] for r in rows)),
        "steps": steps, "renderer": renderer_name, "lr": args.lr, "lora_rank": args.lora_rank,
        "student_prompt_tokens_max": max(s_lens), "teacher_prompt_tokens_max": max(t_lens),
        "problems_where_teacher_context_cuts_completion_below_max_tokens": too_tight,
        "approx_sampled_tokens_per_step": sampled_per_step,
        "approx_generation_usd_per_step": round(sampled_per_step / 1e6 * USD_PER_M_SAMPLED, 2),
        "approx_generation_usd_total": round(steps * sampled_per_step / 1e6 * USD_PER_M_SAMPLED, 2),
        "note": "generation only (teacher scoring + training extra); from measured single-attempt eval rates",
        "log_path": str(ROOT / "runs" / "train" / args.name),
    }
    print(json.dumps(summary, indent=2))
    i = max(range(len(rows)), key=lambda k: t_lens[k])
    print("\n--- example teacher user message (longest) ---\n" + TEACHER_TEMPLATE.format(question=rows[i]["question"], golden_answer=rows[i]["context"])[-1800:])
    if args.dry_run:
        return

    from tinker_cookbook import cli_utils
    from tinker_cookbook.distillation import sdft

    from posttrain.evaluators import AgenticEvaluator, SingleAttemptEvaluator

    evaluators = []
    if args.eval_single:
        evaluators.append(lambda: SingleAttemptEvaluator(model_name=args.model, renderer_name=renderer_name, limit=args.eval_single))
    if args.eval_agentic:
        evaluators.append(lambda: AgenticEvaluator(model_name=args.model, renderer_name=renderer_name, limit=args.eval_agentic))
    log_path = str(ROOT / "runs" / "train" / args.name)
    cfg = sdft.Config(
        model_name=args.model, recipe_name="vals_hacks_self_distill", renderer_name=renderer_name,
        lora_rank=args.lora_rank, learning_rate=args.lr, max_tokens=args.max_tokens, temperature=args.temperature,
        topk=args.topk, reverse=args.reverse, demo_template=TEACHER_TEMPLATE, teacher_sync_every=args.teacher_sync_every,
        max_context_length=args.max_context_length, evaluator_builders=evaluators,
        eval_every=args.eval_every if evaluators else 0, save_every=args.save_every, log_path=log_path,
        wandb_project=args.wandb_project, wandb_name=args.name if args.wandb_project else None, session_name=args.name,
        load_checkpoint_path=args.load_checkpoint, max_steps=args.max_steps,
    )
    dataset = SelfDistillDataset(rows, batch_size=args.batch_size, group_size=args.group_size, renderer=renderer,
                                 seed=args.seed, epochs=args.epochs)
    cli_utils.check_log_dir(log_path, behavior_if_exists=args.if_exists)
    asyncio.run(sdft.main(cfg, dataset))
    ck = Path(log_path) / "checkpoints.jsonl"
    if ck.exists():
        last = [json.loads(line) for line in open(ck)][-1]
        print(f"\nfinal checkpoint: {json.dumps(last)}")
        if last.get("sampler_path"):
            print(f"evaluate: uv run python scripts/eval_agentic.py --model {args.model} --model-path {last['sampler_path']} "
                  f"--limit 50 --name agent-{args.name}")


if __name__ == "__main__":
    main()

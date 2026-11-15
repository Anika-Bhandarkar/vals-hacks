"""On-policy distillation on Tinker (cookbook SDFT loop, top-K distillation): self-distillation or a teacher model.

  default            student = LoRA on --model, teacher = frozen --model with the hint (self-distillation)
  --teacher-model X  teacher = frozen X (must share the student's tokenizer), scoring the student's own
                     tokens; teacher prompt defaults to the student's exact prompt (--teacher-prompt same)

    # 27B -> 9B on-policy distillation, plain prompts
    uv run python scripts/build_distill_data.py --context none --out data/distill/plain.jsonl
    uv run python scripts/train_distill.py --name opd-27b-9b-v1 --data data/distill/plain.jsonl \
        --model Qwen/Qwen3.5-9B --teacher-model Qwen/Qwen3.8-27B --lora-rank 32 --max-tokens 20000
    #   ...or let the 27B also see the editorial: --teacher-prompt hint with a --context editorial file

    # check data, a rendered teacher prompt, token budget and a rough cost (no Tinker calls)
    uv run python scripts/train_distill.py --name sd-editorial-v1 --data data/distill/editorial.jsonl --dry-run

    # train
    uv run python scripts/train_distill.py --name sd-editorial-v1 --data data/distill/editorial.jsonl \
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

# Billed Tinker rates (tinker billing usage, Nov 2026), USD per million tokens. Only Qwen3-8B is measured so far;
# other models fall back to it and are flagged. Re-check with:  uv run tinker billing usage <start> <end> --csv -
BILLED_USD_PER_M = {"Qwen/Qwen3-8B": {"sample": 0.60, "prefill": 0.195, "train": 0.44}}
FALLBACK_RATES = BILLED_USD_PER_M["Qwen/Qwen3-8B"]


def summarize_lengths(path: Path, max_tokens: int) -> None:
    """Per-step student answer length and cut-off rate (from the length log)."""
    if not path.exists():
        return
    from collections import defaultdict

    by = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        by[r["batch"]].append(r)
    print(f"\nstudent answers per step (max_tokens={max_tokens}):")
    for b in sorted(by):
        rs = by[b]
        lens = sorted(r["n_tokens"] for r in rs)
        print(f"  step {b}: n={len(rs)}  mean {sum(lens) / len(lens):.0f}  median {lens[len(lens) // 2]}  "
              f"cut off {sum(r['cut_off'] for r in rs) / len(rs):.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="JSONL from build_distill_data.py")
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
    ap.add_argument("--max-context-length", type=int, default=None, help="default: min(student, teacher) context on Tinker")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=20, help="teacher top-K tokens (0 = importance-sampling fallback)")
    ap.add_argument("--reverse", action="store_true", help="reverse KL over the top-K (mode-seeking)")
    ap.add_argument("--teacher-sync-every", type=int, default=None, help="copy student into teacher every N steps (self-distillation only)")
    ap.add_argument("--teacher-model", default=None, help="teacher base model (default: --model, i.e. self-distillation)")
    ap.add_argument("--teacher-checkpoint", default=None, help="tinker:// sampler path for the teacher (optional)")
    ap.add_argument("--teacher-prompt", default="auto", choices=["auto", "same", "hint"],
                    help="auto = hint for self-distillation, same for a different teacher")
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

    from posttrain.distill import TEACHER_PROMPTS, DistillDataset, check_same_tokenizer, teacher_redirect
    from posttrain.sft_data import read_jsonl

    teacher = args.teacher_model or args.model
    other_teacher = teacher != args.model or args.teacher_checkpoint is not None
    prompt_mode = args.teacher_prompt if args.teacher_prompt != "auto" else ("same" if other_teacher else "hint")
    TEACHER_TEMPLATE = TEACHER_PROMPTS[prompt_mode]
    if other_teacher and args.teacher_sync_every:
        raise SystemExit("--teacher-sync-every copies the student into the teacher: only for self-distillation")
    if teacher != args.model:
        check_same_tokenizer(args.model, teacher)
    if args.max_context_length is None:
        import tinker

        caps = {m.model_name: m.max_context_length for m in tinker.RestClient().get_server_capabilities().result().supported_models}
        ctx = [caps.get(args.model), caps.get(teacher)]
        args.max_context_length = min(c for c in ctx if c) if any(ctx) else 32768

    renderer_name = args.renderer or get_recommended_renderer_name(args.model)
    tok = get_tokenizer(args.model)
    renderer = renderers.get_renderer(renderer_name, tok, model_name=args.model)
    rows = [r for f in args.data for r in read_jsonl(f)]
    if not rows:
        raise SystemExit("no rows in --data")
    if prompt_mode == "hint" and not any(r["context"] for r in rows):
        raise SystemExit("--teacher-prompt hint but the data has no contexts (built with --context none?)")

    # Budget check: teacher prompt + student completion must fit the context, or completions get cut.
    t_lens = [build_sdft_teacher_prompt(r["question"], r["context"], renderer, demo_template=TEACHER_TEMPLATE).length for r in rows]
    s_lens = [renderer.build_generation_prompt([{"role": "user", "content": r["question"]}]).length for r in rows]
    room = [args.max_context_length - t for t in t_lens]
    too_tight = sum(x < args.max_tokens for x in room)
    steps = -(-len(rows) * args.epochs // args.batch_size)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    sampled_per_step = args.batch_size * args.group_size * min(args.max_tokens, 15_000)  # ~15K mean thinking answer
    s_rate = BILLED_USD_PER_M.get(args.model, FALLBACK_RATES)
    t_rate = BILLED_USD_PER_M.get(teacher, FALLBACK_RATES)
    # student samples; teacher scores the same tokens (prefill with top-K); training on the same tokens
    usd_step = sampled_per_step / 1e6 * (s_rate["sample"] + t_rate["prefill"] + s_rate["train"])
    summary = {
        "student": args.model, "teacher": teacher + (f" @ {args.teacher_checkpoint}" if args.teacher_checkpoint else ""),
        "teacher_prompt": prompt_mode, "max_context_length": args.max_context_length,
        "problems": len(rows), "context_types": dict(Counter(r["context_type"] for r in rows)),
        "steps": steps, "renderer": renderer_name, "lr": args.lr, "lora_rank": args.lora_rank,
        "student_prompt_tokens_max": max(s_lens), "teacher_prompt_tokens_max": max(t_lens),
        "problems_where_teacher_context_cuts_completion_below_max_tokens": too_tight,
        "approx_sampled_tokens_per_step": sampled_per_step,
        "approx_usd_per_step": round(usd_step, 2),
        "approx_usd_total": round(steps * usd_step, 2),
        "note": ("billed rates: " + ", ".join(f"{m}: {'measured' if m in BILLED_USD_PER_M else 'UNMEASURED (Qwen3-8B rates used)'}"
                                             for m in dict.fromkeys([args.model, teacher]))),
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
        model_name=args.model, recipe_name="vals_hacks_distill", renderer_name=renderer_name,
        lora_rank=args.lora_rank, learning_rate=args.lr, max_tokens=args.max_tokens, temperature=args.temperature,
        topk=args.topk, reverse=args.reverse, demo_template=TEACHER_TEMPLATE, teacher_sync_every=args.teacher_sync_every,
        max_context_length=args.max_context_length, evaluator_builders=evaluators,
        eval_every=args.eval_every if evaluators else 0, save_every=args.save_every, log_path=log_path,
        wandb_project=args.wandb_project, wandb_name=args.name if args.wandb_project else None, session_name=args.name,
        load_checkpoint_path=args.load_checkpoint, max_steps=args.max_steps,
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=args.if_exists)
    dataset = DistillDataset(rows, batch_size=args.batch_size, group_size=args.group_size, renderer=renderer,
                                 seed=args.seed, epochs=args.epochs,
                                 length_log_path=str(Path(log_path) / "student_lengths.jsonl"))
    (Path(log_path)).mkdir(parents=True, exist_ok=True)
    (Path(log_path) / "distill_config.json").write_text(json.dumps({**vars(args), "teacher": teacher,
                                                                    "teacher_prompt_mode": prompt_mode}, indent=2))
    with teacher_redirect(args.model, teacher, args.teacher_checkpoint):
        asyncio.run(sdft.main(cfg, dataset))
    summarize_lengths(Path(log_path) / "student_lengths.jsonl", args.max_tokens)
    ck = Path(log_path) / "checkpoints.jsonl"
    if ck.exists():
        last = [json.loads(line) for line in open(ck)][-1]
        print(f"\nfinal checkpoint: {json.dumps(last)}")
        if last.get("sampler_path"):
            print(f"evaluate: uv run python scripts/eval_agentic.py --model {args.model} --model-path {last['sampler_path']} "
                  f"--limit 50 --name agent-{args.name}")


if __name__ == "__main__":
    main()

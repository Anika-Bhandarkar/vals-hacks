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

    # the cautious 27B -> 9B retry (after opd-27b-9b got worse): low lr with warmup + cosine decay,
    # reverse KL, T=0.7, 16K answers, cut-off answers dropped, early stop on a held-out pass@1 drop
    uv run python scripts/train_distill.py --name opd-27b-9b-v2 --data data/distill/plain.jsonl \
        --model Qwen/Qwen3.5-9B --teacher-model Qwen/Qwen3.8-27B --lora-rank 32 \
        --lr 1e-4 --warmup-steps 2 --lr-schedule cosine --reverse --temperature 0.7 --max-tokens 16384 \
        --cutoff drop --batch-size 8 --group-size 4 --max-steps 12 --save-every 4 \
        --eval-single 30 --eval-every 4 --eval-final --early-stop-drop 0.10 \
        --length-prior runs/train/opd-27b-9b/student_lengths.jsonl --dry-run
    # same flags on Modal (survives the laptop closing): scripts/train_distill_modal.py

Output: runs/train/<name>/ (metrics.jsonl, checkpoints.jsonl). Evaluate the final sampler checkpoint
with scripts/eval_agentic.py / eval_lcb.py as usual.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

RUNS_DIR = Path(os.environ.get("VALS_RUNS_DIR", ROOT / "runs" / "train"))

# Billed Tinker rates, USD per million tokens (tinker billing usage, Nov 2026; equal to the published list).
# The opd-27b-9b session: 9B sample 0.807, 9B train 1.463 (billed on prompt + completion), 27B teacher
# prefill ~0.68 (topk prompt logprobs bill as prefill only). Re-check with:
#   uv run tinker billing usage <start>Z <end>Z --csv -
BILLED_USD_PER_M = {
    "Qwen/Qwen3-8B": {"sample": 0.60, "prefill": 0.195, "train": 0.44},
    "Qwen/Qwen3.5-9B": {"sample": 0.807, "prefill": 0.30, "train": 1.463},
    "Qwen/Qwen3.8-27B": {"sample": 5.595, "prefill": 0.6975, "train": 4.103},
}
FALLBACK_RATES = BILLED_USD_PER_M["Qwen/Qwen3-8B"]


def summarize_lengths(path: Path, max_tokens: int) -> None:
    """Per-step student answer length and cut-off rate (from the length log)."""
    if not path.exists():
        return
    from collections import defaultdict

    by, forced = defaultdict(list), defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        (forced if r.get("kind") == "forced_answer" else by)[r["batch"]].append(r)
    print(f"\nstudent answers per step (max_tokens={max_tokens}):")
    for b in sorted(by):
        rs = by[b]
        lens = sorted(r["n_tokens"] for r in rs)
        f = forced.get(b, [])
        extra = (f"  forced stop {len(f)} (answer mean {sum(r['n_tokens'] for r in f) / len(f):.0f}, "
                 f"still cut off {sum(r['cut_off'] for r in f)})") if f else ""
        print(f"  step {b}: n={len(rs)}  mean {sum(lens) / len(lens):.0f}  median {lens[len(lens) // 2]}  "
              f"cut off {sum(r['cut_off'] for r in rs) / len(rs):.0%}{extra}")


def summarize_evals(log_path: Path, metric: str) -> None:
    """Held-out evals (evals.jsonl) next to the checkpoint saved at that step, if any."""
    p = log_path / "evals.jsonl"
    if not p.exists():
        return
    ck = log_path / "checkpoints.jsonl"
    paths = {}
    for c in ([json.loads(line) for line in open(ck)] if ck.exists() else []):
        if c.get("sampler_path"):
            paths[c.get("batch")] = c["sampler_path"]
    print(f"\nheld-out evals ({metric}):")
    for r in (json.loads(line) for line in open(p)):
        vals = "  ".join(f"{k}={v:.3f}" for k, v in r.items() if k not in ("step", "final") and isinstance(v, float))
        print(f"  step {r['step']}{' (final)' if r.get('final') else ''}: {vals}  {paths.get(r['step'], '')}")
    es = log_path / "early_stop.json"
    if es.exists():
        print(f"EARLY STOPPED: {es.read_text()}")


def length_prior(path: str, max_tokens: int, max_batch: int = 3) -> dict:
    """Answer lengths from an earlier run's student_lengths.jsonl (its first `max_batch` steps, i.e. a
    near-base student), re-capped at max_tokens: mean sampled tokens, fraction that finishes, mean
    finished length."""
    rs = [r for r in (json.loads(line) for line in open(path)) if r["batch"] < max_batch and r.get("kind") != "forced_answer"]
    fin = [r for r in rs if not r["cut_off"] and r["n_tokens"] < max_tokens]
    return {"n": len(rs), "mean_sampled": sum(min(r["n_tokens"], max_tokens) for r in rs) / len(rs),
            "finish_frac": len(fin) / len(rs), "mean_finished": sum(r["n_tokens"] for r in fin) / max(1, len(fin))}


def resolve_data(f: str) -> str:
    """--data paths are relative to the repo root when they don't exist relative to the cwd (Modal)."""
    return f if Path(f).exists() or Path(f).is_absolute() else str(ROOT / f)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="JSONL from build_distill_data.py")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--renderer", default=None)
    ap.add_argument("--load-checkpoint", default=None)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4, help="cookbook: 5e-4..1e-3 for LoRA, lower end recommended "
                    "(opd-27b-9b at 5e-4 got worse; 1e-4 for the retry)")
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "linear", "cosine"],
                    help="decay after warmup, down to --min-lr-frac * lr at the last step")
    ap.add_argument("--warmup-steps", type=int, default=0, help="linear warmup from lr/N to lr")
    ap.add_argument("--min-lr-frac", type=float, default=0.1, help="final lr as a fraction of --lr (linear/cosine)")
    ap.add_argument("--batch-size", type=int, default=16, help="problems per step")
    ap.add_argument("--group-size", type=int, default=4, help="student samples per problem")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=24000, help="student generation budget")
    ap.add_argument("--max-context-length", type=int, default=None, help="default: min(student, teacher) context on Tinker")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=20, help="teacher top-K tokens (0 = importance-sampling fallback)")
    ap.add_argument("--reverse", action="store_true",
                    help="reverse KL over the teacher's top-K (mode-seeking; cookbook sdft custom loss, REINFORCE "
                         "form, skips the first 3 completion tokens). Default: forward KL = top-K cross-entropy "
                         "toward the teacher (mode-covering). Reverse runs a client-side forward + "
                         "forward_backward per step, so training is probably billed ~2x")
    ap.add_argument("--cutoff", default="keep", choices=["keep", "drop", "force"],
                    help="student answers that hit --max-tokens: keep (train on the truncated prefix), drop (not "
                         "scored or trained: costs sampling only), force (append </think> and sample the solution, "
                         "then train; still-unfinished ones dropped)")
    ap.add_argument("--teacher-sync-every", type=int, default=None, help="copy student into teacher every N steps (self-distillation only)")
    ap.add_argument("--teacher-model", default=None, help="teacher base model (default: --model, i.e. self-distillation)")
    ap.add_argument("--teacher-checkpoint", default=None, help="tinker:// sampler path for the teacher (optional)")
    ap.add_argument("--teacher-prompt", default="auto", choices=["auto", "same", "hint"],
                    help="auto = hint for self-distillation, same for a different teacher")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-single", type=int, default=0, help="N held-out LCB problems, single attempt")
    ap.add_argument("--eval-agentic", type=int, default=0, help="N held-out LCB tasks, competition format")
    ap.add_argument("--eval-final", action="store_true", help="also run the evals on the final checkpoint")
    ap.add_argument("--early-stop-drop", type=float, default=None,
                    help="stop the run when --early-stop-metric falls more than this below step 0 (e.g. 0.10)")
    ap.add_argument("--early-stop-metric", default="single/pass@1")
    ap.add_argument("--length-prior", default=None,
                    help="dry-run cost estimate from an earlier run's student_lengths.jsonl (first 3 steps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--if-exists", default="raise", choices=["raise", "resume", "delete", "ask"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    args.data = [resolve_data(f) for f in args.data]
    if args.early_stop_drop is not None and not (args.eval_single or args.eval_agentic):
        raise SystemExit("--early-stop-drop needs an eval (--eval-single N or --eval-agentic N)")

    from collections import Counter

    from tinker_cookbook import renderers
    from tinker_cookbook.distillation.sdft import build_sdft_teacher_prompt
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from posttrain.distill import TEACHER_PROMPTS, DistillDataset, check_same_tokenizer, lr_at, teacher_redirect
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
    s_rate = BILLED_USD_PER_M.get(args.model, FALLBACK_RATES)
    t_rate = BILLED_USD_PER_M.get(teacher, FALLBACK_RATES)
    n_ans = args.batch_size * args.group_size
    if args.length_prior and not Path(args.length_prior).exists():  # e.g. inside the Modal container
        print(f"--length-prior {args.length_prior} not found here; cost estimate uses the heuristic")
        args.length_prior = None
    if args.length_prior:
        prior = length_prior(args.length_prior, args.max_tokens)
    else:  # ~15K mean thinking answer; guess half finish when not keeping cut-offs
        prior = {"n": 0, "mean_sampled": min(args.max_tokens, 15_000), "finish_frac": 0.5,
                 "mean_finished": min(args.max_tokens, 15_000) * 0.6}
    sampled = n_ans * prior["mean_sampled"]
    if args.cutoff == "keep":
        trained_ans, trained_tokens = n_ans, sampled
    elif args.cutoff == "drop":
        trained_ans = n_ans * prior["finish_frac"]
        trained_tokens = trained_ans * prior["mean_finished"]
    else:  # force: every unfinished answer gets one more call (~3K answer, a guess), most then finish
        n_forced = n_ans * (1 - prior["finish_frac"])
        sampled += n_forced * 3000
        trained_ans = n_ans * prior["finish_frac"] + 0.8 * n_forced
        trained_tokens = n_ans * prior["finish_frac"] * prior["mean_finished"] + 0.8 * n_forced * (args.max_tokens + 3000)
    mean_t_prompt = sum(t_lens) / len(t_lens)
    seq_tokens = trained_tokens + trained_ans * mean_t_prompt  # teacher prefill and training bill prompt + completion
    train_mult = 2.0 if args.reverse else 1.0
    usd_step = (sampled * s_rate["sample"] + seq_tokens * t_rate["prefill"] + seq_tokens * s_rate["train"] * train_mult) / 1e6
    # each eval: N single-attempt answers at the evaluator's 28K cap (base 9B: ~77% hit it), 9B sampling
    n_evals = (math.ceil(steps / args.eval_every) if args.eval_every > 0 else 0) + int(args.eval_final)
    usd_eval = (args.eval_single * 25_000 * s_rate["sample"] + args.eval_agentic * 60_000 * s_rate["sample"]) / 1e6
    usd_total = steps * usd_step + n_evals * usd_eval
    summary = {
        "student": args.model, "teacher": teacher + (f" @ {args.teacher_checkpoint}" if args.teacher_checkpoint else ""),
        "teacher_prompt": prompt_mode, "max_context_length": args.max_context_length,
        "problems": len(rows), "context_types": dict(Counter(r["context_type"] for r in rows)),
        "steps": steps, "renderer": renderer_name, "lr": args.lr, "lora_rank": args.lora_rank,
        "student_prompt_tokens_max": max(s_lens), "teacher_prompt_tokens_max": max(t_lens),
        "problems_where_teacher_context_cuts_completion_below_max_tokens": too_tight,
        "lr_schedule": [round(lr_at(i, steps, args.lr, args.lr_schedule, args.warmup_steps, args.min_lr_frac), 7)
                        for i in range(steps)],
        "loss": "reverse KL (top-K)" if args.reverse else "forward KL (top-K CE)",
        "temperature": args.temperature, "max_tokens": args.max_tokens, "cutoff": args.cutoff,
        "early_stop": (f"{args.early_stop_metric} more than {args.early_stop_drop} below step 0"
                       if args.early_stop_drop is not None else None),
        "length_prior": {k: round(v, 3) for k, v in prior.items()} | {"from": args.length_prior or "heuristic"},
        "approx_sampled_tokens_per_step": round(sampled),
        "approx_trained_answers_per_step": round(trained_ans, 1),
        "approx_trained_tokens_per_step": round(seq_tokens),
        "approx_usd_per_step": round(usd_step, 2),
        "approx_usd_per_eval": round(usd_eval, 2), "evals": n_evals,
        "approx_usd_total": round(usd_total, 2),
        "note": ("billed rates: " + ", ".join(f"{m}: {'measured' if m in BILLED_USD_PER_M else 'UNMEASURED (Qwen3-8B rates used)'}"
                                             for m in dict.fromkeys([args.model, teacher]))
                 + ("; reverse KL training counted at 2x (client-side forward + forward_backward)" if args.reverse else "")),
        "log_path": str(RUNS_DIR / args.name),
    }
    print(json.dumps(summary, indent=2))
    i = max(range(len(rows)), key=lambda k: t_lens[k])
    print("\n--- example teacher user message (longest) ---\n" + TEACHER_TEMPLATE.format(question=rows[i]["question"], golden_answer=rows[i]["context"])[-1800:])
    if args.dry_run:
        return

    from tinker_cookbook import cli_utils
    from tinker_cookbook.distillation import sdft

    from tinker_cookbook import checkpoint_utils

    from posttrain.distill import EarlyStop, GuardedEvaluator, TrainState, training_controls
    from posttrain.evaluators import AgenticEvaluator, SingleAttemptEvaluator

    log_path = str(RUNS_DIR / args.name)
    cli_utils.check_log_dir(log_path, behavior_if_exists=args.if_exists)
    if (Path(log_path) / "early_stop.json").exists():
        print(f"{log_path} was early-stopped; not resuming (--if-exists delete to start over):")
        summarize_evals(Path(log_path), args.early_stop_metric)
        return
    last = checkpoint_utils.get_last_checkpoint(log_path)
    state = TrainState(step=last.batch if last else 0)
    # the cookbook calls each builder to get an evaluator; GuardedEvaluator logs evals.jsonl and early-stops
    inner = []
    if args.eval_single:
        inner.append(lambda: SingleAttemptEvaluator(model_name=args.model, renderer_name=renderer_name, limit=args.eval_single))
    if args.eval_agentic:
        inner.append(lambda: AgenticEvaluator(model_name=args.model, renderer_name=renderer_name, limit=args.eval_agentic))
    evaluators = [lambda b=b: GuardedEvaluator(b(), state, log_path, args.early_stop_metric, args.early_stop_drop)
                  for b in inner]
    cfg = sdft.Config(
        model_name=args.model, recipe_name="vals_hacks_distill", renderer_name=renderer_name,
        lora_rank=args.lora_rank, learning_rate=args.lr, max_tokens=args.max_tokens, temperature=args.temperature,
        topk=args.topk, reverse=args.reverse, demo_template=TEACHER_TEMPLATE, teacher_sync_every=args.teacher_sync_every,
        max_context_length=args.max_context_length, evaluator_builders=evaluators,
        eval_every=args.eval_every if evaluators else 0, save_every=args.save_every, log_path=log_path,
        wandb_project=args.wandb_project, wandb_name=args.name if args.wandb_project else None, session_name=args.name,
        load_checkpoint_path=args.load_checkpoint, max_steps=args.max_steps,
    )
    dataset = DistillDataset(rows, batch_size=args.batch_size, group_size=args.group_size, renderer=renderer,
                             seed=args.seed, epochs=args.epochs, cutoff=args.cutoff,
                             length_log_path=str(Path(log_path) / "student_lengths.jsonl"))
    (Path(log_path)).mkdir(parents=True, exist_ok=True)
    (Path(log_path) / "distill_config.json").write_text(json.dumps({**vars(args), "teacher": teacher,
                                                                    "teacher_prompt_mode": prompt_mode}, indent=2))
    total = min(len(dataset), args.max_steps or len(dataset))
    lr_fn = lambda step: lr_at(step, total, args.lr, args.lr_schedule, args.warmup_steps, args.min_lr_frac)  # noqa: E731
    stopped = None
    try:
        with teacher_redirect(args.model, teacher, args.teacher_checkpoint), \
                training_controls(state, lr_fn, args.cutoff, renderer.get_stop_sequences()):
            asyncio.run(sdft.main(cfg, dataset))
    except EarlyStop as e:  # a clean exit (no Modal retry); early_stop.json keeps a resume from continuing
        stopped = e
        print(f"\nEARLY STOP: {e}")
    if args.eval_final and stopped is None:
        final_eval(Path(log_path), state, [b() for b in evaluators], total)
    summarize_lengths(Path(log_path) / "student_lengths.jsonl", args.max_tokens)
    summarize_evals(Path(log_path), args.early_stop_metric)
    ck = Path(log_path) / "checkpoints.jsonl"
    if ck.exists():
        last = [json.loads(line) for line in open(ck)][-1]
        print(f"\nfinal checkpoint: {json.dumps(last)}")
        if last.get("sampler_path"):
            print(f"evaluate: uv run python scripts/eval_agentic.py --model {args.model} --model-path {last['sampler_path']} "
                  f"--limit 50 --name agent-{args.name}")


def final_eval(log_path: Path, state, evaluators: list, total: int) -> None:
    """Run the in-training evals once more on the final sampler checkpoint (skipped if already logged)."""
    import tinker

    from posttrain.distill import EarlyStop

    evals = log_path / "evals.jsonl"
    if evals.exists() and any(json.loads(line).get("final") for line in open(evals)):
        return
    ck = [json.loads(line) for line in open(log_path / "checkpoints.jsonl")]
    final = next((c for c in reversed(ck) if c.get("sampler_path")), None)
    if final is None:
        print("no sampler checkpoint for the final eval")
        return
    state.step = total

    async def run():
        client = tinker.ServiceClient().create_sampling_client(model_path=final["sampler_path"])
        for e in evaluators:
            try:
                print(f"final eval ({final['sampler_path']}): {await e(client, final=True)}")
            except EarlyStop as err:
                print(f"final eval: {err}")

    asyncio.run(run())


if __name__ == "__main__":
    main()

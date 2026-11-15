"""SFT on Tinker (LoRA) from our JSONL data, with optional in-training evals on held-out problems.

    # check data + config only (no Tinker calls)
    uv run python scripts/train_sft.py --name sft-rft-v1 --data data/sft/mix.jsonl --dry-run

    # train
    uv run python scripts/train_sft.py --name sft-rft-v1 --data data/sft/rft_agentic.jsonl data/sft/r1_cots.jsonl:3000 \
        --epochs 1 --save-every 25 --eval-agentic 20 --eval-every 50

    # continue from a checkpoint (e.g. RL -> SFT -> RL), keeping the optimizer state
    uv run python scripts/train_sft.py --name sft-v2 --data ... --load-checkpoint tinker://<run>/weights/<name>

Output: runs/train/<name>/ (cookbook metrics.jsonl, checkpoints.jsonl with state + sampler paths).
Then evaluate a checkpoint with the full eval scripts:
    uv run python scripts/eval_agentic.py --model Qwen/Qwen3-8B --model-path <sampler path> --limit 50 --name agent-<name>
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


def build_config(args):
    from tinker_cookbook.hyperparam_utils import get_lr
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    from posttrain.evaluators import AgenticEvaluator, SingleAttemptEvaluator
    from posttrain.sft_dataset import SFTJsonlBuilder

    renderer_name = args.renderer or get_recommended_renderer_name(args.model)
    files, caps = [], []
    for spec in args.data:
        path, _, cap = spec.partition(":")
        if not Path(path).exists():
            raise SystemExit(f"missing data file {path}")
        files.append(str(Path(path).resolve()))
        caps.append(int(cap or 0))
    lr = args.lr if args.lr is not None else get_lr(args.model)
    dataset = SFTJsonlBuilder(
        files=tuple(files), caps=tuple(caps), test_size=args.test_size, shuffle_seed=args.seed,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=args.model, renderer_name=renderer_name,
            max_length=args.max_length, batch_size=args.batch_size,
        ),
    )
    evaluators, infrequent = [], []
    if args.eval_single:
        evaluators.append(lambda: SingleAttemptEvaluator(model_name=args.model, renderer_name=renderer_name, limit=args.eval_single))
    if args.eval_agentic:
        infrequent.append(lambda: AgenticEvaluator(model_name=args.model, renderer_name=renderer_name, limit=args.eval_agentic,
                                                   context_window=args.max_length))
    log_path = ROOT / "runs" / "train" / args.name
    config = train.Config(
        log_path=str(log_path), model_name=args.model, recipe_name="vals_hacks_sft", renderer_name=renderer_name,
        load_checkpoint_path=args.load_checkpoint, dataset_builder=dataset, learning_rate=lr,
        lr_schedule=args.lr_schedule, num_epochs=args.epochs, max_steps=args.max_steps, lora_rank=args.lora_rank,
        save_every=args.save_every, eval_every=args.eval_every if args.eval_single else 0,
        infrequent_eval_every=args.eval_every if args.eval_agentic else 0,
        evaluator_builders=evaluators, infrequent_evaluator_builders=infrequent,
        ttl_seconds=args.ttl_days * 86400 if args.ttl_days else None,
        wandb_project=args.wandb_project, wandb_name=args.name if args.wandb_project else None, session_name=args.name,
    )
    return config, files, caps, renderer_name, lr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="JSONL path, or path:max_examples")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--renderer", default=None)
    ap.add_argument("--load-checkpoint", default=None, help="tinker://.../weights/<name> to start from")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default: cookbook get_lr(model)")
    ap.add_argument("--lr-schedule", default="linear", choices=["linear", "cosine", "constant"])
    ap.add_argument("--batch-size", type=int, default=32, help="conversations per step (multi-turn ones expand to several examples)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=32768, help="drop examples longer than this (tokens)")
    ap.add_argument("--test-size", type=int, default=0, help="held-out conversations for NLL eval")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--ttl-days", type=int, default=14, help="periodic checkpoint TTL (final is kept); 0 = keep all")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-single", type=int, default=0, help="N held-out LCB problems, single attempt (0 = off)")
    ap.add_argument("--eval-agentic", type=int, default=0, help="N held-out LCB tasks, competition format (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--if-exists", default="raise", choices=["raise", "resume", "delete", "ask"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config, files, caps, renderer_name, lr = build_config(args)
    from posttrain.sft_dataset import count_examples

    st = count_examples(files, args.model, renderer_name, args.max_length)
    steps = -(-st["conversations"] // args.batch_size) * args.epochs
    print(json.dumps({"data": dict(zip(files, caps)), "renderer": renderer_name, "lr": lr, "lora_rank": args.lora_rank,
                      "batch_size": args.batch_size, "approx_steps": steps if not args.max_steps else min(steps, args.max_steps),
                      "trained_tokens_per_epoch": st["trained_tokens"], **st, "log_path": config.log_path}, indent=2))
    if args.dry_run:
        return
    from tinker_cookbook import cli_utils
    from tinker_cookbook.supervised import train

    cli_utils.check_log_dir(config.log_path, behavior_if_exists=args.if_exists)
    asyncio.run(train.main(config))
    ck = Path(config.log_path) / "checkpoints.jsonl"
    if ck.exists():
        last = [json.loads(line) for line in open(ck)][-1]
        print(f"\nfinal checkpoint: {json.dumps(last)}")
        sp = last.get("sampler_path")
        if sp:
            print(f"evaluate: uv run python scripts/eval_agentic.py --model {args.model} --model-path {sp} --limit 50 --name agent-{args.name}")


if __name__ == "__main__":
    main()

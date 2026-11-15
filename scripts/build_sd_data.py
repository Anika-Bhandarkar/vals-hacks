"""Build self-distillation data: per training problem, the student prompt + a privileged context.

    # editorial hints (Codeforces problems that have one)
    uv run python scripts/build_sd_data.py --context editorial --out data/sd/editorial.jsonl

    # the model's own failed attempts + verdicts (from `build_sft_data.py rft-single`'s .failures.jsonl)
    uv run python scripts/build_sd_data.py --context feedback --failures data/sft/rft_single.failures.jsonl --out data/sd/feedback.jsonl

    # correct solutions as the hint (from rft-single / traces outputs)
    uv run python scripts/build_sd_data.py --context reference --solutions data/sft/rft_single.jsonl --out data/sd/reference.jsonl

    # one row per problem, best available context in priority order
    uv run python scripts/build_sd_data.py --context editorial,feedback,tags --failures ... --out data/sd/mixed.jsonl

    # focus on problems the model doesn't already solve reliably
    ... --pass-rates data/sft/rft_single.pass_rates.json --max-pass-rate 0.75

The student prompt is exactly the single-attempt prompt used for SFT (posttrain.problems.single_prompt).
Training problems only (Codeforces train split, decontaminated against every eval).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cppgrade.grader import extract_cpp  # noqa: E402
from posttrain.problems import load_pool, single_prompt  # noqa: E402
from posttrain.self_distill import context_editorial, context_feedback, context_reference, context_tags  # noqa: E402
from posttrain.sft_data import read_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", default="editorial",
                    help="comma list, priority order: editorial, feedback, reference, tags")
    ap.add_argument("--failures", nargs="*", default=[], help=".failures.jsonl from rft-single (for feedback)")
    ap.add_argument("--solutions", nargs="*", default=[], help="SFT JSONL with correct answers (for reference)")
    ap.add_argument("--pass-rates", default=None, help="rft-single .pass_rates.json, for --min/--max-pass-rate")
    ap.add_argument("--min-pass-rate", type=float, default=0.0)
    ap.add_argument("--max-pass-rate", type=float, default=1.0)
    ap.add_argument("--min-rating", type=int, default=0)
    ap.add_argument("--max-rating", type=int, default=10_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kinds = [k.strip() for k in args.context.split(",") if k.strip()]
    bad = set(kinds) - {"editorial", "feedback", "reference", "tags"}
    if bad:
        raise SystemExit(f"unknown context types {bad}")

    fails: dict[str, list[dict]] = defaultdict(list)
    for f in args.failures:
        for r in read_jsonl(f):
            fails[r["problem_id"]].append(r)
    refs: dict[str, list[str]] = defaultdict(list)
    for f in args.solutions:
        for r in read_jsonl(f):
            code = extract_cpp(r["messages"][-1]["content"])
            if code:
                refs[r["meta"]["problem_id"]].append(code)
    rates = json.loads(Path(args.pass_rates).read_text()) if args.pass_rates else {}

    pool = load_pool(min_rating=args.min_rating, max_rating=args.max_rating, seed=args.seed)
    rows, why = [], Counter()
    for p in pool:
        if rates and p["id"] in rates and not (args.min_pass_rate <= rates[p["id"]] <= args.max_pass_rate):
            why["pass_rate_filtered"] += 1
            continue
        ctx, kind = None, None
        for k in kinds:
            if k == "editorial":
                ctx = context_editorial(p)
            elif k == "tags":
                ctx = context_tags(p)
            elif k == "feedback" and fails.get(p["id"]):
                # most informative failure: the one that got furthest (fewest remaining failing tests)
                ctx = context_feedback(max(fails[p["id"]], key=lambda x: x.get("passed", 0)))
            elif k == "reference" and refs.get(p["id"]):
                ctx = context_reference(min(refs[p["id"]], key=len))
            if ctx:
                kind = k
                break
        if not ctx:
            why["no_context"] += 1
            continue
        why[kind] += 1
        rows.append({"problem_id": p["id"], "question": single_prompt(p), "context": ctx, "context_type": kind,
                     "rating": p.get("rating"), "pass_rate": rates.get(p["id"])})
        if args.limit and len(rows) >= args.limit:
            break
    n = write_jsonl(args.out, rows)
    print(f"wrote {n} problems -> {args.out}   {dict(why.most_common())}")


if __name__ == "__main__":
    main()

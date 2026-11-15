"""Test A: does a hint make the model solve problems it otherwise can't? (diagnostic for
self-distillation from hints, and for hinted RL rollouts)

Same Codeforces problems, single attempt, three conditions:
  none       just the statement
  tags       + the problem's Codeforces tags
  editorial  + the official editorial (code stripped: a hint, not an answer)

    # once: uv run python scripts/prepare_codeforces.py
    uv run python scripts/hint_uplift.py --limit 2 --samples 1 --name hint-smoke     # check it works
    uv run python scripts/hint_uplift.py --limit 50 --samples 2 --name hint-uplift   # the test (~300 answers)
    uv run python scripts/hint_uplift.py --summarize runs/evals/hint-uplift

Re-running resumes; --limit/--samples can grow under the same name.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

CONDITIONS = ("none", "tags", "editorial")
MAY_CHANGE_ON_RESUME = {"limit", "samples", "seed", "concurrency"}
INSTRUCTION = ("Write a complete C++20 program that reads the input from standard input and writes the answer "
               "to standard output. Do not print anything else. Put your final solution in a single ```cpp code block.")


def build_prompt(p: dict, condition: str) -> str:
    from cppgrade.codeforces import hint_text, statement

    hint = hint_text(p, condition)
    return "Solve the following competitive programming problem.\n\n" + statement(p) + ("\n\n" + hint if hint else "") + "\n\n" + INSTRUCTION


def select(index: list[dict], limit: int | None, seed: int) -> list[dict]:
    probs = sorted(index, key=lambda p: p["id"])
    random.Random(seed).shuffle(probs)  # stable, nested: growing --limit only adds problems
    return probs[:limit] if limit else probs


def summarize(out_dir: Path) -> dict:
    rows = [json.loads(line) for line in open(out_dir / "results.jsonl")]
    rate = defaultdict(list)  # (cond, pid) -> [accepted]
    meta = {}
    for r in rows:
        rate[(r["condition"], r["id"])].append(r["verdict"] == "accepted")
        meta[r["id"]] = r["rating"]
    pids = sorted({pid for (_, pid) in rate if all((c, pid) in rate for c in CONDITIONS)})
    s: dict = {"problems": len(pids), "pass@1": {}, "by_rating": {}, "verdicts": {}, "transitions": {}}
    pr = lambda c, pid: sum(rate[(c, pid)]) / len(rate[(c, pid)])
    for c in CONDITIONS:
        s["pass@1"][c] = sum(pr(c, p) for p in pids) / max(1, len(pids))
        s["verdicts"][c] = dict(Counter(r["verdict"] for r in rows if r["condition"] == c and r["id"] in pids).most_common())
        buckets = defaultdict(list)
        for p in pids:
            buckets[(meta[p] // 400) * 400].append(pr(c, p))
        s["by_rating"][c] = {f"{b}-{b + 399}": (sum(v) / len(v), len(v)) for b, v in sorted(buckets.items())}
    for c in ("tags", "editorial"):
        solved = lambda cond, p: pr(cond, p) >= 0.5
        s["transitions"][c] = {
            "newly_solved": sum(not solved("none", p) and solved(c, p) for p in pids),
            "lost": sum(solved("none", p) and not solved(c, p) for p in pids),
            "unsolved_both": sum(not solved("none", p) and not solved(c, p) for p in pids),
        }
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def print_summary(s: dict) -> None:
    print(f"\n== {s['problems']} problems (rating 1600-2400), pass@1 by condition")
    base = s["pass@1"]["none"]
    for c in CONDITIONS:
        d = s["pass@1"][c]
        print(f"   {c:10} {d:6.1%}  ({d - base:+.1%} vs none)   by rating: "
              + "  ".join(f"{k}={v[0]:.0%} (n={v[1]})" for k, v in s["by_rating"][c].items()))
    for c, t in s["transitions"].items():
        print(f"   {c:10} vs none (solved = pass rate >= 0.5): newly solved {t['newly_solved']}, lost {t['lost']}, unsolved in both {t['unsolved_both']}")
    for c in CONDITIONS:
        print(f"   verdicts [{c}]: {s['verdicts'][c]}")


async def run(args) -> Path | None:
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from cppgrade.codeforces import load_index
    from cppgrade.grader import grade, make_ondemand_pool

    out_dir = ROOT / "runs" / "evals" / args.name
    results_path = out_dir / "results.jsonl"
    cfg = {k: v for k, v in vars(args).items() if k not in ("summarize", "dry_run")}
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v and k not in MAY_CHANGE_ON_RESUME}
        if diff:
            raise SystemExit(f"{cfg_path} exists with different settings {diff}. Use a new --name.")

    probs = select(load_index(), args.limit, args.seed)
    done = Counter()
    if results_path.exists():
        for line in open(results_path):
            r = json.loads(line)
            done[(r["id"], r["condition"])] += 1
    todo = [(p, c) for p in probs for c in CONDITIONS if done[(p["id"], c)] < args.samples]
    n_new = sum(args.samples - done[(p["id"], c)] for p, c in todo)
    print(f"{len(probs)} problems x {len(CONDITIONS)} conditions x {args.samples} samples: {n_new} answers to sample")
    if args.dry_run:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    renderer = renderers.get_renderer(args.renderer or get_recommended_renderer_name(args.model),
                                      get_tokenizer(args.model), model_name=args.model)
    sc = tinker.ServiceClient()
    client = await (sc.create_sampling_client(model_path=args.model_path) if args.model_path
                    else sc.create_sampling_client(base_model=args.model))
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                                   stop=renderer.get_stop_sequences())
    pool = make_ondemand_pool(32)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = Counter()
    t0 = time.time()

    async def job(p: dict, cond: str):
        need = args.samples - done[(p["id"], cond)]
        async with sem:
            try:
                prompt = renderer.build_generation_prompt([{"role": "user", "content": build_prompt(p, cond)}])
                resp = await client.sample(prompt, num_samples=need, sampling_params=params)
                parsed = []
                for seq in resp.sequences:
                    msg, term = renderer.parse_response(seq.tokens)
                    parsed.append((renderers.get_text_content(msg), term.is_clean and seq.stop_reason != "length", len(seq.tokens)))
                grades = await asyncio.gather(*(
                    grade(pool, text, tests_file=p["tests_file"], n_tests=p["n_tests"], time_limit=float(p["time_limit"]),
                          memory_mb=max(512, int(p["memory_limit"])), cpp_std="c++20")
                    for text, _, _ in parsed))
            except Exception as e:
                stats["job_errors"] += 1
                print(f"  ! {p['id']}/{cond}: {type(e).__name__}: {str(e)[:200]} (will retry on re-run)")
                return
        lines = []
        for i, ((text, clean, ntok), g) in enumerate(zip(parsed, grades)):
            if g.verdict == "infra_error":
                stats["infra_errors"] += 1
                continue
            verdict = "truncated" if (g.verdict == "format_error" and not clean) else g.verdict
            lines.append(json.dumps({"id": p["id"], "rating": p["rating"], "condition": cond, "sample": done[(p["id"], cond)] + i,
                                     "verdict": verdict, "subtype": g.subtype, "passed": g.passed, "total": g.total,
                                     "n_tokens": ntok, "clean_stop": clean, "code": g.code, "response": text}))
            stats[f"{cond}:{verdict == 'accepted'}"] += 1
        async with lock:
            with open(results_path, "a") as f:
                f.write("".join(line + "\n" for line in lines))
            stats["jobs"] += 1
            if stats["jobs"] % max(1, len(todo) // 10) == 0 or stats["jobs"] == len(todo):
                acc = {c: stats[f"{c}:True"] for c in CONDITIONS}
                print(f"  [{stats['jobs']}/{len(todo)}] {time.time() - t0:.0f}s  accepted so far: {acc}")

    await asyncio.gather(*(job(p, c) for p, c in todo))
    if stats["infra_errors"] or stats["job_errors"]:
        print(f"{stats['infra_errors']} infra errors, {stats['job_errors']} failed jobs: re-run to retry them.")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--renderer", default=None)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=28000)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        print_summary(summarize(args.summarize))
        return
    if not args.name:
        ap.error("--name is required")
    out = asyncio.run(run(args))
    if out is not None:
        print_summary(summarize(out))


if __name__ == "__main__":
    main()

"""LiveCodeBench-CPP eval: sample on Tinker, grade on Modal, in C++ and/or Python on the SAME problems.

    # smoke test: 5 problems, both languages
    uv run python scripts/eval_lcb.py --model Qwen/Qwen3-8B --limit 5 --name smoke

    # full AtCoder set, 4 samples each
    uv run python scripts/eval_lcb.py --model Qwen/Qwen3-8B --samples 4 --name qwen3-8b-base

    # a trained checkpoint (sampler weights path from training)
    uv run python scripts/eval_lcb.py --model Qwen/Qwen3-8B \
        --model-path tinker://<run>/sampler_weights/step-100 --name step100

    # re-print a summary
    uv run python scripts/eval_lcb.py --summarize runs/evals/qwen3-8b-base

Results go to runs/evals/<name>/: config.json, results.jsonl (one line per sample, with the
model's answer, extracted code and verdict), summary.json. Re-running the same command resumes.

Prereq (once): uv run python scripts/prepare_lcb.py
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

# Settings that only pick WHICH problems / HOW MANY samples, so a run can grow (e.g. --limit 50 ->
# full set, --samples 1 -> 4) under the same --name. Anything that changes how a single sample is
# produced or graded (model, max tokens, temperature, time limit, ...) is locked per --name.
MAY_CHANGE_ON_RESUME = {"limit", "seed", "difficulty", "after", "samples", "concurrency", "pool"}

LANG_NAME = {"cpp": "C++17", "python": "Python 3"}
FENCE = {"cpp": "cpp", "python": "python"}

PROMPT = """Solve the following competitive programming problem.

{statement}

Write a complete {lang_name} program that reads the input from standard input and writes the answer to standard output. Do not print anything else. Put your final solution in a single ```{fence} code block."""


def build_prompt(problem: dict, lang: str) -> str:
    return PROMPT.format(statement=problem["question_content"].strip(), lang_name=LANG_NAME[lang], fence=FENCE[lang])


# ---------------------------------------------------------------- selection


def select_problems(index: list[dict], args) -> list[dict]:
    probs = [p for p in index if p["n_tests"] > 0]
    if args.platform != "all":
        probs = [p for p in probs if p["platform"] == args.platform]
    if args.difficulty:
        probs = [p for p in probs if p["difficulty"] in args.difficulty]
    if args.after:
        probs = [p for p in probs if p["contest_date"] >= args.after]
    probs.sort(key=lambda p: p["question_id"])
    if args.limit:
        probs = random.Random(args.seed).sample(probs, min(args.limit, len(probs)))
    return probs


# ---------------------------------------------------------------- summary


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def summarize(out_dir: Path) -> dict:
    rows = [json.loads(line) for line in open(out_dir / "results.jsonl")]
    by = defaultdict(list)  # (lang, qid) -> rows
    for r in rows:
        by[(r["lang"], r["question_id"])].append(r)
    langs = sorted({r["lang"] for r in rows})
    summary: dict = {"n_samples": len(rows), "langs": {}}

    for lang in langs:
        lr = [r for r in rows if r["lang"] == lang]
        groups = {q: v for (l, q), v in by.items() if l == lang}
        k_max = min(len(v) for v in groups.values())
        per_prob = {q: sum(r["verdict"] == "accepted" for r in v) / len(v) for q, v in groups.items()}
        diff_of = {q: v[0]["difficulty"] for q, v in groups.items()}
        by_diff = defaultdict(list)
        for q, rate in per_prob.items():
            by_diff[diff_of[q]].append(rate)
        verdicts = Counter(r["verdict"] for r in lr)
        subtypes = Counter(f"{r['verdict']}/{r['subtype']}" for r in lr if r.get("subtype"))
        ks = sorted({1, k_max} | ({4} if k_max >= 4 else set()))
        summary["langs"][lang] = {
            "problems": len(groups),
            "samples": len(lr),
            "pass@k": {
                k: sum(_pass_at_k(len(v), sum(r["verdict"] == "accepted" for r in v), k) for v in groups.values())
                / len(groups)
                for k in ks
            },
            "pass@1_by_difficulty": {d: (sum(v) / len(v), len(v)) for d, v in sorted(by_diff.items())},
            "verdicts": {v: c / len(lr) for v, c in verdicts.most_common()},
            "subtypes": {s: c / len(lr) for s, c in subtypes.most_common()},
            "truncated_rate": sum(not r["clean_stop"] for r in lr) / len(lr),
            "mean_tokens": sum(r["n_tokens"] for r in lr) / len(lr),
            "_per_problem": per_prob,
        }

    if {"cpp", "python"} <= set(langs):
        cpp, py = summary["langs"]["cpp"]["_per_problem"], summary["langs"]["python"]["_per_problem"]
        common = sorted(set(cpp) & set(py))
        cells = Counter()
        cpp_only_gap = []
        for q in common:
            c, p = cpp[q] >= 0.5, py[q] >= 0.5
            cells[("both" if c and p else "cpp_only" if c else "python_only" if p else "neither")] += 1
            if p and not c:
                cpp_only_gap.append(q)
        summary["cross"] = {
            "problems": len(common),
            "solved_rule": "pass rate >= 0.5",
            **{k: cells[k] for k in ("both", "python_only", "cpp_only", "neither")},
            "python_only_ids": cpp_only_gap,
        }

    for lang in langs:
        summary["langs"][lang].pop("_per_problem")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_summary(s: dict) -> None:
    for lang, d in s["langs"].items():
        pk = "  ".join(f"pass@{k}={v:.1%}" for k, v in d["pass@k"].items())
        print(f"\n== {lang}: {d['problems']} problems, {d['samples']} samples   {pk}")
        print("   by difficulty: " + "  ".join(f"{k}={v[0]:.1%} (n={v[1]})" for k, v in d["pass@1_by_difficulty"].items()))
        print(f"   truncated={d['truncated_rate']:.1%}  mean_tokens={d['mean_tokens']:.0f}")
        print("   verdicts:  " + "  ".join(f"{k}={v:.1%}" for k, v in d["verdicts"].items()))
        if d["subtypes"]:
            print("   subtypes:  " + "  ".join(f"{k}={v:.1%}" for k, v in list(d["subtypes"].items())[:8]))
    if "cross" in s:
        c = s["cross"]
        print(f"\n== same problems, both languages ({c['problems']} problems, solved = {c['solved_rule']})")
        print(f"   both={c['both']}  python_only={c['python_only']}  cpp_only={c['cpp_only']}  neither={c['neither']}")
        print("   python_only = C++-specific gap; neither = reasoning gap")


# ---------------------------------------------------------------- run


async def run(args) -> Path:
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from cppgrade.grader import grade, make_ondemand_pool
    from cppgrade.lcb import load_index

    out_dir = ROOT / "runs" / "evals" / args.name
    results_path = out_dir / "results.jsonl"
    cfg = {k: v for k, v in vars(args).items() if k not in ("summarize", "dry_run")}
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v and k not in MAY_CHANGE_ON_RESUME}
        if diff:
            raise SystemExit(f"{cfg_path} exists with different settings {diff}. Use a new --name.")

    problems = select_problems(load_index(args.split), args)
    langs = args.langs.split(",")
    done = Counter()
    if results_path.exists():
        for line in open(results_path):
            r = json.loads(line)
            done[(r["lang"], r["question_id"])] += 1
    todo = [(p, l) for p in problems for l in langs if done[(l, p["question_id"])] < args.samples]
    n_samples = sum(args.samples - done[(l, p["question_id"])] for p, l in todo)
    already = sum(min(done[(l, p["question_id"])], args.samples) for p in problems for l in langs)
    print(f"{len(problems)} problems x {langs} x {args.samples} samples: "
          f"{already} samples already done, {n_samples} to run ({len(todo)} (problem, lang) jobs)")
    if args.dry_run:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    renderer_name = args.renderer or get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, get_tokenizer(args.model), model_name=args.model)
    if args.endpoint:  # a model served on Modal (scripts/serve_hf_model.py), e.g. a teacher not on Tinker
        from cppgrade.endpoint_sampler import EndpointSampler

        url = args.endpoint
        if url == "modal":
            from scripts.serve_hf_model import endpoint_url
            url = endpoint_url()
        key = (ROOT / "data" / "serve_api_key").read_text().strip()
        client = EndpointSampler(url, key, get_tokenizer(args.model), renderer.get_stop_sequences())
        print(f"sampling from endpoint {url}")
    else:
        sc = tinker.ServiceClient()
        if args.model_path:
            client = await sc.create_sampling_client(model_path=args.model_path)
        else:
            client = await sc.create_sampling_client(base_model=args.model)
    params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=renderer.get_stop_sequences(),
    )
    pool = make_ondemand_pool(max_concurrent=args.pool)
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    stats = Counter()
    t0 = time.time()

    tokenizer = get_tokenizer(args.model)

    def _thinking_text(msg, tokens, clean: bool) -> str:
        """The model's reasoning. A truncated answer may never close <think>, so the renderer can't
        split it out; then keep the raw decoded completion so the truncation can still be read."""
        content = msg["content"]
        if not isinstance(content, str):
            thinking = "".join(p["thinking"] for p in content if p["type"] == "thinking")
            if thinking or clean:
                return thinking
        return "" if clean else tokenizer.decode(tokens)

    async def job(problem: dict, lang: str):
        qid = problem["question_id"]
        need = args.samples - done[(lang, qid)]
        async with sem:
            try:
                prompt = renderer.build_generation_prompt([{"role": "user", "content": build_prompt(problem, lang)}])
                resp = await client.sample(prompt, num_samples=need, sampling_params=params)
                parsed = []
                for seq in resp.sequences:
                    msg, term = renderer.parse_response(seq.tokens)
                    clean = term.is_clean and seq.stop_reason != "length"  # False = ran out of max_tokens
                    parsed.append((renderers.get_text_content(msg), clean, len(seq.tokens),
                                   _thinking_text(msg, seq.tokens, clean)))
                grades = await asyncio.gather(
                    *(
                        grade(pool, text, tests_file=problem["tests_file"], n_tests=problem["n_tests"],
                              lang=lang, time_limit=args.time_limit, sanitize=args.sanitize)
                        for text, _, _, _ in parsed
                    )
                )
            except Exception as e:
                stats["job_errors"] += 1
                print(f"  ! {qid}/{lang}: {type(e).__name__}: {str(e)[:200]} (will retry on re-run)")
                return
        lines = []
        for i, ((text, clean, ntok, thinking), g) in enumerate(zip(parsed, grades)):
            if g.verdict == "infra_error":
                stats["infra_errors"] += 1  # not written -> retried on re-run
                continue
            verdict = "truncated" if (g.verdict == "format_error" and not clean) else g.verdict
            stats[f"{lang}:{verdict}"] += 1
            lines.append(json.dumps({
                "question_id": qid, "lang": lang, "difficulty": problem["difficulty"],
                "contest_date": problem["contest_date"], "sample": done[(lang, qid)] + i,
                "verdict": verdict, "subtype": g.subtype, "passed": g.passed, "total": g.total,
                "n_tokens": ntok, "clean_stop": clean, "detail": g.detail[:1500],
                "code": g.code, "response": text, "thinking": thinking,
            }))
        async with write_lock:
            with open(results_path, "a") as f:
                f.write("".join(line + "\n" for line in lines))
            stats["jobs"] += 1
            if stats["jobs"] % max(1, len(todo) // 20) == 0 or stats["jobs"] == len(todo):
                acc = {l: stats[f"{l}:accepted"] for l in langs}
                print(f"  [{stats['jobs']}/{len(todo)}] {time.time() - t0:.0f}s  accepted so far: {acc}")

    await asyncio.gather(*(job(p, l) for p, l in todo))
    await pool.terminate()
    if stats["infra_errors"] or stats["job_errors"]:
        print(f"\n{stats['infra_errors']} grading infra errors, {stats['job_errors']} failed jobs: re-run the same command to retry them.")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", type=Path, help="just (re)print the summary for an output dir")
    ap.add_argument("--name", help="output dir name under runs/evals/")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="base model (also picks tokenizer/renderer)")
    ap.add_argument("--model-path", default=None, help="tinker://... sampler checkpoint; default = base model")
    ap.add_argument("--endpoint", default=None,
                    help="sample from an OpenAI-compatible endpoint instead of Tinker: a URL, or 'modal' for "
                         "the app deployed by scripts/serve_hf_model.py (--model must name the served model)")
    ap.add_argument("--renderer", default=None, help="e.g. qwen3_disable_thinking; default = recommended")
    ap.add_argument("--langs", default="cpp,python")
    ap.add_argument("--split", default="v6_2408_2505")
    ap.add_argument("--platform", default="atcoder", choices=["atcoder", "all"],
                    help="atcoder = stdin/stdout problems (LeetCode function-call problems not supported yet)")
    ap.add_argument("--difficulty", nargs="*", choices=["easy", "medium", "hard"])
    ap.add_argument("--after", default=None, help="only contests on/after this date, e.g. 2025-02-01 (contamination)")
    ap.add_argument("--limit", type=int, default=None, help="random subset of N problems (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1, help="samples per problem per language")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--time-limit", type=float, default=6.0, help="seconds per test (LiveCodeBench uses 6)")
    ap.add_argument("--sanitize", action="store_true", help="C++ with ASan/UBSan (diagnosis only)")
    ap.add_argument("--concurrency", type=int, default=32, help="problems sampling at once")
    ap.add_argument("--pool", type=int, default=32, help="max concurrent Modal sandboxes")
    ap.add_argument("--dry-run", action="store_true", help="print what would run (done vs. to-do) and exit")
    args = ap.parse_args()

    if args.summarize:
        print_summary(summarize(args.summarize))
        return
    if not args.name:
        ap.error("--name is required")
    out_dir = asyncio.run(run(args))
    if out_dir is None:  # --dry-run
        return
    s = summarize(out_dir)
    print_summary(s)
    print(f"\nWrote {out_dir.relative_to(ROOT)}/{{results.jsonl,summary.json}}")


if __name__ == "__main__":
    main()

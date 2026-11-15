"""Test B: does seeing its own failure help the model fix it? (diagnostic for "learn from failed attempts")

Takes C++ failures from a finished eval_lcb run (one per problem) in three buckets:
  wrong_approach  fails the public sample tests
  hidden_fail     passes samples, fails a hidden test (wrong answer / crash / too slow)
  compile_error
and asks the model to retry, as a second turn after its own failed answer, in two conditions:
  generic   "Your solution was judged incorrect. Try again."  (no information)
  feedback  verdict + the failing test's input / expected / actual output, or the compiler error
Compared with each problem's fresh-attempt pass rate from the original eval (no retry at all).

    uv run python scripts/retry_uplift.py --source runs/evals/qwen3-8b --name retry-uplift
    uv run python scripts/retry_uplift.py --summarize runs/evals/retry-uplift

Diagnostic only: these are eval problems, never train on the outputs.
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

from scripts.eval_lcb import build_prompt  # noqa: E402  (same first-turn prompt as the original eval)

HEAD = 1500  # chars of failing input / outputs shown to the model
GENERIC = "Your solution was judged incorrect. Try again. Put the complete corrected C++17 program in a single ```cpp code block."
FIX = "Fix your solution. Put the complete corrected C++17 program in a single ```cpp code block."


def _block(s: str, n_total: int | None = None) -> str:
    s = s.rstrip("\n")
    note = f"  [first {HEAD} of {n_total} chars]" if n_total and n_total > HEAD else ""
    return f"```\n{s[:HEAD]}\n```{note}"


def feedback_message(bucket: str, fail: dict, compile_stderr: str = "") -> str:
    if bucket == "compile_error":
        return f"Your solution did not compile:\n```\n{compile_stderr[:3000]}\n```\n{FIX}"
    k, size = fail["idx"] + 1, fail["in_bytes"]
    inp = _block(fail.get("input_head", ""), size)
    if fail["status"] == "wrong_answer":
        return (f"Wrong answer on test #{k}.\nInput:\n{inp}\nExpected output:\n{_block(fail.get('want_head', ''))}\n"
                f"Your output:\n{_block(fail.get('got', ''))}\n{FIX}")
    if fail["status"] == "time_limit":
        return f"Time limit exceeded on test #{k} (input is {size} bytes).\nInput:\n{inp}\n{FIX}"
    return (f"Runtime error on test #{k} (exit code {fail.get('rc')}).\nInput:\n{inp}\n"
            f"stderr:\n{_block(fail.get('stderr', ''))}\n{FIX}")


def select(source: Path, index: dict, n_wrong: int, n_hidden: int, n_compile: int, seed: int):
    rows = [json.loads(line) for line in open(source / "results.jsonl")]
    cpp = [r for r in rows if r["lang"] == "cpp"]
    fresh = defaultdict(list)
    for r in cpp:
        fresh[r["question_id"]].append(r["verdict"] == "accepted")
    buckets = defaultdict(dict)  # bucket -> qid -> first failing sample
    for r in sorted(cpp, key=lambda r: (r["question_id"], r["sample"])):
        if not r.get("code") or r["verdict"] not in ("wrong_answer", "runtime_error", "time_limit", "compile_error"):
            continue
        n_pub = sum(t.get("testtype") == "stdin" for t in index[r["question_id"]]["public_tests"])
        b = ("compile_error" if r["verdict"] == "compile_error"
             else "wrong_approach" if r["passed"] < n_pub else "hidden_fail")
        buckets[b].setdefault(r["question_id"], r)
    rng = random.Random(seed)
    picked = []
    for b, n in (("wrong_approach", n_wrong), ("hidden_fail", n_hidden), ("compile_error", n_compile)):
        qids = sorted(buckets[b])
        rng.shuffle(qids)
        picked += [(b, buckets[b][q]) for q in qids[:n]]
    return picked, {q: sum(v) / len(v) for q, v in fresh.items()}


async def run(args) -> Path:
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from cppgrade.grader import TESTS_MOUNT, _base_cfg, _execute, grade, make_ondemand_pool
    from cppgrade.lcb import load_index

    out_dir = ROOT / "runs" / "evals" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    src_cfg = json.loads((args.source / "config.json").read_text())
    (out_dir / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()}, "source_config": src_cfg}, indent=2))

    index = {p["question_id"]: p for p in load_index(src_cfg["split"])}
    picked, fresh = select(args.source, index, args.n_wrong, args.n_hidden, args.n_compile, args.seed)
    done = set()
    if results_path.exists():
        for line in open(results_path):
            r = json.loads(line)
            done.add((r["question_id"], r["condition"]))
    print(f"{len(picked)} failed samples picked: {Counter(b for b, _ in picked)}")

    model = src_cfg["model"]
    renderer = renderers.get_renderer(src_cfg.get("renderer") or get_recommended_renderer_name(model), get_tokenizer(model), model_name=model)
    sc = tinker.ServiceClient()
    client = await (sc.create_sampling_client(model_path=src_cfg["model_path"]) if src_cfg.get("model_path")
                    else sc.create_sampling_client(base_model=model))
    params = tinker.SamplingParams(max_tokens=src_cfg["max_tokens"], temperature=src_cfg["temperature"],
                                   top_p=src_cfg["top_p"], stop=renderer.get_stop_sequences())
    pool = make_ondemand_pool(32)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = Counter()
    t0 = time.time()
    grade_kw = dict(time_limit=src_cfg["time_limit"])  # same limits as the original eval

    async def job(bucket: str, r: dict):
        qid = r["question_id"]
        p = index[qid]
        async with sem:
            # Re-run the failed code to recover the failing test's input (and check the failure reproduces).
            fail, cerr = None, r["detail"] if bucket == "compile_error" else ""
            if bucket != "compile_error":
                cfg = _base_cfg("tests", src_cfg["time_limit"], 1024, "c++17", stop_on_first_failure=True,
                                tests_file=f"{TESTS_MOUNT}/{p['tests_file']}", report_input_chars=HEAD)
                out, err = await _execute(pool, r["code"], cfg, budget=60 + int(src_cfg["time_limit"] * p["n_tests"]) + 60)
                if out is None or not out.get("compiled"):
                    stats["skipped"] += 1
                    return
                fail = next((x for x in out["results"] if x["status"] != "ok"), None)
                if fail is None:
                    stats["did_not_reproduce"] += 1
                    return
            first = [{"role": "user", "content": build_prompt(p, "cpp")}, {"role": "assistant", "content": r["response"]}]
            convs = {"generic": GENERIC, "feedback": feedback_message(bucket, fail or {}, cerr)}
            for cond, msg in convs.items():
                if (qid, cond) in done:
                    continue
                prompt = renderer.build_generation_prompt(first + [{"role": "user", "content": msg}])
                resp = await client.sample(prompt, num_samples=1, sampling_params=params)
                seq = resp.sequences[0]
                parsed, term = renderer.parse_response(seq.tokens)
                text = renderers.get_text_content(parsed)
                g = await grade(pool, text, tests_file=p["tests_file"], n_tests=p["n_tests"], **grade_kw)
                verdict = "truncated" if (g.verdict == "format_error" and (not term.is_clean or seq.stop_reason == "length")) else g.verdict
                rec = {"question_id": qid, "difficulty": p["difficulty"], "bucket": bucket, "condition": cond,
                       "verdict": verdict, "subtype": g.subtype, "passed": g.passed, "total": g.total,
                       "fresh_pass_rate": fresh[qid], "orig_verdict": r["verdict"], "n_tokens": len(seq.tokens),
                       "feedback": msg, "code": g.code, "response": text}
                async with lock:
                    with open(results_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    stats[f"{cond}:{verdict == 'accepted'}"] += 1
                    stats["n"] += 1
                    if stats["n"] % 20 == 0:
                        print(f"  [{stats['n']}] {time.time() - t0:.0f}s  fixed so far: generic {stats['generic:True']}  feedback {stats['feedback:True']}")

    await asyncio.gather(*(job(b, r) for b, r in picked))
    print(f"skipped (infra) {stats['skipped']}, failure did not reproduce {stats['did_not_reproduce']}")
    return out_dir


def summarize(out_dir: Path) -> None:
    rows = [json.loads(line) for line in open(out_dir / "results.jsonl")]
    by = defaultdict(dict)
    for r in rows:
        by[(r["bucket"], r["question_id"])][r["condition"]] = r
    print(f"\n{'bucket':16} {'n':>3}  {'fresh attempt':>13}  {'generic retry':>13}  {'feedback retry':>14}")
    allrows = defaultdict(list)
    for bucket in ("wrong_approach", "hidden_fail", "compile_error", "ALL"):
        items = [v for (b, _), v in by.items() if (b == bucket or bucket == "ALL") and {"generic", "feedback"} <= set(v)]
        if not items:
            continue
        n = len(items)
        fresh = sum(v["generic"]["fresh_pass_rate"] for v in items) / n
        gen = sum(v["generic"]["verdict"] == "accepted" for v in items) / n
        fb = sum(v["feedback"]["verdict"] == "accepted" for v in items) / n
        print(f"{bucket:16} {n:3}  {fresh:13.0%}  {gen:13.0%}  {fb:14.0%}")
        allrows[bucket] = (n, fresh, gen, fb)
    for cond in ("generic", "feedback"):
        c = Counter(r["verdict"] for r in rows if r["condition"] == cond)
        print(f"  {cond} retry verdicts: {dict(c.most_common())}")
    hard = [v for v in by.values() if {"generic", "feedback"} <= set(v) and v["feedback"]["difficulty"] == "hard"]
    if hard:
        print(f"  hard only (n={len(hard)}): generic {sum(v['generic']['verdict'] == 'accepted' for v in hard) / len(hard):.0%}"
              f"  feedback {sum(v['feedback']['verdict'] == 'accepted' for v in hard) / len(hard):.0%}")
    (out_dir / "summary.json").write_text(json.dumps(allrows, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=ROOT / "runs/evals/qwen3-8b")
    ap.add_argument("--name", default="retry-uplift")
    ap.add_argument("--summarize", type=Path)
    ap.add_argument("--n-wrong", type=int, default=40)
    ap.add_argument("--n-hidden", type=int, default=40)
    ap.add_argument("--n-compile", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    out = asyncio.run(run(args))
    summarize(out)


if __name__ == "__main__":
    main()

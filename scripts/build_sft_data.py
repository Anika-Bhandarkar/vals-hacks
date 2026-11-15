"""Build SFT data (JSONL, see posttrain/sft_data.py) from any source, filtered by the hidden-test grader.

Prereq: the training pool on the Modal volume (once):
    uv run python scripts/prepare_codeforces.py --split train --min-rating 1200 --max-rating 2600 \
        --no-require-editorial --max-problems 2000

Sources:
  # 1. rejection sampling, single attempt: the model's own shortest correct answers.
  #    Also writes <out>.pass_rates.json (per-problem pass rate; reuse it to pick RL problems with 0 < p < 1).
  uv run python scripts/build_sft_data.py rft-single --out data/sft/rft_single.jsonl \
      --model Qwen/Qwen3-8B --pool-limit 500 --samples 4 --keep 1

  # 2. rejection sampling, agentic: the model's best competition-format sessions on training problems
  uv run python scripts/build_sft_data.py rft-agentic --out data/sft/rft_agentic.jsonl \
      --model Qwen/Qwen3-8B --pool-limit 300 --samples 2
  #    ...or filter sessions you already have (eval problems are refused):
  uv run python scripts/build_sft_data.py rft-agentic --out data/sft/rft_agentic.jsonl --from-results runs/.../results.jsonl

  # 3. external traces, kept only if correct: R1 traces from open-r1/codeforces-cots, or any JSONL of
  #    {"problem_id": ..., "response": "<think>...</think>..."} (e.g. a Modal-hosted Qwen3-32B)
  uv run python scripts/build_sft_data.py traces --out data/sft/r1_cots.jsonl --hf open-r1/codeforces-cots
  uv run python scripts/build_sft_data.py traces --out data/sft/qwen32b.jsonl --file runs/teacher/generations.jsonl

  # mix sources (caps per file), dedupe, shuffle;  and offline size check (no Tinker calls)
  uv run python scripts/build_sft_data.py mix --out data/sft/mix.jsonl --inputs data/sft/rft_agentic.jsonl data/sft/r1_cots.jsonl:2000
  uv run python scripts/build_sft_data.py stats data/sft/mix.jsonl

Every source drops eval problems (posttrain/decontam.py) and examples over --max-example-tokens.
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

from posttrain.sft_data import (  # noqa: E402
    assistant, code_hash, from_agentic_transcript, read_jsonl, record, session_ok, split_thinking, write_jsonl,
)


def _done_ids(out: Path) -> set[str]:
    return {r["meta"]["problem_id"] for r in read_jsonl(out)} if out.exists() else set()


async def _sampling_setup(args):
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer_name = args.renderer or get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, get_tokenizer(args.model), model_name=args.model)
    sc = tinker.ServiceClient()
    client = await (sc.create_sampling_client(model_path=args.model_path) if args.model_path
                    else sc.create_sampling_client(base_model=args.model))
    return renderer, client


def _pool(args, exclude: set[str] = frozenset()):
    from posttrain.problems import load_pool

    return load_pool(min_rating=args.min_rating, max_rating=args.max_rating, limit=args.pool_limit,
                     seed=args.seed, exclude_ids=set(exclude))


def _parts(msg) -> tuple[str, str]:
    c = msg["content"]
    if isinstance(c, str):
        return split_thinking(c)
    return ("".join(p.get("thinking", "") for p in c if p["type"] == "thinking"),
            "".join(p.get("text", "") for p in c if p["type"] == "text"))


# ---------------------------------------------------------------- rft-single


async def rft_single(args) -> None:
    import tinker
    from tinker_cookbook import renderers  # noqa: F401

    from cppgrade.grader import grade, make_ondemand_pool
    from posttrain.problems import limits, single_prompt

    out = Path(args.out)
    done = _done_ids(out)
    rates_path = out.with_suffix(".pass_rates.json")
    rates = json.loads(rates_path.read_text()) if rates_path.exists() else {}
    pool_rows = [p for p in _pool(args) if p["id"] not in done and p["id"] not in rates]
    print(f"rft-single: {len(pool_rows)} problems to sample ({len(rates)} already done)")
    renderer, client = await _sampling_setup(args)
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=0.95,
                                   stop=renderer.get_stop_sequences())
    gpool = make_ondemand_pool(args.sandboxes)
    sem, lock, stats, t0 = asyncio.Semaphore(args.concurrency), asyncio.Lock(), Counter(), time.time()

    async def job(p):
        async with sem:
            try:
                prompt_text = single_prompt(p)
                prompt = renderer.build_generation_prompt([{"role": "user", "content": prompt_text}])
                resp = await client.sample(prompt, num_samples=args.samples, sampling_params=params)
                tl, mem = limits(p)
                cands = []
                for seq in resp.sequences:
                    msg, term = renderer.parse_response(seq.tokens)
                    thinking, text = _parts(msg)
                    if seq.stop_reason == "length" or not term.is_clean:
                        cands.append((None, seq, thinking, text))
                        stats["v:truncated"] += 1
                        continue
                    g = await grade(gpool, text, tests_file=p["tests_file"], n_tests=p["n_tests"],
                                    time_limit=tl, memory_mb=mem, cpp_std="c++20")
                    stats[f"v:{g.verdict}"] += 1
                    cands.append((g, seq, thinking, text))
            except Exception as e:
                stats["errors"] += 1
                print(f"  ! {p['id']}: {type(e).__name__}: {str(e)[:200]}")
                return
        good = sorted((c for c in cands if c[0] is not None and c[0].accepted and len(c[1].tokens) <= args.max_example_tokens),
                      key=lambda c: len(c[1].tokens))
        recs = [record([{"role": "user", "content": prompt_text}, assistant(text, thinking)],
                       source="rft-single", problem_id=p["id"], rating=p.get("rating"), n_tokens=len(seq.tokens),
                       pass_rate=sum(c[0] is not None and c[0].accepted for c in cands) / len(cands), code_hash=code_hash(g.code))
                for g, seq, thinking, text in good[: args.keep]]
        async with lock:
            write_jsonl(out, recs, append=True)
            rates[p["id"]] = sum(c[0] is not None and c[0].accepted for c in cands) / len(cands)
            rates_path.write_text(json.dumps(rates))
            stats["problems"] += 1
            stats["kept"] += len(recs)
            if stats["problems"] % 20 == 0 or stats["problems"] == len(pool_rows):
                solved = sum(v > 0 for v in rates.values())
                print(f"  [{stats['problems']}/{len(pool_rows)}] {time.time() - t0:.0f}s  kept {stats['kept']}  "
                      f"problems with >=1 correct: {solved}/{len(rates)}")

    await asyncio.gather(*(job(p) for p in pool_rows))
    vals = list(rates.values())
    print("sample verdicts:", {k[2:]: v for k, v in stats.items() if k.startswith("v:")})
    print(f"done: {stats['kept']} examples; pass-rate buckets: "
          f"0: {sum(v == 0 for v in vals)}  (0,1): {sum(0 < v < 1 for v in vals)}  1: {sum(v == 1 for v in vals)}"
          f"  -> {rates_path.relative_to(ROOT) if rates_path.is_relative_to(ROOT) else rates_path}")


# ---------------------------------------------------------------- rft-agentic


def _filter_sessions(sessions: list[dict], args) -> list[dict]:
    from posttrain.decontam import eval_ids

    blocked = eval_ids()
    by_problem: dict[str, list[dict]] = defaultdict(list)
    reasons = Counter()
    for s in sessions:
        if s["question_id"] in blocked:
            reasons["eval_problem"] += 1
            continue
        ok, why = session_ok(s, min_score=args.min_score, max_turns=args.max_turns_keep,
                             allow_resubmits=args.allow_resubmits, require_exit=not args.no_require_exit)
        reasons[why] += 1
        if ok:
            by_problem[s["question_id"]].append(s)
    print(f"session filter: {dict(reasons.most_common())}")
    recs = []
    for qid, ss in by_problem.items():
        for s in sorted(ss, key=lambda s: (s["turns"], s["sampled_tokens"]))[: args.keep]:
            recs.append(record(from_agentic_transcript(s["transcript"]), source="rft-agentic", problem_id=qid,
                               score=s["score"], turns=s["turns"], sampled_tokens=s["sampled_tokens"]))
    return recs


async def rft_agentic(args) -> None:
    out = Path(args.out)
    if args.from_results:
        sessions = [s for f in args.from_results for s in read_jsonl(f)]
        n = write_jsonl(out, _filter_sessions(sessions, args))
        print(f"wrote {n} sessions -> {out}")
        return

    from tinker_cookbook.completers import TinkerTokenCompleter
    from tinker_cookbook.rl.rollouts import do_single_rollout

    from cppgrade.agent_env import EnvSettings, build_competition_env
    from cppgrade.grader import make_ondemand_pool
    from posttrain.problems import agent_task
    from scripts.eval_agentic import _is_exit, _msg_to_json

    raw_path = out.with_suffix(".sessions.jsonl")  # every session, kept for analysis / RL warm-start
    done = Counter(s["question_id"] for s in read_jsonl(raw_path)) if raw_path.exists() else Counter()
    rows = [p for p in _pool(args) if done[p["id"]] < args.samples]
    print(f"rft-agentic: {len(rows)} problems x up to {args.samples} sessions")
    renderer, client = await _sampling_setup(args)
    policy = TinkerTokenCompleter(client, max_tokens=args.max_tokens, temperature=args.temperature,
                                  context_window=args.context_window)
    settings = EnvSettings(max_turns=args.max_turns, context_window=args.context_window)
    gpool = make_ondemand_pool(args.sandboxes)
    sem, lock, stats, t0 = asyncio.Semaphore(args.concurrency), asyncio.Lock(), Counter(), time.time()

    async def job(p, k):
        task = agent_task(p)
        async with sem:
            try:
                env, tools = build_competition_env(task, renderer, gpool, settings)
                traj = await do_single_rollout(policy, env)
                final = await tools.final_score()
                scored = [await tools.score_code(c) for c in dict.fromkeys(s.code for s in tools.submissions if s.compiled)]
            except Exception as e:
                stats["errors"] += 1
                print(f"  ! {p['id']}: {type(e).__name__}: {str(e)[:200]}")
                return
        hist = [_msg_to_json(m) for m in env.message_env.history]
        rec = {"question_id": p["id"], "sample": k, "rating": p.get("rating"), "score": final.score,
               "best_score": max([x.score for x in scored], default=0.0), "stop_reason": traj.stop_reason,
               "turns": sum(m["role"] == "assistant" for m in hist), "clean_exit": _is_exit(hist),
               "parse_errors": int(sum(t.metrics.get("parse_error", 0) for t in traj.transitions)),
               "sampled_tokens": sum(len(t.ac.tokens) for t in traj.transitions), "transcript": hist}
        async with lock:
            write_jsonl(raw_path, [rec], append=True)
            stats["n"] += 1
            stats["score"] += final.score
            if stats["n"] % 20 == 0:
                print(f"  [{stats['n']}] {time.time() - t0:.0f}s  mean score {stats['score'] / stats['n']:.1f}")

    await asyncio.gather(*(job(p, k) for p in rows for k in range(done[p["id"]], args.samples)))
    n = write_jsonl(out, _filter_sessions(list(read_jsonl(raw_path)), args))
    print(f"wrote {n} sessions -> {out} (all sessions in {raw_path.name})")


# ---------------------------------------------------------------- traces


def _load_traces(args, pool_ids: set[str]) -> list[dict]:
    """-> [{"problem_id", "response"}]"""
    if args.file:
        return [{"problem_id": r["problem_id"], "response": r.get("response") or r["messages"][-1]["content"]}
                for r in read_jsonl(args.file) if r["problem_id"] in pool_ids]
    import datasets

    ds = datasets.load_dataset(args.hf, args.hf_config, split="train")
    ds = ds.filter(lambda r: r["id"] in pool_ids and r.get("finish_reason", "stop") == "stop")
    return [{"problem_id": r["id"], "response": r["generation"]} for r in ds]


async def traces(args) -> None:
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from cppgrade.grader import grade, make_ondemand_pool
    from posttrain.problems import limits, single_prompt

    pool = {p["id"]: p for p in _pool(args)}
    cands = _load_traces(args, set(pool))
    tok = get_tokenizer(args.model)
    out = Path(args.out)
    done = {(r["meta"]["problem_id"], r["meta"]["code_hash"]) for r in read_jsonl(out)} if out.exists() else set()
    print(f"traces: {len(cands)} candidates on {len({c['problem_id'] for c in cands})} pool problems")
    gpool = make_ondemand_pool(args.sandboxes)
    sem, lock, stats = asyncio.Semaphore(args.concurrency), asyncio.Lock(), Counter()
    per_problem = Counter(r["meta"]["problem_id"] for r in read_jsonl(out)) if out.exists() else Counter()

    async def job(c):
        p = pool[c["problem_id"]]
        thinking, text = split_thinking(c["response"])
        n_tok = len(tok.encode(c["response"]))
        if n_tok > args.max_example_tokens:
            stats["too_long"] += 1
            return
        async with sem:
            tl, mem = limits(p)
            g = await grade(gpool, text, tests_file=p["tests_file"], n_tests=p["n_tests"], time_limit=tl,
                            memory_mb=mem, cpp_std="c++20")
        stats[g.verdict] += 1
        if not g.accepted:
            return
        key = (p["id"], code_hash(g.code))
        async with lock:
            if key in done or per_problem[p["id"]] >= args.keep:
                stats["dup_or_cap"] += 1
                return
            done.add(key)
            per_problem[p["id"]] += 1
            write_jsonl(out, [record([{"role": "user", "content": single_prompt(p)}, assistant(text, thinking)],
                                     source=f"traces:{args.file or args.hf}", problem_id=p["id"], rating=p.get("rating"),
                                     n_tokens=n_tok, code_hash=key[1])], append=True)

    # shortest first, so the per-problem cap keeps the most concise correct traces
    cands.sort(key=lambda c: len(c["response"]))
    await asyncio.gather(*(job(c) for c in cands))
    print(f"traces done: {dict(stats.most_common())}; kept {sum(per_problem.values())} -> {out}")


# ---------------------------------------------------------------- mix / stats


def mix(args) -> None:
    rng = random.Random(args.seed)
    recs, seen = [], set()
    for spec in args.inputs:
        path, _, cap = spec.partition(":")
        rs = list(read_jsonl(path))
        if cap:
            rng.shuffle(rs)
            rs = rs[: int(cap)]
        for r in rs:
            key = (r["meta"].get("problem_id"), r["meta"].get("code_hash") or json.dumps(r["messages"])[-2000:])
            if key not in seen:
                seen.add(key)
                recs.append(r)
        print(f"  {path}: {len(rs)}")
    rng.shuffle(recs)
    print(f"wrote {write_jsonl(args.out, recs)} -> {args.out}  sources: {dict(Counter(r['meta'].get('source') for r in recs))}")


def stats(args) -> None:
    from posttrain.sft_dataset import count_examples

    print(count_examples(args.files, args.model, args.renderer or "qwen3", args.max_length))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", required=True)
        p.add_argument("--model", default="Qwen/Qwen3-8B")
        p.add_argument("--model-path", default=None, help="tinker:// sampler checkpoint (default: base model)")
        p.add_argument("--renderer", default=None)
        p.add_argument("--pool-limit", type=int, default=None)
        p.add_argument("--min-rating", type=int, default=0)
        p.add_argument("--max-rating", type=int, default=10_000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-tokens", type=int, default=28000, help="sampling budget (per turn for agentic)")
        p.add_argument("--max-example-tokens", type=int, default=28000, help="drop longer examples")
        p.add_argument("--temperature", type=float, default=0.6)
        p.add_argument("--keep", type=int, default=1, help="examples kept per problem")
        p.add_argument("--concurrency", type=int, default=32)
        p.add_argument("--sandboxes", type=int, default=32)

    p = sub.add_parser("rft-single"); common(p)
    p.add_argument("--samples", type=int, default=4)
    p = sub.add_parser("rft-agentic"); common(p)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--from-results", nargs="*")
    p.add_argument("--max-turns", type=int, default=15, help="session cap while sampling")
    p.add_argument("--context-window", type=int, default=32768)
    p.add_argument("--min-score", type=float, default=100.0)
    p.add_argument("--max-turns-keep", type=int, default=8, help="keep only sessions with at most this many turns")
    p.add_argument("--allow-resubmits", type=int, default=0)
    p.add_argument("--no-require-exit", action="store_true")
    p = sub.add_parser("traces"); common(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hf", help="HF dataset with id/generation columns, e.g. open-r1/codeforces-cots")
    g.add_argument("--file", help='JSONL of {"problem_id", "response"}')
    p.add_argument("--hf-config", default="solutions_decontaminated")
    p = sub.add_parser("mix")
    p.add_argument("--out", required=True)
    p.add_argument("--inputs", nargs="+", required=True, help="path or path:cap")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("stats")
    p.add_argument("files", nargs="+")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--renderer", default=None)
    p.add_argument("--max-length", type=int, default=32768)

    args = ap.parse_args()
    if args.cmd == "rft-single":
        asyncio.run(rft_single(args))
    elif args.cmd == "rft-agentic":
        asyncio.run(rft_agentic(args))
    elif args.cmd == "traces":
        asyncio.run(traces(args))
    elif args.cmd == "mix":
        mix(args)
    else:
        stats(args)


if __name__ == "__main__":
    main()

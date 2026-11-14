"""Tinker smoke test: lists models (flagging <=9B active params), then samples one C++ answer
and grades it on Modal.

    uv run python scripts/check_tinker.py                       # list + sample Qwen/Qwen3-8B
    uv run python scripts/check_tinker.py --model Qwen/Qwen3.5-35B-A3B
    uv run python scripts/check_tinker.py --list-only
    uv run python scripts/check_tinker.py --no-grade            # skip Modal

Needs TINKER_API_KEY in the environment or in .env. Costs a few cents at most.
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import tinker  # noqa: E402
from tinker_cookbook import renderers  # noqa: E402
from tinker_cookbook.model_info import get_model_attributes, get_recommended_renderer_name  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

ACTIVE_LIMIT_B = 9.0

PROMPT = (
    "Read two integers a and b (|a|,|b| <= 1e18) from standard input and print a+b. "
    "Write a complete C++17 program in a single ```cpp code block."
)
TESTS = [
    {"input": "1 2\n", "output": "3\n"},
    {"input": "1000000000000000000 -1\n", "output": "999999999999999999\n"},
]


def active_params_b(model_name: str) -> float | None:
    """'30B-A3B' -> 3.0, '8B' -> 8.0, '21B-A3.6B' -> 3.6. None if unknown to the cookbook."""
    try:
        size = get_model_attributes(model_name).size_str
    except Exception:
        return None
    m = re.search(r"A([\d.]+)B", size) or re.match(r"([\d.]+)B", size)
    return float(m.group(1)) if m else None


def list_models(sc: tinker.ServiceClient) -> None:
    caps = tinker.RestClient().get_server_capabilities().result()
    rows = []
    for m in caps.supported_models:
        act = active_params_b(m.model_name or "")
        rows.append((m.model_name, act, m.trainable, m.max_context_length))
    rows.sort(key=lambda r: (r[1] is None, r[1] or 0, r[0] or ""))
    print(f"{'model':55} {'active':>7} {'train':>6} {'ctx':>8}  <=9B-active?")
    for name, act, tr, ctx in rows:
        ok = "yes" if act is not None and act <= ACTIVE_LIMIT_B else ("?" if act is None else "")
        print(f"{name or '?':55} {('%.1fB' % act) if act else '?':>7} {str(tr):>6} {ctx or '?':>8}  {ok}")


async def sample_and_grade(sc: tinker.ServiceClient, model: str, renderer_name: str | None, grade_it: bool):
    renderer_name = renderer_name or get_recommended_renderer_name(model)
    tok = get_tokenizer(model)
    renderer = renderers.get_renderer(renderer_name, tok, model_name=model)
    client = await sc.create_sampling_client(base_model=model)

    prompt = renderer.build_generation_prompt([{"role": "user", "content": PROMPT}])
    params = tinker.SamplingParams(max_tokens=4096, temperature=0.6, stop=renderer.get_stop_sequences())
    resp = await client.sample(prompt, num_samples=1, sampling_params=params)
    seq = resp.sequences[0]
    msg, _ = renderer.parse_response(seq.tokens)
    text = renderers.get_text_content(msg)

    print(f"\n--- {model} (renderer={renderer_name}, {len(seq.tokens)} tokens) ---")
    print(text[-1500:])

    if grade_it:
        from cppgrade.grader import grade, make_pool

        pool = make_pool(pool_size=1)
        try:
            r = await grade(pool, text, TESTS)
        finally:
            await pool.terminate()
        print(f"\nGrade: {r.verdict}/{r.subtype or '-'}  passed={r.passed}/{r.total}")
        if r.detail:
            print(r.detail[:800])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--renderer", default=None, help="override, e.g. qwen3_disable_thinking")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--no-grade", action="store_true")
    args = ap.parse_args()

    sc = tinker.ServiceClient()
    list_models(sc)
    if not args.list_only:
        asyncio.run(sample_and_grade(sc, args.model, args.renderer, not args.no_grade))
    print("\nTinker OK. Console:", sc.get_console_url())


if __name__ == "__main__":
    main()

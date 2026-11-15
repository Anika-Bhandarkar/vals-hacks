"""Create a Tinker training run with an untrained LoRA and save it for sampling.

A fresh LoRA has B=0, so this checkpoint behaves exactly like the base model. Use it to test the
submission flow and to get a reference score.

    uv run python scripts/save_baseline.py                       # Qwen/Qwen3-8B
    uv run python scripts/save_baseline.py --model Qwen/Qwen3.5-4B

Then it samples the saved checkpoint on A+B and grades it on Modal.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import tinker  # noqa: E402
from tinker_cookbook import renderers  # noqa: E402
from tinker_cookbook.model_info import get_recommended_renderer_name  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from scripts.check_tinker import PROMPT, TESTS  # noqa: E402


async def main(model: str, name: str, rank: int):
    sc = tinker.ServiceClient()
    tc = await sc.create_lora_training_client(
        base_model=model, rank=rank, user_metadata={"purpose": "baseline", "steps": "0"}
    )
    saved = await tc.save_weights_for_sampler(name=name, user_metadata={"purpose": "baseline", "steps": "0"})
    path = saved.path
    print(f"Sampler checkpoint: {path}")

    # Sanity check: sample from the checkpoint itself, then grade on Modal.
    tok = get_tokenizer(model)
    renderer = renderers.get_renderer(get_recommended_renderer_name(model), tok, model_name=model)
    client = await sc.create_sampling_client(model_path=path)
    prompt = renderer.build_generation_prompt([{"role": "user", "content": PROMPT}])
    params = tinker.SamplingParams(max_tokens=4096, temperature=0.6, stop=renderer.get_stop_sequences())
    resp = await client.sample(prompt, num_samples=1, sampling_params=params)
    msg, _ = renderer.parse_response(resp.sequences[0].tokens)
    text = renderers.get_text_content(msg)

    from cppgrade.grader import grade, make_pool

    pool = make_pool(pool_size=1)
    try:
        r = await grade(pool, text, TESTS)
    finally:
        await pool.terminate()
    print(f"Checkpoint sample graded: {r.verdict}/{r.subtype or '-'} passed={r.passed}/{r.total}")
    print(f"\nSubmit this path: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--name", default="baseline-step0")
    ap.add_argument("--rank", type=int, default=32)
    a = ap.parse_args()
    asyncio.run(main(a.model, a.name, a.rank))

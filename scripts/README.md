# scripts

## Smoke tests

```bash
uv run python scripts/check_modal.py          # 9 programs, one per verdict; should print 9× PASS
uv run python scripts/check_tinker.py --list-only   # Tinker models, flagged by <=9B active params
uv run python scripts/check_tinker.py --model Qwen/Qwen3-8B   # sample A+B, grade on Modal
```

## LiveCodeBench-CPP eval (`scripts/eval_lcb.py`)

Samples on Tinker and grades on Modal, in **C++ and Python on the same problems**. Problems a model solves in Python but not C++ show the C++-specific gap; problems it solves in neither show the reasoning gap.

```bash
uv run python scripts/prepare_lcb.py        # once: loads nvidia/LiveCodeBench-CPP into a Modal Volume (runs on Modal)
uv run python scripts/eval_lcb.py --model Qwen/Qwen3-8B --limit 5 --name smoke          # quick check
uv run python scripts/eval_lcb.py --model Qwen/Qwen3-8B --samples 4 --name qwen3-8b     # all AtCoder problems
uv run python scripts/eval_lcb.py --model Qwen/Qwen3-8B --model-path tinker://<run>/sampler_weights/<ckpt> --name ckpt-100
uv run python scripts/eval_lcb.py --summarize runs/evals/qwen3-8b
```

- **Problems:** the AtCoder (stdin/stdout) subset of the v6 split, 287 problems. LeetCode function-call problems aren't supported yet.
- **Tests:** public plus private tests, 6s per test (LiveCodeBench's limit), 1 CPU and 2GB per sandbox. Tests are read from the Volume, so nothing large is uploaded per sample.
- **Output:** `runs/evals/<name>/results.jsonl` has one line per sample: verdict, subtype, tokens, extracted code and the full answer. `summary.json` has pass@k, breakdowns by difficulty and verdict, the truncation rate, and the cross-language table.
- **Resume:** re-running the same `--name` skips finished samples and retries failed ones.
  - You can change `--limit`, `--samples`, `--seed`, `--difficulty` and `--after` under the same name to grow a run, e.g. `--limit 50` → no limit, or `--samples 1` → `4`. A seeded `--limit N` subset is stable and contained in the full set.
  - Changing any setting that affects how an answer is produced or graded (model, tokens, temperature, time limit, …) is refused.
  - `--dry-run` shows how many samples are done and how many are left.
- **Useful flags:**
  - `--after 2025-02-01`: newer contests only, to reduce contamination
  - `--difficulty easy medium`
  - `--renderer qwen3_disable_thinking`: no thinking mode
  - `--max-tokens`: default 16384; truncated answers are labeled `truncated`
  - `--sanitize`: ASan/UBSan builds for C++

## Agentic eval (`scripts/eval_agentic.py`, `scripts/eval_agentic_modal.py`)

The competition format: multi-turn, with a compile-and-run tool and a submit tool. The last compiling submission is graded on the hidden tests (the same LiveCodeBench-CPP AtCoder problems as above).

```bash
uv run python scripts/eval_agentic.py --model Qwen/Qwen3-8B --limit 3 --name agent-smoke                      # quick check
uv run python scripts/eval_agentic.py --model Qwen/Qwen3.5-9B --model-path tinker://.../sampler_weights/x --limit 50 --name agent-ckpt
uv run python scripts/eval_agentic.py --summarize runs/evals/agent-ckpt                                          # re-print the summary
uv run python scripts/eval_agentic.py --show runs/evals/agent-ckpt abc365_e                                      # one transcript

# matching the competition harness (github.com/vals-ai/hackathon-post-training-agent): its tools and prompt,
# earlier thinking kept, 16K tokens per turn, temperature 1.0; all 287 problems, 2 samples each
uv run python scripts/eval_agentic.py --name lcb287-ckpt --model Qwen/Qwen3.5-9B --model-path tinker://... \
    --harness vals --renderer qwen3_5_preserve_thinking --difficulty easy medium hard --limit 287 --samples 2 \
    --max-tokens 16384 --max-turns 20 --max-parse-errors 3 --temperature 1.0 --concurrency 128 --pool 128
```

- **`--harness`:**
  - `ours` (the default): tools `run_cpp` and `submit`, with our system prompt.
  - `vals`: the competition harness's `cpp_executor` and `submission` tools, their descriptions and JSON outputs, and its instructions as the only user message.
- **Metrics** (per session in `results.jsonl`; averaged in `summary.json`):
  - `score_proportional`: partial credit, 40 × share of small tests passed + 60 × share of large tests passed. Closest to the competition's grading.
  - `score`: 40 if every small test passes, plus 60 if every large one does.
  - `score == 100`: passes every test (Codeforces-style accepted, i.e. pass@1).
  - Small vs large is a proxy by input size (`cppgrade/grader.py`).
- **Defaults:** medium + hard problems, 15 turns, 28000 tokens per turn, temperature 0.6, the model's own Tinker context window.
- **Output:** `runs/evals/<name>/` holds `config.json`, `results.jsonl` (one line per session, with the full transcript, tool calls and per-test verdicts) and `summary.json`.
- **Resume:** re-running the same `--name` resumes, and `--limit` / `--samples` can grow under it.
- **On Modal** (`eval_agentic_modal.py`): runs the same script in a CPU container, so it survives the laptop closing, and bills the Tinker account you pick.
  - All `eval_agentic.py` flags pass through.
  - `--account main|personal` picks the Modal Secret holding that account's key (`tinker-api` or `tinker-api-personal`).
  - Results go to the `cpp-posttrain-runs` Volume at `evals/<name>/`, committed every 30 s.

  ```bash
  uv run modal run --detach scripts/eval_agentic_modal.py --account personal --name lcb287-ckpt <eval_agentic.py flags>
  uv run modal run scripts/eval_agentic_modal.py --pull lcb287-ckpt        # copy to runs/evals/lcb287-ckpt/
  ```

  A checkpoint trained on one account can only be sampled from the other if it's published (`tinker checkpoint publish <sampler path>`).

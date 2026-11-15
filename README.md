# C++ post-training workspace (Tinker + Modal)

- **Tinker** does sampling and LoRA training (the GPU side).
- **Modal** runs CPU sandboxes that compile and run model-written C++ (eval and RL reward).

## One-time auth (run these yourself)

```bash
uv run tinker auth login        # browser flow; or put TINKER_API_KEY in .env (see .env.example)
uv run tinker auth status
uv run modal token new          # browser flow; writes ~/.modal.toml
```

In the Modal dashboard (Settings → Usage & Billing), set a **workspace spend limit** below $200 so a runaway sandbox pool can't eat the budget.

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

## Grader (`cppgrade/grader.py`)

`await grade(pool, model_text, tests)` returns a verdict plus subtype, using the failure taxonomy from the diagnosis plan:

| verdict | subtype |
|---|---|
| `format_error` | no fenced code block found (`truncated` in evals if the answer ran out of tokens) |
| `compile_error` | `missing_include`, `undeclared`, `type_error`, `syntax`, `other`, `timeout` |
| `runtime_error` | C++: `segfault`, `abort`, `asan_oob`, `asan_other`, `ubsan`, `nonzero_exit`. Python: the exception name, e.g. `IndexError` |
| `time_limit` | |
| `wrong_answer` | |
| `accepted` | |

- Compiler is GCC 14 (Debian 13, from Modal's Python 3.12 slim image), `-std=c++17 -O2`. `bits/stdc++.h` works.
- `sanitize=True` builds with ASan and UBSan, so silent out-of-bounds bugs show up as `asan_oob` instead of a wrong answer. It's slower, so use it for diagnosis, not for the RL reward.
- Output check compares whitespace-separated tokens, with relative float tolerance `float_tol`.
- `lang="python"` runs the same tests with `python3`. A SyntaxError counts as `compile_error/syntax`.
- Two sandbox pools, both one sandbox per grade:
  - `make_ondemand_pool()` creates sandboxes as needed and leaves nothing idle. It's best for evals.
  - `make_pool()` keeps a warm pool, for low latency in RL. Set its size with `MODAL_POOL_SIZE`.
- Tests are passed either inline (`tests=[...]`) or as a file on the Volume (`tests_file=..., n_tests=...`).

## Reference

- RL recipe to adapt: `tinker_cookbook/recipes/code_rl/` (DeepCoder-style, Python-graded; swap in `cppgrade.grade` for the reward).
- **The rule is ≤9B *total* parameters.** Tinker models that qualify: Qwen/Qwen3-8B (8.19B) and Qwen/Qwen3.5-4B (4.66B). Qwen3.5-9B is 9.65B including its vision encoder, so check with the organizers. Run `check_tinker.py --list-only` for the current list.

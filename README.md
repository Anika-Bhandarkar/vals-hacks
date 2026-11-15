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

## SFT (`posttrain/`, `scripts/build_sft_data.py`, `scripts/train_sft.py`)

Every SFT source writes one JSONL format (`posttrain/sft_data.py`). Every example is checked against the hidden-test grader, and eval problems are removed (`posttrain/decontam.py`: ID blocklist plus overlap with eval statements).

```bash
# 1. training problem pool on the Modal Volume (once): Codeforces train split, newest first
uv run python scripts/prepare_codeforces.py --split train --min-rating 1200 --max-rating 2600 --no-require-editorial --max-problems 2000

# 2. data: pick any of these
uv run python scripts/build_sft_data.py rft-single  --out data/sft/rft_single.jsonl  --pool-limit 500 --samples 4   # own shortest correct answers
uv run python scripts/build_sft_data.py rft-agentic --out data/sft/rft_agentic.jsonl --pool-limit 300 --samples 2   # own best competition sessions
uv run python scripts/build_sft_data.py traces --out data/sft/r1_cots.jsonl --hf open-r1/codeforces-cots           # teacher traces, correct ones only
uv run python scripts/build_sft_data.py traces --out data/sft/teacher.jsonl --file generations.jsonl               # any {"problem_id","response"} file
uv run python scripts/build_sft_data.py mix --out data/sft/mix.jsonl --inputs data/sft/rft_agentic.jsonl data/sft/r1_cots.jsonl:3000
uv run python scripts/build_sft_data.py stats data/sft/mix.jsonl                                                    # offline: examples, tokens, too-long

# 3. train (LoRA on Tinker); --dry-run checks data and config without calling Tinker
uv run python scripts/train_sft.py --name sft-v1 --data data/sft/mix.jsonl --dry-run
uv run python scripts/train_sft.py --name sft-v1 --data data/sft/mix.jsonl --save-every 25 --eval-agentic 20 --eval-every 50

# 4. evaluate the final checkpoint with the full evals (the command is printed at the end of training)
uv run python scripts/eval_agentic.py --model Qwen/Qwen3-8B --model-path tinker://.../sampler_weights/final --limit 50 --name agent-sft-v1
```

- **Multi-turn sessions** become one training example per assistant turn. Each example's context drops earlier thinking, so it matches the prompt the model sees at inference with Qwen3's template, token for token.
- **Session filter** (`rft-agentic`): score 100, at most 8 turns, no identical resubmits, no malformed calls, no regression, ends with `EXIT`. Each threshold is a flag.
- **Too-long examples are dropped, not truncated.** Truncating would cut off the final code.
- `rft-single` also writes `<out>.pass_rates.json`. Problems with a pass rate strictly between 0 and 1 are the ones RL learns from.
- **In-training evals** (`--eval-single N`, `--eval-agentic N`) run on the first N held-out LiveCodeBench problems, the same subset the eval scripts use. Keep N small; the eval scripts are the source of truth.
- **Checkpoints and metrics:** `runs/train/<name>/checkpoints.jsonl` and `metrics.jsonl`. Periodic checkpoints expire after `--ttl-days`; the final one is kept.

## Distillation (`posttrain/distill.py`, `scripts/build_distill_data.py`, `scripts/train_distill.py`)

On-policy distillation on Tinker, built on the cookbook's SDFT loop. The **student** (LoRA) sees only the problem and generates its own solution. A frozen **teacher** scores those same tokens, and the student is trained toward the teacher's top-20 token distribution. There are two ways to run it:

- **Self-distillation** (the default): the teacher is the student's own base model, and it sees the problem plus a **privileged context**, such as the editorial or a failed attempt. The student learns what it would have written had it known the hint, without needing the hint at test time.
- **From a bigger teacher** (`--teacher-model`): for example, Qwen3.8-27B scoring Qwen3.5-9B's answers. The teacher sees the student's exact prompt by default (`--teacher-prompt same`), or the prompt plus a hint (`--teacher-prompt hint`).
  - Teacher and student must share a tokenizer, which is checked at startup.
  - The context length defaults to the smaller of the two models' Tinker limits.

```bash
# contexts (one row per training problem; priority order with a comma list)
uv run python scripts/build_distill_data.py --context editorial --out data/distill/editorial.jsonl
uv run python scripts/build_distill_data.py --context feedback --failures data/sft/rft_single.failures.jsonl --out data/distill/feedback.jsonl
uv run python scripts/build_distill_data.py --context editorial,feedback,tags --failures data/sft/rft_single.failures.jsonl \
    --pass-rates data/sft/rft_single.pass_rates.json --max-pass-rate 0.75 --out data/distill/mixed.jsonl

# no hint at all, for a bigger teacher scoring plain prompts
uv run python scripts/build_distill_data.py --context none --out data/distill/plain.jsonl

# train (--dry-run: teacher prompt, context-length check, cost estimate from billed Tinker rates)
uv run python scripts/train_distill.py --name sd-v1 --data data/distill/mixed.jsonl --dry-run
uv run python scripts/train_distill.py --name sd-v1 --data data/distill/mixed.jsonl --batch-size 16 --group-size 4 --eval-single 30 --eval-every 10
uv run python scripts/train_distill.py --name opd-27b-9b-v1 --data data/distill/plain.jsonl \
    --model Qwen/Qwen3.5-9B --teacher-model Qwen/Qwen3.8-27B --lora-rank 32 --max-tokens 24000 --eval-single 30 --eval-every 10
```

- **Context types:**
  - `editorial`: the official solution sketch, with code stripped.
  - `tags`: topic tags only, a weak hint.
  - `reference`: a correct solution, from `rft-single` or `traces` output.
  - `feedback`: one of the model's own failed attempts and its verdict, from `rft-single`'s `.failures.jsonl`.
- **Gate it on Test A first.** If the hinted model doesn't solve much more than the plain one, the teacher has nothing to teach.
- **Student budget:** `--max-tokens` defaults to 24000. The dry run checks that teacher prompt plus completion fits the 32K context; overlong completions are trimmed before scoring.
- **Teacher model:** in self-distillation, the teacher is always the base model, even with `--load-checkpoint`. Use `--teacher-sync-every N` to copy the student into the teacher periodically; that option is refused with a different teacher.
- **Answer lengths:** `runs/train/<name>/student_lengths.jsonl` records every student answer's length and whether it was cut off, per step. The cookbook loop only logs means.

## Reference

- RL recipe to adapt: `tinker_cookbook/recipes/code_rl/` (DeepCoder-style, Python-graded; swap in `cppgrade.grade` for the reward).
- **The rule is ≤9B *total* parameters.** Tinker models that qualify: Qwen/Qwen3-8B (8.19B) and Qwen/Qwen3.5-4B (4.66B). Qwen3.5-9B is 9.65B including its vision encoder, so check with the organizers. Run `check_tinker.py --list-only` for the current list.

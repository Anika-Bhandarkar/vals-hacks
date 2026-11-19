# posttrain

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

- **Pool parts:** the training pool is split into disjoint parts (`posttrain/problems.py`): `distill` 40%, `rl` 50%, `holdout` 10%. The split is by contest round, with re-posted problems grouped together, and the same problem always lands in the same part.
  - `build_distill_data.py` uses `--part distill` by default.
  - RL uses `load_pool(part="rl")`.
  - Never train on `holdout`; it's for checking memorization.
- **Context types:**
  - `editorial`: the official solution sketch, with code stripped.
  - `tags`: topic tags only, a weak hint.
  - `reference`: a correct solution, from `rft-single` or `traces` output.
  - `feedback`: one of the model's own failed attempts and its verdict, from `rft-single`'s `.failures.jsonl`.
- **Gate it on Test A first.** If the hinted model doesn't solve much more than the plain one, the teacher has nothing to teach.
- **Student budget:** `--max-tokens` defaults to 24000. The dry run checks that teacher prompt plus completion fits the 32K context; overlong completions are trimmed before scoring.
- **Teacher model:** in self-distillation, the teacher is always the base model, even with `--load-checkpoint`. Use `--teacher-sync-every N` to copy the student into the teacher periodically; that option is refused with a different teacher.
- **Answer lengths:** `runs/train/<name>/student_lengths.jsonl` records every student answer's length and whether it was cut off, per step. The cookbook loop only logs means.
- **Controls the cookbook loop lacks** (added in `posttrain/distill.py`; the defaults keep the old behaviour):
  - `--lr-schedule constant|linear|cosine`, `--warmup-steps N`, `--min-lr-frac`: the per-step LR is logged as `optim/lr`.
  - `--cutoff keep|drop|force`, for answers that hit `--max-tokens`.
    - `drop`: the answer isn't scored or trained, so it costs sampling only.
    - `force`: append `</think>`, sample the solution, then train; the forced tokens aren't trained.
  - `--early-stop-drop 0.10`: stops the run cleanly when held-out `single/pass@1` falls more than 10 points below step 0. It writes `early_stop.json`, and a resume won't continue past it.
    - Every eval goes to `evals.jsonl`, and `--eval-final` evaluates the last checkpoint too.
    - Use `--save-every` equal to `--eval-every`, so each evaluated step has a checkpoint.
  - `--reverse`: the cookbook's analytical reverse KL over the teacher's top-K (mode-seeking). Its training is probably billed about 2×, because the client-side custom loss runs a forward pass and then a forward/backward.
  - `--length-prior runs/train/<old>/student_lengths.jsonl`: the dry-run cost estimate uses the lengths from that run's first steps.
- **On Modal** (`scripts/train_distill_modal.py`): same pattern and flags as the RL launcher, with the `tinker-api` secret and the `cpp-posttrain-runs` volume.
  - `--check` runs the dry run inside the container, which is free.
  - `modal run --detach ... --name X ...` starts it; `--pull X` copies results back.
  - The launcher refuses to delete or resume a folder that isn't a distillation run.

## Agentic RL (`posttrain/rl_env.py`, `scripts/train_rl.py`, `scripts/train_rl_modal.py`)

RL in the competition environment, using the cookbook's `rl/train.py` loop. Each step runs 8 tasks with 4 sessions each. A session uses the `run_cpp` and `submit` tools, and its last compiling submission is graded on the hidden tests. Tasks come from the `rl` part of the pool, rated 1400–2200 (639 problems).

```bash
uv run python scripts/train_rl.py --name rl-v1 --dry-run                                     # config, pool, worst-case cost
uv run python scripts/train_rl.py --name rl-v1 --load-checkpoint tinker://.../weights/final   # local
uv run python scripts/train_rl.py --summarize runs/train/rl-v1                                # per-step tables
```

- **Reward:** `score/100 - 0.01 × turns` when the score is above 0, and 0 otherwise; then `- 0.05` per malformed turn.
  - It's computed after each group finishes, so a session that was cut off or ended on malformed calls is still graded on its last compiling submission.
  - Groups where every session got the same reward are dropped, since they carry no signal.
- **Defaults:**
  - 16K tokens per turn, 8 turns, 64K context, LoRA rank 32, lr 4e-5.
  - Async: sampling runs at most 1 step ahead of training.
  - A full checkpoint every 5 steps, and a state-only resume point every step.
- **`--load-checkpoint`** needs a *state* path (`…/weights/…`), not `sampler_weights`.
- **Logs:**
  - `rollouts.jsonl` has one line per session, including groups dropped for constant reward.
  - `metrics.jsonl` has the cookbook's metrics: `env/all/rl/*`, `time/*`, `tokens/*`, and `cost/step_usd` at Tinker's published rates.

**On Modal** (the laptop can close). The training loop runs in a CPU container, and the run folder lives on the Volume `cpp-posttrain-runs`.

```bash
uv run modal secret create tinker-api TINKER_API_KEY=...     # once
uv run modal run --detach scripts/train_rl_modal.py --name rl-v1 --load-checkpoint tinker://.../weights/final
uv run modal app logs <app id>                                # watch (the app id is printed at start; or `modal app list`)
uv run modal app stop -y <app id>                             # stop (resume later with --if-exists resume)
uv run modal run scripts/train_rl_modal.py --pull rl-v1       # copy to runs/train/rl-v1/
```

If the container is preempted, crashes or hits the 24h limit, it's retried up to 4 times. Each retry resumes from the newest checkpoint.

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

## Grader (`cppgrade/grader.py`)

`await grade(pool, model_text, tests)` returns a verdict plus subtype, using the failure taxonomy from the diagnosis plan:

| verdict | subtype |
|---|---|
| `format_error` | no fenced C++ block found |
| `compile_error` | `missing_include`, `undeclared`, `type_error`, `syntax`, `other`, `timeout` |
| `runtime_error` | `segfault`, `abort`, `asan_oob`, `asan_other`, `ubsan`, `nonzero_exit` |
| `time_limit` | |
| `wrong_answer` | |
| `accepted` | |

- Compiler is GCC 12 (Debian bookworm), `-std=c++17 -O2`. `bits/stdc++.h` works.
- `sanitize=True` builds with ASan and UBSan, so silent out-of-bounds bugs show up as `asan_oob` instead of a wrong answer. It's slower, so use it for diagnosis, not for the RL reward.
- Output check compares whitespace-separated tokens, with relative float tolerance `float_tol`.
- Each grade uses a fresh sandbox from `ModalSandboxPool` (from `tinker_cookbook`). Set the pool size with `MODAL_POOL_SIZE`.

## Reference

- RL recipe to adapt: `tinker_cookbook/recipes/code_rl/` (DeepCoder-style, Python-graded; swap in `cppgrade.grade` for the reward).
- To list which models Tinker serves now, run `check_tinker.py --list-only`. Models in the cookbook that meet the ≤9B-active rule include Qwen3-8B, Qwen3.5-9B, Qwen3-30B-A3B, Qwen3.5/3.6-35B-A3B, gpt-oss-20b (3.6B active), and Nemotron-3 Nano / 3.5 Lightning 30B-A3B.

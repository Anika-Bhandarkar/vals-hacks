# cppgrade

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

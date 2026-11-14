"""C++ grader on Modal sandboxes, with a failure taxonomy for diagnosis and RL rewards.

Verdicts (first failure wins, in this order):
    format_error   -> no ```cpp block could be extracted from the model output
    compile_error  -> subtype: missing_include | undeclared | type_error | syntax | other
    runtime_error  -> subtype: segfault | abort | asan_oob | asan_other | ubsan | nonzero_exit
    time_limit
    wrong_answer
    accepted

Usage:
    pool = make_pool(pool_size=16)
    res = await grade(pool, model_text, tests=[{"input": "1 2\\n", "output": "3\\n"}])
    print(res.verdict, res.subtype, res.passed, "/", res.total)
    await pool.terminate()
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import modal
from tinker_cookbook.sandbox.modal_sandbox import ModalSandboxPool

APP_NAME = "cpp-posttrain-grader"
RUNNER_SRC = (Path(__file__).parent / "runner.py").read_text()

# Debian bookworm -> GCC 12. bits/stdc++.h available, same as most judges.
CPP_IMAGE = modal.Image.debian_slim(python_version="3.12").apt_install("g++")

BASE_FLAGS = ["-std=c++17", "-O2", "-pipe"]
SANITIZE_FLAGS = ["-std=c++17", "-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]


def make_pool(pool_size: int = 16, sandbox_timeout_secs: int = 600) -> ModalSandboxPool:
    """Must be called inside a running event loop. Each sandbox is used once then recycled."""
    return ModalSandboxPool(
        pool_size=pool_size,
        sandbox_timeout_secs=sandbox_timeout_secs,
        image=CPP_IMAGE,
        app_name=APP_NAME,
    )


# ---------------------------------------------------------------- extraction

_FENCE = re.compile(r"```[ \t]*(c\+\+|cpp|cc|cxx|C\+\+|CPP)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_cpp(text: str) -> str | None:
    """Return the last fenced C++ block (models often draft then finalize). Falls back to an
    untagged fence only if it looks like C++."""
    blocks = _FENCE.findall(text)
    tagged = [code for tag, code in blocks if tag]
    if tagged:
        return tagged[-1]
    for _, code in reversed(blocks):
        if "#include" in code or "int main" in code:
            return code
    return None


# ---------------------------------------------------------------- classification

_COMPILE_RULES: list[tuple[str, re.Pattern[str]]] = [
    # g++ prints this note when a std symbol is used without its header.
    ("missing_include", re.compile(r"probably fixable by adding '#include|did you forget to '#include")),
    ("undeclared", re.compile(r"was not declared in this scope|has not been declared|undefined reference to|"
                              r"is not a member of|does not name a type|use of undeclared")),
    ("type_error", re.compile(r"no matching function|cannot convert|invalid conversion|no match for|"
                              r"invalid operands|cannot bind|conversion from .* to non-scalar|"
                              r"template argument deduction/substitution failed|incomplete type")),
    ("syntax", re.compile(r"expected .* before|expected primary-expression|expected unqualified-id|"
                          r"stray '|missing terminating")),
]


def classify_compile(stderr: str) -> str:
    # GCC uses curly quotes (‘ ’) under a UTF-8 locale; normalize so the rules match either way.
    stderr = stderr.replace("‘", "'").replace("’", "'")
    # Use the FIRST error line: later errors are usually cascades.
    first = next((ln for ln in stderr.splitlines() if "error:" in ln), stderr)
    window = stderr[: stderr.find(first) + len(first) + 600] if first in stderr else stderr
    for name, pat in _COMPILE_RULES:
        if pat.search(window):
            return name
    return "other"


def classify_runtime(rc: int, stderr: str) -> str:
    if "AddressSanitizer" in stderr:
        if re.search(r"(heap|stack|global)-buffer-overflow|out of bounds", stderr):
            return "asan_oob"
        return "asan_other"
    if "runtime error:" in stderr:  # UBSan
        return "ubsan"
    if rc in (-11, 139):
        return "segfault"
    if rc in (-6, 134):
        return "abort"
    return "nonzero_exit"


# ---------------------------------------------------------------- grading


@dataclass
class GradeResult:
    verdict: str
    subtype: str | None = None
    passed: int = 0
    total: int = 0
    detail: str = ""
    code: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.verdict == "accepted"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def grade(
    pool: ModalSandboxPool,
    model_output: str,
    tests: list[dict[str, str]],
    *,
    code: str | None = None,
    time_limit: float = 2.0,
    memory_mb: int = 1024,
    sanitize: bool = False,
    float_tol: float = 1e-6,
    stop_on_first_failure: bool = True,
    compile_timeout: int = 60,
) -> GradeResult:
    """Grade one completion. Pass `code` to skip extraction (e.g. reference solutions).

    sanitize=True compiles with ASan/UBSan so out-of-bounds shows up as `asan_oob` instead of
    a silent wrong answer. Slower (~2-3x); use it for diagnosis, not RL reward.
    """
    if code is None:
        code = extract_cpp(model_output)
    if code is None:
        return GradeResult("format_error", total=len(tests))

    cfg = {
        "compile_flags": SANITIZE_FLAGS if sanitize else BASE_FLAGS,
        "compile_timeout": compile_timeout,
        "time_limit": time_limit * (3 if sanitize else 1),
        "memory_mb": memory_mb,
        "sanitize": sanitize,
        "float_tol": float_tol,
        "stop_on_first_failure": stop_on_first_failure,
    }
    files = {
        "sol.cpp": code,
        "tests.json": json.dumps(tests),
        "config.json": json.dumps(cfg),
        "runner.py": RUNNER_SRC,
    }
    budget = compile_timeout + int(cfg["time_limit"] * len(tests)) + 30
    res = await pool.run_in_workdir(files, ["python3", "runner.py", "config.json"], timeout=budget)

    try:
        out = json.loads(res.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return GradeResult("infra_error", total=len(tests), code=code,
                           detail=f"exit={res.exit_code} stderr={res.stderr[-1000:]}")

    if not out["compiled"]:
        err = out["compile_stderr"]
        sub = "timeout" if err == "COMPILE_TIMEOUT" else classify_compile(err)
        return GradeResult("compile_error", sub, total=len(tests), code=code, detail=err[:3000])

    results = out["results"]
    passed = sum(r["status"] == "ok" for r in results)
    fail = next((r for r in results if r["status"] != "ok"), None)
    if fail is None:
        return GradeResult("accepted", passed=passed, total=len(tests), code=code)

    if fail["status"] == "runtime_error":
        return GradeResult("runtime_error", classify_runtime(fail["rc"], fail["stderr"]), passed,
                           len(tests), detail=fail["stderr"][-1500:], code=code, extra={"test": fail["idx"]})
    if fail["status"] == "time_limit":
        return GradeResult("time_limit", None, passed, len(tests), code=code, extra={"test": fail["idx"]})
    return GradeResult("wrong_answer", None, passed, len(tests), code=code,
                       detail=f"got={fail['got']!r}\nwant={fail['want']!r}", extra={"test": fail["idx"]})

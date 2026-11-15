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

import asyncio
import contextlib
import json
import logging
import re
import shlex
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import modal
from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox, ModalSandboxPool
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

from cppgrade.infra import APP_NAME, CPP_IMAGE, SANDBOX_CPU, SANDBOX_MEMORY_MB, TESTS_MOUNT, TESTS_VOLUME

logger = logging.getLogger(__name__)
RUNNER_SRC = (Path(__file__).parent / "runner.py").read_text()

BASE_FLAGS = ["-std=c++17", "-O2", "-pipe"]
SANITIZE_FLAGS = ["-std=c++17", "-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]

_app: modal.App | None = None


async def _create_sandbox(timeout: int) -> ModalSandbox:
    global _app
    if _app is None:  # one lookup per process (avoids Modal's app-create rate limit)
        _app = await modal.App.lookup.aio(APP_NAME, create_if_missing=True)
    sb = await modal.Sandbox.create.aio(
        app=_app,
        image=CPP_IMAGE,
        timeout=timeout,
        cpu=SANDBOX_CPU,
        memory=SANDBOX_MEMORY_MB,
        volumes={TESTS_MOUNT: TESTS_VOLUME},
    )
    return ModalSandbox(timeout=timeout, image=CPP_IMAGE, app=_app, sandbox=sb)


class _ResourcedPool(ModalSandboxPool):
    """Cookbook warm pool, but with fixed CPU/memory and the tests volume mounted."""

    async def _create(self) -> ModalSandbox:
        return await _create_sandbox(self._sandbox_timeout_secs)


class OnDemandPool:
    """Creates one sandbox per call, nothing idles. Best for evals, where sampling
    dominates wall-clock and a warm pool would sit billing (and its sandboxes expire)."""

    def __init__(self, max_concurrent: int = 32, creation_rate: float = 4.0):
        self._sem = asyncio.Semaphore(max_concurrent)
        self._interval = 1.0 / creation_rate
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self._interval

    async def run_in_workdir(self, files: dict[str, str], command: list[str], timeout: int | None = None):
        timeout = timeout or 600
        async with self._sem:
            await self._throttle()
            sb = await _create_sandbox(timeout + 300)  # headroom for file writes
            try:
                workdir = "/workspace/run"
                await sb.run_command(f"mkdir -p {workdir}")
                writes = await asyncio.gather(*(sb.write_file(f"{workdir}/{n}", c) for n, c in files.items()))
                bad = next((w for w in writes if w.exit_code != 0), None)
                if bad is not None:
                    return SandboxResult(stdout="", stderr=f"write failed: {bad.stderr}", exit_code=-1)
                return await sb.run_command(shlex.join(command), workdir=workdir, timeout=timeout)
            finally:
                with contextlib.suppress(Exception):
                    await sb.cleanup()

    async def terminate(self) -> None:
        pass


def make_pool(pool_size: int = 16, sandbox_timeout_secs: int = 600) -> ModalSandboxPool:
    """Warm pool (low latency, for RL). Must be called inside a running event loop."""
    return _ResourcedPool(pool_size=pool_size, sandbox_timeout_secs=sandbox_timeout_secs, app_name=APP_NAME)


def make_ondemand_pool(max_concurrent: int = 32) -> OnDemandPool:
    return OnDemandPool(max_concurrent=max_concurrent)


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


_PY_FENCE = re.compile(r"```[ \t]*(python|py|python3)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python(text: str) -> str | None:
    """Last ```python block; falls back to an untagged fence that isn't obviously C++."""
    blocks = _PY_FENCE.findall(text)
    tagged = [code for tag, code in blocks if tag]
    if tagged:
        return tagged[-1]
    for _, code in reversed(blocks):
        if "#include" not in code:
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
                              r"template argument deduction/substitution failed|incomplete type|"
                              r"inconsistent types .* deduced|cannot be used as a function")),
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


def classify_runtime(rc: int, stderr: str, lang: str = "cpp") -> str:
    if lang == "python":
        # Last "SomeError: msg" line of the traceback -> "IndexError", "RecursionError", ...
        m = re.findall(r"^(\w+(?:Error|Exception|Exit|Interrupt))\b", stderr, re.MULTILINE)
        if m:
            return m[-1]
        if stderr == "OUTPUT_LIMIT_EXCEEDED":
            return "output_limit"
        return "killed" if rc < 0 else "nonzero_exit"
    if stderr == "OUTPUT_LIMIT_EXCEEDED":
        return "output_limit"
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


def _flags(sanitize: bool, cpp_std: str) -> list[str]:
    flags = SANITIZE_FLAGS if sanitize else BASE_FLAGS
    return [f"-std={cpp_std}" if f.startswith("-std=") else f for f in flags]


async def _execute(
    pool: ModalSandboxPool | OnDemandPool,
    code: str,
    cfg: dict[str, Any],
    *,
    lang: str = "cpp",
    tests: list[dict[str, str]] | None = None,
    extra_files: dict[str, str] | None = None,
    budget: int,
    retries: int = 2,
) -> tuple[dict[str, Any] | None, str]:
    """Run runner.py in a fresh sandbox; retry on a new sandbox if Modal hiccups.
    Returns (runner JSON, "") or (None, error description)."""
    files = {
        ("sol.cpp" if lang == "cpp" else "sol.py"): code,
        "config.json": json.dumps(cfg),
        "runner.py": RUNNER_SRC,
        **(extra_files or {}),
    }
    if tests is not None:
        files["tests.json"] = json.dumps(tests)
    last_err = ""
    for attempt in range(retries + 1):
        res = None
        try:
            res = await pool.run_in_workdir(files, ["python3", "runner.py", "config.json"], timeout=budget)
            return json.loads(res.stdout.strip().splitlines()[-1]), ""
        except (IndexError, json.JSONDecodeError):
            last_err = f"exit={res.exit_code} stderr={res.stderr[-1000:]}" if res else "no result"
        except Exception as e:  # sandbox died / Modal hiccup: retry on a fresh sandbox
            last_err = f"{type(e).__name__}: {e}"
        logger.warning("sandbox attempt %d failed: %s", attempt + 1, last_err[:200])
    return None, last_err


async def grade(
    pool: ModalSandboxPool | OnDemandPool,
    model_output: str,
    tests: list[dict[str, str]] | None = None,
    *,
    tests_file: str | None = None,
    n_tests: int | None = None,
    retries: int = 2,
    code: str | None = None,
    time_limit: float = 2.0,
    memory_mb: int = 1024,
    sanitize: bool = False,
    float_tol: float = 1e-6,
    stop_on_first_failure: bool = True,
    compile_timeout: int = 60,
    lang: str = "cpp",
    cpp_std: str = "c++17",
) -> GradeResult:
    """Grade one completion. Pass `code` to skip extraction (e.g. reference solutions).

    lang="python" runs the same tests with python3 (SyntaxError -> compile_error/syntax,
    runtime subtype = exception name), so C++ and Python results are directly comparable.

    sanitize=True compiles with ASan/UBSan so out-of-bounds shows up as `asan_oob` instead of
    a silent wrong answer. Slower (~2-3x); use it for diagnosis, not RL reward. C++ only.

    Tests come either inline (`tests`, uploaded with each call) or from the tests volume
    (`tests_file` = path relative to the volume root, gzipped JSON; pass `n_tests` too).
    Use the volume for big test sets so they aren't re-uploaded per sample.
    """
    if lang not in ("cpp", "python"):
        raise ValueError(f"lang must be 'cpp' or 'python', got {lang!r}")
    if (tests is None) == (tests_file is None):
        raise ValueError("pass exactly one of tests / tests_file")
    total = len(tests) if tests is not None else (n_tests or 0)
    sanitize = sanitize and lang == "cpp"
    if code is None:
        code = extract_cpp(model_output) if lang == "cpp" else extract_python(model_output)
    if code is None:
        return GradeResult("format_error", total=total)

    cfg = {
        "lang": lang,
        "compile_flags": _flags(sanitize, cpp_std),
        "compile_timeout": compile_timeout,
        "time_limit": time_limit * (3 if sanitize else 1),
        "memory_mb": memory_mb,
        "sanitize": sanitize,
        "float_tol": float_tol,
        "stop_on_first_failure": stop_on_first_failure,
        "tests_file": f"{TESTS_MOUNT}/{tests_file}" if tests_file else None,
    }
    budget = compile_timeout + int(cfg["time_limit"] * max(total, 1)) + 60
    out, last_err = await _execute(pool, code, cfg, lang=lang, tests=tests, budget=budget, retries=retries)
    if out is None:
        return GradeResult("infra_error", total=total, code=code, detail=last_err)

    total = out.get("n_tests", total)
    if not out["compiled"]:
        err = out["compile_stderr"]
        if err == "COMPILE_TIMEOUT":
            sub = "timeout"
        else:
            sub = classify_compile(err) if lang == "cpp" else "syntax"
        return GradeResult("compile_error", sub, total=total, code=code, detail=err[:3000])

    results = out["results"]
    passed = sum(r["status"] == "ok" for r in results)
    fail = next((r for r in results if r["status"] != "ok"), None)
    if fail is None:
        return GradeResult("accepted", passed=passed, total=total, code=code)

    if fail["status"] == "runtime_error":
        return GradeResult("runtime_error", classify_runtime(fail["rc"], fail["stderr"], lang), passed,
                           total, detail=fail["stderr"][-1500:], code=code, extra={"test": fail["idx"]})
    if fail["status"] == "time_limit":
        return GradeResult("time_limit", None, passed, total, code=code, extra={"test": fail["idx"]})
    return GradeResult("wrong_answer", None, passed, total, code=code,
                       detail=f"got={fail['got']!r}\nwant={fail['want']!r}", extra={"test": fail["idx"]})


# ---------------------------------------------------------------- agentic setting (run / samples / score)

AGENT_STD = "c++20"  # the competition compiles C++20


def _base_cfg(mode: str, time_limit: float, memory_mb: int, cpp_std: str, **kw) -> dict[str, Any]:
    return {
        "mode": mode, "lang": "cpp", "compile_flags": _flags(False, cpp_std), "compile_timeout": 60,
        "time_limit": time_limit, "memory_mb": memory_mb, "sanitize": False, "float_tol": 1e-6,
        "stop_on_first_failure": False, "tests_file": None, **kw,
    }


async def run_program(
    pool: ModalSandboxPool | OnDemandPool,
    code: str,
    stdin: str,
    *,
    time_limit: float = 2.0,
    memory_mb: int = 1024,
    cpp_std: str = AGENT_STD,
    output_chars: int = 4000,
) -> dict[str, Any]:
    """Compile and run once on `stdin`. Returns the runner's JSON: compiled, compile_stderr,
    stdout, stderr, exit_code, timed_out, time_s, output_limit. On sandbox failure: {"infra_error": ...}."""
    cfg = _base_cfg("run", time_limit, memory_mb, cpp_std, run_output_chars=output_chars)
    out, err = await _execute(pool, code, cfg, extra_files={"run_input.txt": stdin}, budget=int(time_limit) + 90)
    return out if out is not None else {"infra_error": err}


def _verdict(rec: dict[str, Any]) -> str:
    return {"ok": "accepted"}.get(rec["status"], rec["status"])


async def check_samples(
    pool: ModalSandboxPool | OnDemandPool,
    code: str,
    samples: list[dict[str, str]],
    *,
    time_limit: float = 2.0,
    memory_mb: int = 1024,
    cpp_std: str = AGENT_STD,
) -> dict[str, Any]:
    """Compile and run every sample test (no early stop). Returns
    {"compiled", "compile_error", "results": [{"sample", "verdict", "got"?, "want"?}]} or {"infra_error"}."""
    cfg = _base_cfg("tests", time_limit, memory_mb, cpp_std)
    budget = 60 + int(time_limit * max(1, len(samples))) + 60
    out, err = await _execute(pool, code, cfg, tests=samples, budget=budget)
    if out is None:
        return {"infra_error": err}
    if not out["compiled"]:
        return {"compiled": False, "compile_error": out["compile_stderr"][:3000], "results": []}
    results = []
    for r in out["results"]:
        item = {"sample": r["idx"] + 1, "verdict": _verdict(r)}
        if r["status"] == "wrong_answer":
            item.update(got=r["got"], want=r["want"])
        elif r["status"] == "runtime_error":
            item.update(stderr=r.get("stderr", "")[-600:])
        results.append(item)
    return {"compiled": True, "compile_error": "", "results": results}


@dataclass
class ScoreResult:
    """Hidden-test score for one submission, in the competition's 40 (small) / 60 (large) scheme."""

    compiled: bool
    score: float = 0.0               # all-or-nothing per group: 40 if every small test passes, +60 if every large one does
    score_proportional: float = 0.0  # 40 * frac_small_passed + 60 * frac_large_passed
    small_passed: int = 0
    small_total: int = 0
    large_passed: int = 0
    large_total: int = 0
    statuses: dict[str, int] = field(default_factory=dict)
    compile_error: str = ""
    infra_error: str = ""
    per_test: list[tuple[int, str, str]] = field(default_factory=list)  # (input bytes, group, verdict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_small_large(
    results: list[dict[str, Any]], small_frac: float = 0.4, size_cut: float = 0.01
) -> tuple[list[int], list[int]]:
    """Proxy for the competition's small/large hidden tests, which our datasets don't label.

    Primary rule: "small" = input at most `size_cut` (1%) of this problem's largest input. That
    matches "brute force fits" well when sizes spread over orders of magnitude (e.g. abc365_e:
    5 tiny tests vs. 165KB-1.8MB ones, where a 400KB test already times out an O(N^2) brute).
    Fallback when that rule puts <15% or >85% of tests in "small" (sizes barely vary, e.g.
    single-number inputs): the smallest `small_frac` of tests by input size, by rank."""
    n = len(results)
    if n < 2:
        return list(range(n)), []
    mx = max(r["in_bytes"] for r in results)
    small = [i for i, r in enumerate(results) if r["in_bytes"] <= size_cut * mx]
    if not 0.15 * n <= len(small) <= 0.85 * n:
        order = sorted(range(n), key=lambda i: (results[i]["in_bytes"], i))
        small = order[: max(1, round(small_frac * n))]
    small_set = set(small)
    return sorted(small_set), [i for i in range(n) if i not in small_set]


async def score_submission(
    pool: ModalSandboxPool | OnDemandPool,
    code: str,
    *,
    tests_file: str,
    n_tests: int,
    time_limit: float = 2.0,
    memory_mb: int = 1024,
    cpp_std: str = AGENT_STD,
    small_frac: float = 0.4,
) -> ScoreResult:
    """Run ALL hidden tests (no early stop, partial credit needs every verdict) and score 0-100."""
    cfg = _base_cfg("tests", time_limit, memory_mb, cpp_std, tests_file=f"{TESTS_MOUNT}/{tests_file}")
    budget = 60 + int(time_limit * max(1, n_tests)) + 90
    out, err = await _execute(pool, code, cfg, budget=budget)
    if out is None:
        return ScoreResult(compiled=False, infra_error=err)
    if not out["compiled"]:
        return ScoreResult(compiled=False, compile_error=out["compile_stderr"][:3000])
    results = out["results"]
    small, large = split_small_large(results, small_frac)
    ok = lambda idxs: sum(results[i]["status"] == "ok" for i in idxs)
    sp, lp = ok(small), ok(large)
    if not large:  # every test counted as small: it carries the full 100
        score = 100.0 * (sp == len(small))
        prop = 100.0 * sp / max(1, len(small))
    else:
        score = 40.0 * (sp == len(small)) + 60.0 * (lp == len(large))
        prop = 40.0 * sp / len(small) + 60.0 * lp / len(large)
    statuses: dict[str, int] = {}
    for r in results:
        statuses[_verdict(r)] = statuses.get(_verdict(r), 0) + 1
    small_set = set(small)
    per_test = [(r["in_bytes"], "small" if i in small_set else "large", _verdict(r)) for i, r in enumerate(results)]
    return ScoreResult(True, score, round(prop, 2), sp, len(small), lp, len(large), statuses, per_test=per_test)

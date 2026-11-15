"""Runs INSIDE the Modal sandbox. Compiles the solution, then either:

  mode "tests" (default): runs it on every test, compares output, prints one JSON line of verdicts
  mode "run":             runs it once on run_input.txt, prints stdout/stderr/exit code/time

Output comparison happens here (not on your laptop) so only small verdicts cross the wire; the
sandbox stdout stream is capped at ~128KB.

The program's stdout/stderr go to size-capped files (RLIMIT_FSIZE), never into this process's
memory: a wrong solution that prints in a loop until the time limit used to OOM the sandbox.

Stdlib only, so the sandbox image stays tiny.
Usage: python3 runner.py config.json
"""

import gzip
import json
import os
import resource
import signal
import subprocess
import sys
import time

SNIP = 400  # chars of stdout/stderr kept per failing test, for debugging
OUTPUT_LIMIT_RCS = (-signal.SIGXFSZ, 128 + signal.SIGXFSZ)


def _preexec(memory_mb: int | None, out_cap: int):
    def fn():
        if memory_mb:
            b = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (b, b))
        resource.setrlimit(resource.RLIMIT_FSIZE, (out_cap, out_cap))

    return fn


def _run_once(cmd, stdin_path, time_limit, memory_mb, out_cap):
    """Returns (rc or None if timed out, elapsed, stdout_text, stderr_text)."""
    with open(stdin_path, "rb") as fin, open("out.txt", "wb") as fout, open("err.txt", "wb") as ferr:
        start = time.time()
        p = subprocess.Popen(cmd, stdin=fin, stdout=fout, stderr=ferr, preexec_fn=_preexec(memory_mb, out_cap))
        try:
            rc = p.wait(timeout=time_limit)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            rc = None
        elapsed = time.time() - start
    with open("out.txt", "rb") as f:
        out = f.read(out_cap).decode("utf-8", "replace")
    with open("err.txt", "rb") as f:
        f.seek(max(0, os.path.getsize("err.txt") - 16 * SNIP))
        err = f.read().decode("utf-8", "replace")
    return rc, elapsed, out, err


def _tokens_match(got: str, want: str, float_tol: float) -> bool:
    g, w = got.split(), want.split()
    if len(g) != len(w):
        return False
    for a, b in zip(g, w):
        if a == b:
            continue
        if float_tol <= 0:
            return False
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if abs(fa - fb) > float_tol * max(1.0, abs(fb)):
            return False
    return True


def _load_tests(cfg):
    if cfg.get("tests_file"):
        with gzip.open(cfg["tests_file"], "rt") as f:
            return json.load(f)
    return json.load(open("tests.json"))


def main():
    cfg = json.load(open(sys.argv[1]))
    mode = cfg.get("mode", "tests")
    tests = _load_tests(cfg) if mode == "tests" else []
    out_cap = cfg.get("output_limit_bytes", 64 * 1024 * 1024)
    # ASan reserves huge virtual ranges, so RLIMIT_AS would kill it spuriously.
    mem = None if cfg["sanitize"] else cfg["memory_mb"]

    lang = cfg.get("lang", "cpp")
    if lang == "python":
        # "Compile" = byte-compile, so SyntaxErrors land in the same bucket as C++ compile errors.
        build = ["python3", "-m", "py_compile", "sol.py"]
        run_cmd = ["python3", "sol.py"]
    else:
        build = ["g++", *cfg["compile_flags"], "-o", "sol", "sol.cpp"]
        run_cmd = ["./sol"]

    t0 = time.time()
    try:
        c = subprocess.run(build, capture_output=True, text=True, timeout=cfg["compile_timeout"])
    except subprocess.TimeoutExpired:
        print(json.dumps({"compiled": False, "n_tests": len(tests), "compile_stderr": "COMPILE_TIMEOUT", "compile_s": time.time() - t0}))
        return
    compile_s = time.time() - t0
    if c.returncode != 0:
        print(json.dumps({"compiled": False, "n_tests": len(tests), "compile_stderr": c.stderr[-8000:], "compile_s": compile_s}))
        return

    if mode == "run":
        cap = cfg.get("run_output_chars", 4000)
        rc, elapsed, out, err = _run_once(run_cmd, "run_input.txt", cfg["time_limit"], mem, out_cap)
        print(json.dumps({
            "compiled": True, "compile_stderr": c.stderr[-2000:], "compile_s": compile_s,
            "timed_out": rc is None, "exit_code": rc, "time_s": round(elapsed, 3),
            "output_limit": rc in OUTPUT_LIMIT_RCS,
            "stdout": out[:cap], "stdout_truncated": len(out) > cap,
            "stderr": err[-cap // 2:],
        }))
        return

    results = []
    for i, t in enumerate(tests):
        with open("in.txt", "w") as f:
            f.write(t["input"])
        rc, elapsed, out, err = _run_once(run_cmd, "in.txt", cfg["time_limit"], mem, out_cap)
        rec = {"idx": i, "in_bytes": len(t["input"])}
        if rc is None:
            rec.update(status="time_limit")
        elif rc in OUTPUT_LIMIT_RCS:
            rec.update(status="runtime_error", rc=rc, stderr="OUTPUT_LIMIT_EXCEEDED")
        elif rc != 0:
            rec.update(status="runtime_error", rc=rc, stderr=err[-SNIP * 4 :])
        elif _tokens_match(out, t["output"], cfg["float_tol"]):
            rec.update(status="ok")
        else:
            rec.update(status="wrong_answer", got=out[:SNIP], want=t["output"][:SNIP])
        rec["elapsed"] = round(elapsed, 3)
        if rec["status"] != "ok" and cfg.get("report_input_chars"):
            rec["input_head"] = t["input"][: cfg["report_input_chars"]]
            rec["want_head"] = t["output"][: cfg["report_input_chars"]]
        results.append(rec)
        if cfg["stop_on_first_failure"] and rec["status"] != "ok":
            break

    print(json.dumps({"compiled": True, "n_tests": len(tests), "compile_s": compile_s, "results": results}))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[1])))
    main()

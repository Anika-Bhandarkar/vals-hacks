"""Runs INSIDE the Modal sandbox. Compiles sol.cpp, runs it on tests.json, prints one JSON line.

Output comparison happens here (not on your laptop) so only small verdicts cross
the wire; the sandbox stdout stream is capped at ~128KB.

Stdlib only, so the sandbox image stays tiny.
Usage: python3 runner.py config.json
"""

import json
import os
import resource
import subprocess
import sys
import time

SNIP = 400  # chars of stdout/stderr kept per failing test, for debugging


def _limit_memory(mb: int):
    def fn():
        b = mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (b, b))

    return fn


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


def main():
    cfg = json.load(open(sys.argv[1]))
    tests = json.load(open("tests.json"))

    t0 = time.time()
    try:
        c = subprocess.run(
            ["g++", *cfg["compile_flags"], "-o", "sol", "sol.cpp"],
            capture_output=True,
            text=True,
            timeout=cfg["compile_timeout"],
        )
    except subprocess.TimeoutExpired:
        print(json.dumps({"compiled": False, "compile_stderr": "COMPILE_TIMEOUT", "compile_s": time.time() - t0}))
        return
    compile_s = time.time() - t0
    if c.returncode != 0:
        print(json.dumps({"compiled": False, "compile_stderr": c.stderr[-8000:], "compile_s": compile_s}))
        return

    results = []
    for i, t in enumerate(tests):
        start = time.time()
        rec = {"idx": i}
        try:
            p = subprocess.run(
                ["./sol"],
                input=t["input"].encode(),
                capture_output=True,
                timeout=cfg["time_limit"],
                # ASan reserves huge virtual ranges, so RLIMIT_AS would kill it spuriously.
                preexec_fn=None if cfg["sanitize"] else _limit_memory(cfg["memory_mb"]),
            )
            out = p.stdout.decode("utf-8", "replace")
            err = p.stderr.decode("utf-8", "replace")
            if p.returncode != 0:
                rec.update(status="runtime_error", rc=p.returncode, stderr=err[-SNIP * 4 :])
            elif _tokens_match(out, t["output"], cfg["float_tol"]):
                rec.update(status="ok")
            else:
                rec.update(status="wrong_answer", got=out[:SNIP], want=t["output"][:SNIP])
        except subprocess.TimeoutExpired:
            rec.update(status="time_limit")
        rec["elapsed"] = round(time.time() - start, 3)
        results.append(rec)
        if cfg["stop_on_first_failure"] and rec["status"] != "ok":
            break

    print(json.dumps({"compiled": True, "compile_s": compile_s, "results": results}))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[1])))
    main()

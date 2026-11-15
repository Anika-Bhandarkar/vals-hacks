"""open-r1/codeforces problems (with editorials and tags) on the Modal tests volume, for Test A.

One-time prep, runs on Modal (downloads from HF there):
    uv run python scripts/prepare_codeforces.py                         # test split -> Test A (61 problems)
    uv run python scripts/prepare_codeforces.py --split train --max-problems 2000 --no-require-editorial
                                                                        # train split -> SFT/RL problem pool

Splits of the `verifiable` config: test = 422 problems from Aug 2024 - Feb 2025 contests (kept for
Test A, never trained on); train = 8,338 older problems (training pool).
Selection: stdio input, no custom checker (our grader only does token/float comparison),
executable, rating in [min_rating, max_rating], optionally a non-empty editorial; newest first.

Hidden tests = official tests + the dataset's generated tests (generated_tests/test_cases_<contest>.parquet),
deduplicated by input. Written as gzipped JSON like the LiveCodeBench prep, so the same grader works.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import modal

from cppgrade.infra import TESTS_MOUNT, TESTS_VOLUME

HF_REPO = "open-r1/codeforces"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "data" / "codeforces"


def vol_dir(split: str) -> str:
    return f"codeforces/verifiable_{split}"

prep_app = modal.App("cpp-posttrain-codeforces-prep")
prep_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub", "pyarrow")
    .add_local_python_source("cppgrade")
)


def _safe_id(pid: str) -> str:
    return pid.replace("/", "_")


# Prep runs in three steps so a big pool doesn't hit one function's timeout, and a re-run resumes:
#   1. _select_remote:  read problem metadata, pick problems, stage them per contest on the volume
#   2. _contest_remote: one call per contest, in parallel: download its generated tests, write the
#                       per-problem test files, record a done-marker (skipped on re-run)
#   3. _finalize_remote: write index.json from the done-markers of the picked contests
PREP_PARALLEL = 16  # concurrent contest workers (each downloads one contest's test parquet from HF)


def _staging(split: str) -> str:
    return f"{TESTS_MOUNT}/{vol_dir(split)}/_staging"


@prep_app.function(image=prep_image, volumes={TESTS_MOUNT: TESTS_VOLUME}, timeout=3600, cpu=2, memory=16384)
def _select_remote(split: str, min_rating: int, max_rating: int, require_editorial: bool,
                   max_problems: int | None) -> list[str]:
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    files = set(list_repo_files(HF_REPO, repo_type="dataset"))
    meta_files = sorted(f for f in files if f.startswith(f"verifiable/{split}-") and f.endswith(".parquet"))
    cols = ["id", "contest_id", "contest_start", "rating", "tags", "title", "description", "input_format",
            "output_format", "note", "examples", "editorial", "time_limit", "memory_limit", "official_tests",
            "input_mode", "generated_checker", "executable"]
    rows = []
    for mf in meta_files:
        for r in pq.read_table(hf_hub_download(HF_REPO, mf, repo_type="dataset"), columns=cols).to_pylist():
            if (r["input_mode"] == "stdio" and not r["generated_checker"] and r["executable"]
                    and r["rating"] and min_rating <= r["rating"] <= max_rating
                    and (not require_editorial or (r["editorial"] or "").strip())):
                rows.append(r)
    rows.sort(key=lambda r: (-(r["contest_start"] or 0), r["id"]))  # newest first
    picked = rows[:max_problems] if max_problems else rows
    by_contest: dict[str, list[dict]] = {}
    for r in picked:
        by_contest.setdefault(str(r["contest_id"]), []).append(r)

    stage = _staging(split)
    os.makedirs(stage, exist_ok=True)
    has_gen = {cid: f"generated_tests/test_cases_{int(cid):04d}.parquet" in files for cid in by_contest}
    for cid, probs in by_contest.items():
        with gzip.open(f"{stage}/{cid}.json.gz", "wt") as g:
            json.dump({"has_generated": has_gen[cid], "problems": probs}, g)
    TESTS_VOLUME.commit()
    return sorted(by_contest)


@prep_app.function(image=prep_image, volumes={TESTS_MOUNT: TESTS_VOLUME}, timeout=3600, cpu=2, memory=16384,
                   max_containers=PREP_PARALLEL, retries=2)
def _contest_remote(split: str, cid: str) -> list[dict]:
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    TESTS_VOLUME.reload()
    done_path = f"{_staging(split)}/{cid}.done.json"
    with gzip.open(f"{_staging(split)}/{cid}.json.gz", "rt") as g:
        staged = json.load(g)
    ids = sorted(p["id"] for p in staged["problems"])
    if os.path.exists(done_path):
        prev = json.load(open(done_path))
        if prev.get("ids") == ids:  # same selection as last time: nothing to redo
            return prev["entries"]

    gen: dict[str, list[dict]] = {}
    if staged["has_generated"]:
        fname = f"generated_tests/test_cases_{int(cid):04d}.parquet"
        t = pq.read_table(hf_hub_download(HF_REPO, fname, repo_type="dataset"),
                          columns=["problem_id", "input", "output"], filters=[("problem_id", "in", ids)])
        for row in t.to_pylist():
            gen.setdefault(row["problem_id"], []).append({"input": row["input"], "output": row["output"]})
    entries = []
    for p in staged["problems"]:
        seen, tests = set(), []
        for tc in (p["official_tests"] or []) + gen.get(p["id"], []):
            if tc["input"] in seen:
                continue
            seen.add(tc["input"])
            tests.append({"input": tc["input"], "output": tc["output"]})
        if not tests:
            entries.append({"id": p["id"], "no_tests": True})
            continue
        rel = f"{vol_dir(split)}/{_safe_id(p['id'])}.json.gz"
        with gzip.open(f"{TESTS_MOUNT}/{rel}", "wt") as g:
            json.dump(tests, g)
        entries.append({
            "id": p["id"], "contest_id": p["contest_id"], "contest_start": p["contest_start"],
            "rating": p["rating"], "tags": list(p["tags"] or []), "title": p["title"],
            "description": p["description"], "input_format": p["input_format"],
            "output_format": p["output_format"], "note": p["note"],
            "examples": [{"input": e["input"], "output": e["output"]} for e in (p["examples"] or [])],
            "editorial": p["editorial"] or "", "time_limit": p["time_limit"], "memory_limit": p["memory_limit"],
            "n_tests": len(tests), "tests_file": rel,
        })
    with open(done_path, "w") as f:
        json.dump({"ids": ids, "entries": entries}, f)
    TESTS_VOLUME.commit()
    return entries


@prep_app.function(image=prep_image, volumes={TESTS_MOUNT: TESTS_VOLUME}, timeout=600)
def _finalize_remote(split: str, index: list[dict]) -> None:
    with open(f"{TESTS_MOUNT}/{vol_dir(split)}/index.json", "w") as f:
        json.dump(index, f)
    TESTS_VOLUME.commit()


def prepare(split: str = "test", min_rating: int = 1600, max_rating: int = 2400,
            require_editorial: bool = True, max_problems: int | None = None) -> dict:
    """Safe to re-run: contests already written with the same problems are skipped."""
    with modal.enable_output(), prep_app.run():
        cids = _select_remote.remote(split, min_rating, max_rating, require_editorial, max_problems)
        print(f"{len(cids)} contests to prepare ({PREP_PARALLEL} at a time)", flush=True)
        index, failed, no_tests = [], [], 0
        # list() drains the map inside the app context (a half-consumed generator errors on shutdown)
        results = list(_contest_remote.map([split] * len(cids), cids, return_exceptions=True))
        for i, (cid, res) in enumerate(zip(cids, results)):
            if isinstance(res, BaseException):
                failed.append(cid)
                print(f"  ! contest {cid}: {type(res).__name__}: {str(res)[:200]}", flush=True)
                continue
            no_tests += sum(1 for e in res if e.get("no_tests"))
            index += [e for e in res if not e.get("no_tests")]
            if (i + 1) % 25 == 0 or i + 1 == len(cids):
                print(f"  [{i + 1}/{len(cids)} contests] {len(index)} problems written", flush=True)
        index.sort(key=lambda e: (-(e["contest_start"] or 0), e["id"]))
        _finalize_remote.remote(split, index)
    ed_code = sum(bool(re.search(r"#include|int main\s*\(|\bcin\s*>>|\bprintf\s*\(", e["editorial"])) for e in index)
    stats = {"contests": len(cids), "failed_contests": len(failed), "written": len(index), "no_tests": no_tests,
             "with_editorial": sum(bool(e["editorial"].strip()) for e in index), "editorial_has_code": ed_code,
             "tests_per_problem_median": sorted(e["n_tests"] for e in index)[len(index) // 2] if index else 0}
    if failed:
        print(f"{len(failed)} contests failed (index written without them): re-run the same command to retry.")
    return stats


def load_index(split: str = "test", refresh: bool = False) -> list[dict]:
    local = LOCAL_DIR / f"verifiable_{split}_index.json"
    if local.exists() and not refresh:
        return json.loads(local.read_text())
    try:
        data = b"".join(TESTS_VOLUME.read_file(f"{vol_dir(split)}/index.json"))
    except Exception as e:
        raise SystemExit(f"No Codeforces {split} index on the volume. Run: uv run python scripts/prepare_codeforces.py --split {split}") from e
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return json.loads(data)


# ---------------------------------------------------------------- prompts for Test A

_CODE_LINE = re.compile(r"#include|int main\s*\(|\bcin\s*>>|\bcout\s*<<|\bprintf\s*\(|\bscanf\s*\(")


def strip_code(editorial: str) -> str:
    """Hint, not answer: drop fenced blocks and any line that looks like code."""
    text = re.sub(r"```.*?```", "", editorial, flags=re.DOTALL)
    return "\n".join(ln for ln in text.splitlines() if not _CODE_LINE.search(ln)).strip()


def statement_body(p: dict) -> str:
    """Statement without the title (examples, note and limits included)."""
    return statement(p).split("\n", 2)[2]


def statement(p: dict) -> str:
    parts = [p["title"], "", (p["description"] or "").strip(), "", "Input", (p["input_format"] or "").strip(),
             "", "Output", (p["output_format"] or "").strip()]
    for k, ex in enumerate(p["examples"], 1):
        parts += ["", f"Example {k}", "Input:", ex["input"].strip(), "Output:", ex["output"].strip()]
    if (p.get("note") or "").strip():
        parts += ["", "Note", p["note"].strip()]
    parts += ["", f"Time limit: {p['time_limit']:g} seconds. Memory limit: {p['memory_limit']:g} MB."]
    return "\n".join(parts)


def hint_text(p: dict, condition: str) -> str:
    if condition == "none":
        return ""
    if condition == "tags":
        return "Hint: this problem's topics are: " + ", ".join(p["tags"]) + "."
    if condition == "editorial":
        return "Hint (a sketch of the intended solution, from the official editorial):\n" + strip_code(p["editorial"])
    raise ValueError(condition)

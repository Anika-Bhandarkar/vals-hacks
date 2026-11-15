"""LiveCodeBench-CPP (nvidia/LiveCodeBench-CPP) problems, stored on a Modal Volume.

One-time prep (runs on Modal, downloads from HF there, so nothing big touches your laptop):
    uv run python scripts/prepare_lcb.py --split v6_2408_2505

Afterwards `load_index(split)` returns problem metadata (statement, difficulty, test count) and
graders read each problem's tests from the volume at `tests_file`.

Private tests ship as base64(zlib(pickle(json_str))), as in upstream LiveCodeBench. Pickle can
execute code, so we unpickle with a loader that refuses every class lookup: only plain
str/list/dict/number opcodes can load.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import pickle
import zlib
from pathlib import Path

import modal

from cppgrade.infra import TESTS_MOUNT, TESTS_VOLUME

HF_REPO = "nvidia/LiveCodeBench-CPP"
SPLITS = ("v5_2408_2501", "v6_2408_2505")
LOCAL_CACHE = Path(__file__).resolve().parents[1] / "data" / "lcb_cpp"


class _NoGlobalsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def decode_private_tests(blob: str) -> list[dict]:
    if not blob:
        return []
    if blob.lstrip().startswith("["):  # some rows are plain JSON
        return json.loads(blob)
    raw = zlib.decompress(base64.b64decode(blob))
    obj = _NoGlobalsUnpickler(io.BytesIO(raw)).load()
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode("utf-8")
    if isinstance(obj, str):
        obj = json.loads(obj)
    if not isinstance(obj, list):
        raise ValueError(f"unexpected private test payload type {type(obj).__name__}")
    return obj


def _volume_dir(split: str) -> str:
    return f"lcb_cpp/{split}"


# ---------------------------------------------------------------- Modal prep job

prep_app = modal.App("cpp-posttrain-lcb-prep")
prep_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub")
    .add_local_python_source("cppgrade")
)


@prep_app.function(image=prep_image, volumes={TESTS_MOUNT: TESTS_VOLUME}, timeout=3600, cpu=2, memory=8192)
def prepare_remote(split: str) -> dict:
    import os

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, f"test_{split}.jsonl", repo_type="dataset")
    out_dir = f"{TESTS_MOUNT}/{_volume_dir(split)}"
    os.makedirs(out_dir, exist_ok=True)

    index = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            public = json.loads(r["public_test_cases"] or "[]")
            private = decode_private_tests(r["private_test_cases"])
            tests = public + private
            stdin_tests = [{"input": t["input"], "output": t["output"]} for t in tests if t.get("testtype") == "stdin"]
            qid = r["question_id"]
            rel = f"{_volume_dir(split)}/{qid}.json.gz"
            if stdin_tests:
                with gzip.open(f"{TESTS_MOUNT}/{rel}", "wt") as g:
                    json.dump(stdin_tests, g)
            index.append(
                {
                    "question_id": qid,
                    "question_title": r["question_title"],
                    "question_content": r["question_content"],
                    "platform": r["platform"],
                    "contest_id": r["contest_id"],
                    "contest_date": r["contest_date"],
                    "difficulty": r["difficulty"],
                    "starter_code": r["starter_code"],
                    "metadata": r.get("metadata") or r.get("meta_data") or "{}",
                    "public_tests": public,
                    "n_tests": len(stdin_tests),
                    "tests_file": rel if stdin_tests else None,
                }
            )
    with open(f"{out_dir}/index.json", "w") as f:
        json.dump(index, f)
    TESTS_VOLUME.commit()
    by_platform: dict[str, int] = {}
    for p in index:
        by_platform[p["platform"]] = by_platform.get(p["platform"], 0) + 1
    return {"problems": len(index), "by_platform": by_platform, "with_stdin_tests": sum(bool(p["n_tests"]) for p in index)}


def prepare(split: str) -> dict:
    with modal.enable_output(), prep_app.run():
        return prepare_remote.remote(split)


# ---------------------------------------------------------------- local index


def load_index(split: str, refresh: bool = False) -> list[dict]:
    """Problem metadata (no private tests). Cached under data/lcb_cpp/ after first fetch."""
    local = LOCAL_CACHE / f"{split}_index.json"
    if local.exists() and not refresh:
        return json.loads(local.read_text())
    try:
        data = b"".join(TESTS_VOLUME.read_file(f"{_volume_dir(split)}/index.json"))
    except Exception as e:
        raise SystemExit(
            f"No prepared index for split {split!r} on the Modal volume ({type(e).__name__}).\n"
            f"Run once:  uv run python scripts/prepare_lcb.py --split {split}"
        ) from e
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return json.loads(data)

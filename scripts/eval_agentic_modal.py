"""Run scripts/eval_agentic.py inside a Modal CPU container, billed to a chosen Tinker account, so it
survives the laptop closing and can use the second account's credits.

    uv run modal run --detach scripts/eval_agentic_modal.py --account personal --name lcb287-base8b \
        --model Qwen/Qwen3-8B --harness vals --limit 287 --difficulty easy medium hard ...
    uv run modal run scripts/eval_agentic_modal.py --pull lcb287-base8b      # -> runs/evals/lcb287-base8b/
    uv run python scripts/eval_agentic.py --summarize runs/evals/lcb287-base8b

All eval_agentic.py flags pass through. Results go to the runs Volume at evals/<name>/ (committed every
30 s); re-launching the same --name resumes (eval_agentic.py skips finished tasks). `--account` picks the
Modal Secret holding that account's TINKER_API_KEY (see ACCOUNT_SECRETS).
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
RUNS_VOLUME_NAME = "cpp-posttrain-runs"
RUNS_VOLUME = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
RUNS_MOUNT = "/runs"
ACCOUNT_SECRETS = {"main": "tinker-api", "personal": "tinker-api-personal"}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(ROOT / "scripts" / "modal_rl_requirements.txt"), extra_options="--no-deps")
    .env({"VALS_EVALS_DIR": f"{RUNS_MOUNT}/evals", "PYTHONUNBUFFERED": "1"})
    .add_local_file(str(ROOT / "data/codeforces/verifiable_train_index.json"), "/root/data/codeforces/verifiable_train_index.json")
    .add_local_file(str(ROOT / "data/codeforces/verifiable_test_index.json"), "/root/data/codeforces/verifiable_test_index.json")
    .add_local_file(str(ROOT / "data/lcb_cpp/v6_2408_2505_index.json"), "/root/data/lcb_cpp/v6_2408_2505_index.json")
    .add_local_file(str(ROOT / "scripts/__init__.py"), "/root/scripts/__init__.py")
    .add_local_file(str(ROOT / "scripts/eval_agentic.py"), "/root/scripts/eval_agentic.py")
    .add_local_python_source("cppgrade", "posttrain")
)
app = modal.App("cpp-posttrain-eval", image=image)


def _committer(stop: threading.Event, every_s: float = 30.0) -> None:
    while not stop.wait(every_s):
        try:
            RUNS_VOLUME.commit()
        except Exception as e:
            print(f"[eval_agentic_modal] volume commit failed: {type(e).__name__}: {e}", flush=True)


def _eval(argv: list[str]) -> None:
    sys.path.insert(0, "/root")
    try:
        RUNS_VOLUME.reload()
    except RuntimeError as e:  # a retry in the same container still has files open
        print(f"volume reload skipped ({e})", flush=True)
    stop = threading.Event()
    threading.Thread(target=_committer, args=(stop,), daemon=True).start()
    try:
        from scripts import eval_agentic

        sys.argv = ["eval_agentic.py", *argv]
        eval_agentic.main()
    finally:
        stop.set()
        RUNS_VOLUME.commit()


_FN = dict(cpu=2.0, memory=4096, timeout=24 * 3600, volumes={RUNS_MOUNT: RUNS_VOLUME})


@app.function(secrets=[modal.Secret.from_name(ACCOUNT_SECRETS["main"])], **_FN)
def eval_main(argv: list[str]) -> None:
    _eval(argv)


@app.function(secrets=[modal.Secret.from_name(ACCOUNT_SECRETS["personal"])], **_FN)
def eval_personal(argv: list[str]) -> None:
    _eval(argv)


def pull(name: str) -> None:
    rel, dest, n = f"evals/{name}", ROOT / "runs" / "evals" / name, 0
    for entry in RUNS_VOLUME.iterdir(rel, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        out = dest / Path(entry.path).relative_to(rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in RUNS_VOLUME.read_file(entry.path):
                f.write(chunk)
        n += 1
    print(f"pulled {n} files -> {dest}")


@app.local_entrypoint()
def main(*argv: str) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--account", default="main", choices=sorted(ACCOUNT_SECRETS))
    ap.add_argument("--pull")
    a, rest = ap.parse_known_args(list(argv))
    if a.pull:
        pull(a.pull)
        return
    fn = {"main": eval_main, "personal": eval_personal}[a.account]
    call = fn.spawn(rest)
    print(f"started eval on Modal (account {a.account}, secret {ACCOUNT_SECRETS[a.account]})\n"
          f"  app id: {app.app_id}\n  call id: {call.object_id}\n"
          f"  pull:   uv run modal run scripts/eval_agentic_modal.py --pull <name>", flush=True)

"""Run scripts/train_rl.py inside a Modal CPU container, so the run survives the laptop closing.

Tinker still samples and trains; grading still runs in Modal Sandboxes (created from inside the
container). Only the training loop moves off the laptop.

One-time: a Modal Secret holding the Tinker key (only TINKER_API_KEY; nothing else is uploaded):

    modal secret create tinker-api TINKER_API_KEY=<your key>

Start (all train_rl.py flags pass through; `--detach` keeps it running after the client exits):

    uv run modal run --detach scripts/train_rl_modal.py --name rl-v1 --load-checkpoint tinker://.../weights/final

Watch / stop / pull results:

    uv run modal app list                               # the run's app id (also printed at start)
    uv run modal app logs <app id>                      # live logs
    uv run modal app stop -y <app id>                   # stop; resume later with --if-exists resume
    uv run modal run scripts/train_rl_modal.py --pull rl-v1          # -> runs/train/rl-v1/
    uv run python scripts/train_rl.py --summarize runs/train/rl-v1

Run folders live on the Modal Volume `cpp-posttrain-runs` at train/<name>/ (committed every 30 s).

Resume: a preempted, crashed or timed-out container is retried (up to MAX_RETRIES) and resumes from the
newest checkpoint in the run folder; `--rolling-save-every 1` (the default) saves a state-only resume
point every step. `--if-exists` is applied once, here at launch (raise | resume | delete); the
container itself always resumes, so a retry never wipes the run.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

APP_NAME = "cpp-posttrain-rl"
RUNS_VOLUME_NAME = "cpp-posttrain-runs"
RUNS_VOLUME = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
RUNS_MOUNT = "/runs"
MAX_RETRIES = 4
TIMEOUT_S = 24 * 3600  # Modal's maximum; a retry resumes after it

# Same Python and the same locked versions as the .venv (torch from the CPU wheel index: no CUDA).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # the cookbook records the code state with git when it starts logging
    .pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(ROOT / "scripts" / "modal_rl_requirements.txt"), extra_options="--no-deps")
    .env({"VALS_RUNS_DIR": f"{RUNS_MOUNT}/train", "PYTHONUNBUFFERED": "1"})
    .add_local_file(str(ROOT / "data/codeforces/verifiable_train_index.json"), "/root/data/codeforces/verifiable_train_index.json")
    .add_local_file(str(ROOT / "data/codeforces/verifiable_test_index.json"), "/root/data/codeforces/verifiable_test_index.json")
    .add_local_file(str(ROOT / "data/lcb_cpp/v6_2408_2505_index.json"), "/root/data/lcb_cpp/v6_2408_2505_index.json")
    .add_local_file(str(ROOT / "scripts/train_rl.py"), "/root/scripts/train_rl.py")
    .add_local_python_source("cppgrade", "posttrain")
)

app = modal.App(APP_NAME, image=image)


def _launch_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--name")
    ap.add_argument("--if-exists", default="raise")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize")
    ap.add_argument("--pull")
    return ap.parse_known_args(argv)[0]


def _without_if_exists(argv: list[str]) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == "--if-exists":
            skip = True
        elif not a.startswith("--if-exists="):
            out.append(a)
    return out


def _committer(stop: threading.Event, every_s: float = 30.0) -> None:
    while not stop.wait(every_s):
        try:
            RUNS_VOLUME.commit()
        except Exception as e:  # a failed commit is retried next tick
            print(f"[train_rl_modal] volume commit failed: {type(e).__name__}: {e}", flush=True)


@app.function(
    cpu=2.0, memory=4096, timeout=TIMEOUT_S,
    retries=modal.Retries(max_retries=MAX_RETRIES, initial_delay=30.0, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("tinker-api")],
    volumes={RUNS_MOUNT: RUNS_VOLUME},
)
def train(argv: list[str]) -> None:
    sys.path.insert(0, "/root/scripts")
    import train_rl

    try:
        RUNS_VOLUME.reload()  # a fresh container: pick up the run folder's latest commit
    except RuntimeError as e:
        # A retry in the SAME container (Modal reuses it) still has the previous attempt's log file open,
        # so reload refuses; that container's view of the volume is already current.
        print(f"volume reload skipped ({e}); resuming from this container's view", flush=True)
    stop = threading.Event()
    t = threading.Thread(target=_committer, args=(stop,), daemon=True)
    t.start()
    try:
        train_rl.main(_without_if_exists(argv) + ["--if-exists", "resume"])
    finally:
        stop.set()
        RUNS_VOLUME.commit()


def _remote_exists(rel: str) -> bool:
    try:
        return bool(list(RUNS_VOLUME.iterdir(rel)))
    except Exception:  # NotFound
        return False


@app.local_entrypoint()
def main(*argv: str) -> None:
    argv = list(argv)
    a = _launch_args(argv)
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_rl

    if a.pull:
        pull(a.pull)
        return
    if a.dry_run or a.summarize or not a.name:
        train_rl.main(argv)  # validation / help runs locally
        return
    train_rl.main(_without_if_exists(argv) + ["--dry-run"])  # fail fast on bad flags, before any container starts
    rel = f"train/{a.name}"
    if _remote_exists(rel):
        if a.if_exists == "raise":
            raise SystemExit(f"{RUNS_VOLUME_NAME}:{rel} exists. Pass --if-exists resume (continue it) or delete.")
        if a.if_exists == "delete":
            RUNS_VOLUME.remove_file(rel, recursive=True)
            print(f"deleted {RUNS_VOLUME_NAME}:{rel}")
        elif a.if_exists != "resume":
            raise SystemExit(f"--if-exists {a.if_exists} isn't supported on Modal (use raise, resume or delete)")
    print(f"starting {a.name} on Modal; folder {RUNS_VOLUME_NAME}:{rel}\n"
          f"  app id: {app.app_id}\n"
          f"  watch:  uv run modal app logs {app.app_id}\n"
          f"  stop:   uv run modal app stop -y {app.app_id}", flush=True)
    # spawn, not remote: a .remote() call is cancelled when the local client is killed, even with --detach.
    # A spawned call belongs to the detached app, which keeps running until the call finishes.
    call = train.spawn(argv)
    print(f"  call id: {call.object_id}\n"
          f"Launched; this terminal can close. When it's done (`modal app list` shows the app stopped):\n"
          f"  uv run modal run scripts/train_rl_modal.py --pull {a.name}", flush=True)


def pull(name: str) -> None:
    """Copy train/<name>/ from the runs Volume to runs/train/<name>/ (overwrites local copies)."""
    rel = f"train/{name}"
    dest = ROOT / "runs" / "train" / name
    n = 0
    for entry in RUNS_VOLUME.iterdir(rel, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        out = dest / Path(entry.path).relative_to(rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in RUNS_VOLUME.read_file(entry.path):
                f.write(chunk)
        n += 1
    print(f"pulled {n} files to {dest.relative_to(ROOT)}/")

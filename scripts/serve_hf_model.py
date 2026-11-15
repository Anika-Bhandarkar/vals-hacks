"""Serve a Hugging Face model (e.g. a candidate teacher not on Tinker) on Modal with vLLM.

    uv run python scripts/serve_hf_model.py deploy               # Qwen/Qwen3-32B on 2x H100
    MODEL=Qwen/Qwen3-14B GPU=H100:1 TP=1 uv run python scripts/serve_hf_model.py deploy
    uv run python scripts/serve_hf_model.py url                  # print the endpoint URL
    uv run python scripts/serve_hf_model.py stop                 # stop the app (no idle cost after)

OpenAI-compatible endpoint protected by an API key generated on first deploy and kept in
data/serve_api_key (gitignored). Scales to zero after 10 idle minutes. First start downloads the
weights (~65GB for Qwen3-32B) into the `huggingface-cache` Volume; later starts reuse them.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / "data" / "serve_api_key"
APP_NAME = "cpp-posttrain-hf-serve"
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")
GPU = os.environ.get("GPU", "H100:2")
TP = int(os.environ.get("TP", "2"))
CONTEXT = int(os.environ.get("CONTEXT", "32768"))
PORT = 30000
HF_CACHE = "/cache/huggingface"
MINUTES = 60

# Official vLLM OpenAI-server image, a stable release. (The SGLang nightly the cookbook pins has
# been deleted from Docker Hub.)
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.32.0")
    .entrypoint([])
    .env({"HF_HUB_CACHE": HF_CACHE, "HF_XET_HIGH_PERFORMANCE": "1", "MODEL": MODEL,
          "TP": str(TP), "CONTEXT": str(CONTEXT)})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


def _api_key() -> str:
    if not KEY_FILE.exists():
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        KEY_FILE.write_text(secrets.token_urlsafe(32))
        KEY_FILE.chmod(0o600)
    return KEY_FILE.read_text().strip()


# Only resolvable locally (the key file isn't in the container); inside the container the
# value arrives through the secret instead.
_key_secret = modal.Secret.from_dict({"SERVE_API_KEY": _api_key()}) if modal.is_local() else modal.Secret.from_dict({})

app = modal.App(APP_NAME)

with image.imports():
    import requests


@app.server(
    image=image,
    gpu=GPU,
    volumes={HF_CACHE: hf_cache},
    secrets=[_key_secret],
    port=PORT,
    target_concurrency=64,
    scaledown_window=10 * MINUTES,
    startup_timeout=40 * MINUTES,
    exit_grace_period=15,
    unauthenticated=True,  # protected by the server's --api-key instead of Modal proxy tokens
)
class Server:
    @modal.enter()
    def start(self) -> None:
        argv = ["vllm", "serve", os.environ["MODEL"],
                "--served-model-name", "model", "--host", "0.0.0.0", "--port", str(PORT),
                "--tensor-parallel-size", os.environ["TP"], "--max-model-len", os.environ["CONTEXT"],
                "--api-key", os.environ["SERVE_API_KEY"], "--download-dir", HF_CACHE]
        self.proc = subprocess.Popen(argv, env=os.environ.copy())
        deadline = time.time() + 40 * MINUTES
        while time.time() < deadline:
            if (rc := self.proc.poll()) is not None:
                raise subprocess.CalledProcessError(rc, argv)
            try:
                requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5).raise_for_status()
                return
            except Exception:
                time.sleep(5)
        raise TimeoutError("vLLM did not become healthy")

    @modal.exit()
    def stop(self) -> None:
        self.proc.terminate()


def endpoint_url() -> str:
    return modal.Server.from_name(APP_NAME, "Server").get_url()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "deploy"
    if cmd == "deploy":
        _api_key()
        with modal.enable_output():
            app.deploy()
        print(f"Deployed {MODEL} on {GPU} (tp={TP}).")
        try:
            print("URL:", endpoint_url())
        except Exception as e:
            print("(URL lookup failed:", e, ") - see the deploy output above")
    elif cmd == "url":
        print(endpoint_url())
    elif cmd == "stop":
        subprocess.run(["modal", "app", "stop", APP_NAME], check=False)
    else:
        raise SystemExit(f"unknown command {cmd!r}: use deploy | url | stop")

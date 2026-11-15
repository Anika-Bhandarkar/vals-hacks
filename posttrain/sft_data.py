"""One JSONL format for every SFT source, plus conversions and quality filters.

Each line:
    {"messages": [...], "meta": {"source": ..., "problem_id": ..., ...}}

Messages are plain JSON (so files are easy to read and mix):
    {"role": "system" | "user" | "tool", "content": str, "name"?: str}
    {"role": "assistant", "content": str, "thinking"?: str, "tool_calls"?: [{"name": str, "arguments": str}]}

`to_renderer_messages` turns them into cookbook Messages (thinking parts, ToolCall objects).
Multi-turn conversations are trained one example per assistant turn, each with earlier turns'
thinking removed: exactly what the model sees at inference with Qwen3's template.

Sources (built by scripts/build_sft_data.py):
    rft-single   the model's own shortest correct single-attempt answers (rejection sampling)
    rft-agentic  the model's own best competition sessions (score, few turns, no resubmit loops)
    traces       external solutions (e.g. open-r1/codeforces-cots R1 traces, or a Modal-hosted
                 teacher's generations), kept only if they pass the hidden tests
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- io


def write_jsonl(path: str | Path, records: Iterable[dict], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def record(messages: list[dict], **meta: Any) -> dict:
    return {"messages": messages, "meta": meta}


# ---------------------------------------------------------------- message helpers

_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


def split_thinking(text: str) -> tuple[str, str]:
    """'<think>...</think>answer' -> (thinking, answer). Text without a think block -> ("", text)."""
    m = _THINK.match(text)
    return (m.group(1).strip(), text[m.end():]) if m else ("", text)


def assistant(text: str, thinking: str = "", tool_calls: list[dict] | None = None) -> dict:
    m: dict = {"role": "assistant", "content": text}
    if thinking:
        m["thinking"] = thinking
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


def from_agentic_transcript(transcript: list[dict]) -> list[dict]:
    """eval_agentic.py `transcript` -> SFT messages (same shape; drops malformed-call markers)."""
    out = []
    for m in transcript:
        if m["role"] == "assistant":
            out.append(assistant(m.get("content", ""), m.get("thinking", ""), m.get("tool_calls")))
        else:
            mm = {"role": m["role"], "content": m.get("content", "")}
            if m.get("name"):
                mm["name"] = m["name"]
            out.append(mm)
    return out


def to_renderer_messages(messages: list[dict]) -> list[dict]:
    """SFT JSON messages -> cookbook renderer Messages."""
    from tinker_cookbook.renderers.base import ToolCall

    out = []
    for m in messages:
        if m["role"] != "assistant":
            mm = {"role": m["role"], "content": m["content"]}
            if m.get("name"):
                mm["name"] = m["name"]
            out.append(mm)
            continue
        parts = []
        if m.get("thinking"):
            parts.append({"type": "thinking", "thinking": m["thinking"]})
        parts.append({"type": "text", "text": m.get("content", "")})
        mm = {"role": "assistant", "content": parts}
        if m.get("tool_calls"):
            mm["tool_calls"] = [
                ToolCall(function=ToolCall.FunctionBody(name=tc["name"], arguments=tc["arguments"]))
                for tc in m["tool_calls"]
            ]
        out.append(mm)
    return out


# ---------------------------------------------------------------- agentic session quality


def submitted_codes(transcript: list[dict]) -> list[str]:
    codes = []
    for m in transcript:
        for tc in m.get("tool_calls", []) or []:
            if tc["name"] == "submit":
                try:
                    codes.append(json.loads(tc["arguments"]).get("code", ""))
                except json.JSONDecodeError:
                    codes.append(tc["arguments"])
    return codes


def identical_resubmits(transcript: list[dict]) -> int:
    codes = [hashlib.sha1(c.strip().encode()).hexdigest() for c in submitted_codes(transcript)]
    return len(codes) - len(set(codes))


def session_ok(
    rec: dict,
    min_score: float = 100.0,
    max_turns: int = 8,
    allow_resubmits: int = 0,
    require_exit: bool = True,
) -> tuple[bool, str]:
    """Is an eval_agentic-style session record good enough to imitate? Returns (ok, reason)."""
    t = rec["transcript"]
    if rec["score"] < min_score:
        return False, "score"
    if rec["parse_errors"]:
        return False, "parse_error"
    if rec["turns"] > max_turns:
        return False, "turns"
    if rec.get("best_score", rec["score"]) > rec["score"]:
        return False, "regressed"
    if identical_resubmits(t) > allow_resubmits:
        return False, "resubmits"
    if require_exit and not rec.get("clean_exit"):
        return False, "no_exit"
    if rec.get("stop_reason") not in (None, "completed"):
        return False, f"stop:{rec.get('stop_reason')}"
    return True, "ok"


def code_hash(code: str | None) -> str:
    return hashlib.sha1(re.sub(r"\s+", "", code or "").encode()).hexdigest()[:16]

"""Keep eval problems out of training data.

Two checks:
  1. ID blocklist: every LiveCodeBench-CPP problem (the single-attempt and agentic evals) and the
     Codeforces `verifiable` test split (Test A).
  2. Statement overlap: word 13-grams shared with any eval statement (catches the same problem
     re-posted under another ID). 13-grams that appear in several eval statements are boilerplate
     ("the first line of the input contains ...") and are ignored.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

N = 13
OVERLAP_THRESHOLD = 0.2  # fraction of a candidate's 13-grams found in one eval statement


def _ngrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + N]) for i in range(len(words) - N + 1)}


@lru_cache(maxsize=1)
def _eval_set() -> tuple[frozenset[str], dict[tuple[str, ...], frozenset[str]]]:
    from cppgrade.lcb import SPLITS, LOCAL_CACHE, load_index

    ids: set[str] = set()
    statements: dict[str, str] = {}
    for split in SPLITS:
        if (LOCAL_CACHE / f"{split}_index.json").exists() or split == "v6_2408_2505":
            for p in load_index(split):
                ids.add(p["question_id"])
                statements[p["question_id"]] = p["question_content"]
    try:
        from cppgrade.codeforces import load_index as cf_index

        for p in cf_index("test"):
            ids.add(p["id"])
            statements[p["id"]] = f"{p['description']} {p['input_format']} {p['output_format']}"
    except SystemExit:
        pass  # Test A set not prepared yet: nothing to block from it
    owners: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for qid, text in statements.items():
        for g in _ngrams(text):
            owners[g].add(qid)
    index = {g: frozenset(o) for g, o in owners.items() if len(o) <= 2}  # drop boilerplate n-grams
    return frozenset(ids), index


def eval_ids() -> frozenset[str]:
    return _eval_set()[0]


def overlap(text: str) -> tuple[float, str | None]:
    """Largest fraction of `text`'s 13-grams found in a single eval statement, and which one."""
    grams = _ngrams(text)
    if not grams:
        return 0.0, None
    _, index = _eval_set()
    hits: dict[str, int] = defaultdict(int)
    for g in grams:
        for qid in index.get(g, ()):
            hits[qid] += 1
    if not hits:
        return 0.0, None
    qid, n = max(hits.items(), key=lambda kv: kv[1])
    return n / len(grams), qid


def is_contaminated(problem_id: str, statement: str) -> bool:
    return problem_id in eval_ids() or overlap(statement)[0] >= OVERLAP_THRESHOLD

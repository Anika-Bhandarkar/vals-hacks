"""One-time: load open-r1/codeforces problems with editorials into the Modal tests volume (for Test A).

    uv run python scripts/prepare_codeforces.py                     # test split, rating 1600-2400 (Test A)
    uv run python scripts/prepare_codeforces.py --split train --min-rating 1200 --max-rating 2600 \
        --no-require-editorial --max-problems 2000                    # training pool (newest first)

Runs on Modal (downloads there). Safe to re-run.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cppgrade.codeforces import load_index, prepare  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["test", "train"], default="test")
ap.add_argument("--min-rating", type=int, default=1600)
ap.add_argument("--max-rating", type=int, default=2400)
ap.add_argument("--no-require-editorial", action="store_true")
ap.add_argument("--max-problems", type=int, default=None)
a = ap.parse_args()
print(prepare(a.split, a.min_rating, a.max_rating, not a.no_require_editorial, a.max_problems))
print("Cached index locally:", len(load_index(a.split, refresh=True)), "problems")

"""One-time: load LiveCodeBench-CPP into the Modal tests volume (download + decode happen on Modal).

    uv run python scripts/prepare_lcb.py                 # v6 (454 problems, superset of v5)
    uv run python scripts/prepare_lcb.py --split v5_2408_2501

Takes a few minutes; costs cents. Safe to re-run (overwrites).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cppgrade.lcb import SPLITS, load_index, prepare  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=SPLITS, default="v6_2408_2505")
args = ap.parse_args()
print(prepare(args.split))
print("Cached index locally:", len(load_index(args.split, refresh=True)), "problems")

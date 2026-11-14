"""Modal smoke test: grades hand-written programs that should hit every verdict.

    uv run python scripts/check_modal.py

Costs well under $0.01 (a handful of CPU sandboxes for a few seconds each).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cppgrade.grader import grade, make_pool  # noqa: E402

TESTS = [{"input": "1 2\n", "output": "3\n"}, {"input": "40 2\n", "output": "42\n"}]


def fence(code: str) -> str:
    return f"Here is my solution:\n```cpp\n{code}\n```\n"


CASES = [
    ("accepted", None, False, fence("#include <bits/stdc++.h>\nint main(){long long a,b;std::cin>>a>>b;std::cout<<a+b<<'\\n';}")),
    ("format_error", None, False, "I think the answer is a+b."),
    ("compile_error", "missing_include", False, fence("#include <iostream>\nint main(){std::vector<int> v{1};long long a,b;std::cin>>a>>b;std::cout<<a+b;}")),
    ("compile_error", "undeclared", False, fence("#include <iostream>\nint main(){long long a,b;std::cin>>a>>b;std::cout<<add(a,b);}")),
    ("compile_error", "type_error", False, fence("#include <string>\n#include <iostream>\nint main(){std::string s = 5.0; int x = s; std::cout<<x;}")),
    ("wrong_answer", None, False, fence("#include <iostream>\nint main(){long long a,b;std::cin>>a>>b;std::cout<<a-b;}")),
    ("time_limit", None, False, fence("#include <iostream>\nint main(){volatile long long x=0;while(true)x++;}")),
    ("runtime_error", "segfault", False, fence("#include <iostream>\nint main(){int*p=nullptr;*p=1;}")),
    # Off-by-one that silently "works" without sanitizers; ASan should flag it.
    ("runtime_error", "asan_oob", True, fence("#include <bits/stdc++.h>\nint main(){long long a,b;std::cin>>a>>b;"
                                             "std::vector<int> v(2);v[2]=1;std::cout<<a+b+v[0];}")),
]


async def main():
    pool = make_pool(pool_size=len(CASES))
    try:
        results = await asyncio.gather(
            *(grade(pool, text, TESTS, sanitize=san) for _, _, san, text in CASES)
        )
    finally:
        await pool.terminate()

    ok = True
    for (want_v, want_s, san, _), r in zip(CASES, results):
        good = r.verdict == want_v and (want_s is None or r.subtype == want_s)
        ok &= good
        mark = "PASS" if good else "FAIL"
        print(f"[{mark}] want {want_v}/{want_s or '-'} got {r.verdict}/{r.subtype or '-'}"
              f"  passed={r.passed}/{r.total}{'  (asan)' if san else ''}")
        if not good and r.detail:
            print("       ", r.detail[:500].replace("\n", "\n        "))
    print("\nModal grader OK" if ok else "\nSome checks failed (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())

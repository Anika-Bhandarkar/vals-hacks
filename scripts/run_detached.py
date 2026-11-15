"""Start a long command fully detached: its own session (so closing the terminal or app that launched it
doesn't kill it), output to a log file, and `caffeinate -is` so the Mac doesn't sleep while it runs.

    uv run python scripts/run_detached.py runs/opd-27b-9b.out -- uv run python scripts/train_distill.py ...

Prints the PID. Watch with `tail -f <log>`; stop with `kill <pid>` (or `pkill -f train_distill`).
"""

import subprocess
import sys

if len(sys.argv) < 4 or sys.argv[2] != "--":
    sys.exit(__doc__)
log, cmd = sys.argv[1], sys.argv[3:]
p = subprocess.Popen(["caffeinate", "-is", *cmd], stdin=subprocess.DEVNULL, stdout=open(log, "ab"),
                     stderr=subprocess.STDOUT, start_new_session=True)
print(p.pid)

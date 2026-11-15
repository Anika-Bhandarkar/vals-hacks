"""Shared Modal objects. Stdlib + modal only, so this also imports inside Modal containers."""

import modal

APP_NAME = "cpp-posttrain-grader"

# Modal's python3.12 slim image is Debian 13 (trixie) -> GCC 14 (confirmed from compile errors).
# bits/stdc++.h available, as on most judges. python3.12 is used for Python evals.
CPP_IMAGE = modal.Image.debian_slim(python_version="3.12").apt_install("g++")

# Problem test cases live here (written once by cppgrade.lcb.prepare), mounted into every sandbox.
TESTS_VOLUME = modal.Volume.from_name("cpp-posttrain-tests", create_if_missing=True)
TESTS_MOUNT = "/data"

# Fixed resources so timing is comparable across runs (Modal's default is a fraction of a core).
SANDBOX_CPU = 1.0
SANDBOX_MEMORY_MB = 2048

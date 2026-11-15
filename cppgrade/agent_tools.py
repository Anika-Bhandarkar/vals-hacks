"""The two competition tools, bound to one task: run_cpp (compile + run on any input) and submit.

State kept for the episode-end grader: every submission, and the last one that compiled (that's
the graded one). The hidden tests are never run here, only at the end via `final_score()`.

Usage:
    tools = CompetitionTools(task, pool)
    env_tools = [tools.run_cpp, tools.submit]        # cookbook FunctionTools
    ...
    result = await tools.final_score()               # ScoreResult, 0-100
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Annotated, Any

from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

from cppgrade.agent_tasks import MAX_SUBMISSIONS, AgentTask
from cppgrade.grader import ScoreResult, check_samples, run_program, score_submission

STDOUT_CHARS = 4000  # tool output shown to the model


@dataclass
class Submission:
    turn_time: float
    code: str
    compiled: bool
    sample_verdicts: list[str]


@dataclass
class CompetitionTools:
    task: AgentTask
    pool: Any  # OnDemandPool | ModalSandboxPool
    submissions: list[Submission] = field(default_factory=list)
    n_run_calls: int = 0
    n_infra_errors: int = 0
    _score_cache: dict[str, ScoreResult] = field(default_factory=dict, repr=False)

    @property
    def last_compiling(self) -> Submission | None:
        return next((s for s in reversed(self.submissions) if s.compiled), None)

    # ------------------------------------------------------------ tools

    @tool
    async def run_cpp(
        self,
        code: Annotated[str, "A complete C++20 program."],
        stdin: Annotated[str, "The input to feed the program on standard input."],
    ) -> ToolResult:
        """Compile a C++20 program and run it on the given input. Returns compiler errors, or the program's stdout, stderr, exit code and running time. Nothing is submitted."""
        self.n_run_calls += 1
        r = await run_program(self.pool, code, stdin, time_limit=self.task.time_limit_s,
                              memory_mb=self.task.memory_mb, output_chars=STDOUT_CHARS)
        if "infra_error" in r:
            self.n_infra_errors += 1
            return simple_tool_result("Internal error while running your code; please try again.",
                                      metrics={"infra_error": 1.0})
        if not r["compiled"]:
            return simple_tool_result(f"Compilation failed:\n{r['compile_stderr']}", metrics={"run_compile_error": 1.0})
        if r["timed_out"]:
            status = f"TIME LIMIT EXCEEDED (killed after {self.task.time_limit_s:g} s)"
        elif r["output_limit"]:
            status = "OUTPUT LIMIT EXCEEDED"
        elif r["exit_code"] != 0:
            status = f"RUNTIME ERROR (exit code {r['exit_code']})"
        else:
            status = "OK"
        out = r["stdout"] + ("\n[stdout truncated]" if r["stdout_truncated"] else "")
        text = f"Status: {status}\nTime: {r['time_s']:.3f} s\n--- stdout ---\n{out}"
        if r["stderr"].strip():
            text += f"\n--- stderr ---\n{r['stderr']}"
        return simple_tool_result(text)

    @tool
    async def submit(
        self,
        code: Annotated[str, "A complete C++20 program."],
    ) -> ToolResult:
        """Submit a solution. Reports how it does on the sample tests. The last submission that compiles is the one graded on the hidden tests."""
        if len(self.submissions) >= MAX_SUBMISSIONS:
            return simple_tool_result(f"Submission limit reached ({MAX_SUBMISSIONS}). Reply EXIT to finish.")
        r = await check_samples(self.pool, code, self.task.sample_tests,
                                time_limit=self.task.time_limit_s, memory_mb=self.task.memory_mb)
        if "infra_error" in r:
            self.n_infra_errors += 1
            return simple_tool_result("Internal error while judging; this submission was not recorded. Please submit again.",
                                      metrics={"infra_error": 1.0})
        verdicts = [x["verdict"] for x in r["results"]]
        self.submissions.append(Submission(time.time(), code, r["compiled"], verdicts))
        n = len(self.submissions)
        if not r["compiled"]:
            return simple_tool_result(
                f"Submission {n}/{MAX_SUBMISSIONS}: COMPILATION ERROR (not graded; your previous compiling "
                f"submission, if any, still counts).\n{r['compile_error']}"
            )
        lines = [f"Submission {n}/{MAX_SUBMISSIONS} compiled. This is now your graded submission.",
                 f"Sample tests: {sum(v == 'accepted' for v in verdicts)}/{len(verdicts)} passed."]
        for x in r["results"]:
            line = f"  sample {x['sample']}: {x['verdict'].upper().replace('_', ' ')}"
            if x["verdict"] == "wrong_answer":
                line += f"\n    expected: {x['want'].strip()[:300]!r}\n    got:      {x['got'].strip()[:300]!r}"
            elif x["verdict"] == "runtime_error" and x.get("stderr"):
                line += f"\n    stderr: {x['stderr'].strip()[-300:]}"
            lines.append(line)
        return simple_tool_result("\n".join(lines))

    # ------------------------------------------------------------ end of episode

    async def score_code(self, code: str) -> ScoreResult:
        """Hidden-test score of any code, cached by source (scoring runs every hidden test)."""
        if code not in self._score_cache:
            self._score_cache[code] = await score_submission(
                self.pool, code, tests_file=self.task.tests_file, n_tests=self.task.n_tests,
                time_limit=self.task.time_limit_s, memory_mb=self.task.memory_mb)
        return self._score_cache[code]

    async def final_score(self) -> ScoreResult:
        """Hidden-test score of the last compiling submission (0 if there is none). This is what the
        competition grades, however the episode ended (EXIT, turn limit, malformed calls, context)."""
        sub = self.last_compiling
        if sub is None:
            return ScoreResult(compiled=False)
        return await self.score_code(sub.code)

    def stats(self) -> dict[str, Any]:
        return {
            "submissions": len(self.submissions),
            "compiling_submissions": sum(s.compiled for s in self.submissions),
            "run_calls": self.n_run_calls,
            "infra_errors": self.n_infra_errors,
        }

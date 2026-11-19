# C++ post-training workspace

- I was able to improve Qwen3.5-9B by 52 percentage points if partial credit on the LiveCodeBench benchmark adapted for agentic use with supported tool-calling capabilities. Pass@1 went from 11% to 53%. RL setup included a multi-turn agentic environment with outcome rewards. Rewards tested output code on a hidden test suite. 
- More broadly, this repo includes eval code in agentic and non-agentic environments, C++ vs. Python comparisons to isolate C++ capabilities from general coding reasoning capabilities, custom RL environment setup mimicking the competition harness, and functionality to post-train using SFT, RL, or distillation.  

## What this repo does

Post-trains **Qwen3.5-9B** (LoRA on [Tinker](https://thinkingmachines.ai/tinker)) to solve hard competitive-programming problems in C++, in the format of the [competition harness](https://github.com/vals-ai/hackathon-post-training-agent):

- tools: compile-and-run, and submit (up to 50 submits, 100 turns)
- grading: the last compiling submission, on hidden tests, with partial credit
- limits: 16K output tokens per turn, 2 s and 1 GiB per test

| Piece | Where | What it does |
|---|---|---|
| Grader | `cppgrade/grader.py`, `cppgrade/runner.py` | Compiles and runs C++ in Modal sandboxes. Returns a verdict taxonomy, and 40/60 small/large scores, both all-or-nothing and partial credit. Submissions run statically linked in an empty `chroot` as an unprivileged user, so they can't read the hidden tests on the mounted volume. |
| Competition environment | `cppgrade/agent_env.py`, `agent_tools.py`, `agent_tasks.py` | The multi-turn tool environment, used for evals and RL. `harness="ours"` is the original guess at the format before the competition harness code was made available. `harness="vals"` is the competition's exact tools, prompt and outputs. |
| Problem pools | `cppgrade/lcb.py`, `cppgrade/codeforces.py`, `posttrain/problems.py`, `posttrain/decontam.py` | Eval sets: LiveCodeBench-CPP v6 (287 AtCoder problems) and a 61-problem Codeforces test split. Training pool: 2,000 older Codeforces problems, split by contest into `distill` / `rl` / `holdout` parts and decontaminated against the evals. |
| Training | `posttrain/` (SFT, on-policy distillation, agentic RL), `scripts/train_*.py` | Tinker training loops built on the cookbook, plus Modal launchers (`*_modal.py`) so runs survive the laptop closing and can bill a second Tinker account. |
| Evals | `scripts/eval_lcb.py` (single attempt), `scripts/eval_agentic.py` (agentic), `scripts/eval_agentic_modal.py` | Grade checkpoints on the held-out problems. |

Infrastructure: **Tinker** handles sampling and LoRA training. **Modal** runs the CPU sandboxes for grading and the long-running training and eval jobs.

## Results

**Headline:** all 287 LiveCodeBench v6 AtCoder problems (Aug 2024–Apr 2025, never trained on), 2 samples each, in the competition-matched agentic harness:
- its tools and prompt
- earlier thinking kept
- 16K tokens per turn, 20 turns
- temperature 1.0

The comparison is the best checkpoint (`rl-v3` step 4) against its own base model, **Qwen3.5-9B**, problem by problem. Base Qwen3-8B is included for reference.

| | Base Qwen3.5-9B | `rl-v3` step 4 | Gain vs base 9B (95% CI) | Qwen3-8B (ref.) |
|---|---|---|---|---|
| Partial credit (0–100), all | 11.4 | **64.0** | **+52.6** [+47.9, +57.4] | 52.2 |
| — easy (67) | 45.4 | 94.0 | +48.6 [+38.1, +59.0] | 98.5 |
| — medium (68) | 2.9 | 82.7 | +79.8 [+72.0, +86.8] | 66.2 |
| — hard (152) | 0.0 | 42.4 | +42.4 [+36.3, +48.4] | 25.6 |
| pass@1 (all tests pass) | 11.0% | **53.3%** | +42.3 [+37.3, +47.4] | 45.6% |
| pass@2 | 14.6% | 62.7% | | 49.8% |
| No compiling submission | 88.7% | 24.2% | −64.5 | 41.5% |

Problem by problem, `rl-v3` step 4 scores higher on 228, lower on 2 and ties on 57.

Base Qwen3.5-9B struggles to finish within the harness's 16K tokens per turn. 81% of its sessions were cut off mid-reasoning before submitting, even on easy problems. Subsequently, most of the gain from RL is teaching the model to reason, test and submit **within the competition's budget**. On hard problems, the RL model also beats base Qwen3-8B (42.4 vs 25.6).


## Hackathon Methodology
#### What I tried over the course of this hackathon.

**0. Choosing the model.** Before any training, I ran an eval to determine Qwen3.5-9B's failure modes in order to perform more targeted post-training. 

- **9B's main failure was running out of tokens:** 41% of answers were cut off (66% on hard problems), against only 6% wrong answers.
- **The cut-off answers were approach search that never converged,** full of "wait / actually / different approach", rather than repetition.
- **It barely tested:** about 1.5 tool calls per task.

**1. On-policy distillation from Qwen3.8-27B (failed).**
I initially hypothesized that distillation could teach a model stronger coding reasoning that the teacher already has, which moves harder problems less out of distribution; distillation would therefore give a later RL attempt more headroom to hill-climb on harder problems.

My initial configuration used a learning rate of 5e-4 and ran for 30 steps; held out pass@1 went from 40% to 13% at step 10 and model answers degraded to gibberish. I hypothesize that several things contributed to this: my learning rate was too high, I used forwards KL instead of reverse KL, and student responses that hit token limits were kept rather than dropped. A second pass tried with a smaller learning rate, reverse KL and dropping too-long answers. The teacher was further given an editorial hint on how to solve the problem to encourage shorter reasoning; this run was cut off early after a high proportion of student answers hit token limits. 

After this, I abandoned distillation in favor of RL to conserve time left during the hackathon and because distillation seemed to be lengthening model responses past the token limit. 

**2. Agentic RL, five rounds.** Each round used the cookbook RL loop, group-relative advantages and LoRA rank 32, on the `rl` part of the Codeforces pool.

| Round | Change | Outcome |
|---|---|---|
| `rl-v1` | From base 9B. Reward = 40/60 group score − 0.01 per turn − 0.05 per malformed call. 16K per turn, 8 turns. | 50-problem eval 17.2 → 31.6. The model learned to submit instead of thinking until cut off. Ran out of problems after 16 of 25 steps, but the step 10 was able to hill-climb on my eval set, so future runs started from there. |
| *(audit)* | Ran a quick reward-hacking check using Claude of all sessions to sanity-check that intended behaviors were being trained | No deliberate hacking was found, but **submissions could read the hidden tests** (fixed with the jail), and wrong programs scored 60 on weak large-test groups (fixed: the score counts only if the statement examples pass). |
| `rl-v2b` | Partial-credit bonus measured above a constant-answer baseline — trivial solutions could earn several points without solving anything, which was regarded as the trivial baseline. Turn penalty 0.03. Wider ratings. 16 problems per step, with sampling 2 steps ahead (to speed up training time). | Model learned fewer turns but almost no testing, indicating that the turn penalty may have been too harsh. |
| `rl-v3` | Penalty only on turns **after the final submission**. Skip problems that were always or never solved to target problems that would give useful training signal. Groups of 8 for hard problems. | Eval **42.4**; **best competition score (~37)**. |
| `rl-v4` | **32K tokens per turn**, for longer reasoning. | Our 32K eval rose (29 → 42), but the competition gave 30: the harness caps turns at 16K. |
| `rl-v5t` | **Matched the harness exactly:** its tools, prompt and JSON outputs; proportional per-test reward; earlier thinking kept in the context, after a probe showed the endpoint keeps it; 20 turns. | Step 3 performed better on my eval set but worse on the competition's hidden set. A potential cause: the harness continues after a cut-off turn, for up to 100 turns at temperature 1.0, and our eval doesn't. |

**Things I learned and things I would try next**

1. Evals and RL setups should be as similar to the competition graders as possible; my fix came too late to iterate and produce meaningful capability gains.
2. Audit post-training runs for progress early — I could have found degenerate outputs in my distillation runs earlier and edited my hyperparameters early enough to try a full distillation run with the correct settings, saving time and compute.
3. RL did a good job of fixing Qwen3.5-9B's ability to work within a token budget and in an agentic setting, but I think either more RL or better distillation first would have targeted reasoning abilities more.
4. Distillation taught the model to produce longer answers — I should have included some sort of stopping signal or added privileged context to the teacher model to encourage conciseness.
5. If I'd had more time, I would have done some experimentation to determine what kind of privileged context can make the teacher model more concise, then tried another distillation run with better hyperparameters and giving the teacher that privileged context. I would have also done a more thorough job at replicating the competition's harness and identified the failure modes within that harness specifically before kicking off more RL to target those. 

## Documentation

- [scripts/README.md](scripts/README.md): smoke tests, the LiveCodeBench-CPP eval and the agentic eval
- [cppgrade/README.md](cppgrade/README.md): the grader
- [posttrain/README.md](posttrain/README.md): SFT, distillation and agentic RL

## Reference

- RL recipe to adapt: `tinker_cookbook/recipes/code_rl/` (DeepCoder-style, Python-graded; swap in `cppgrade.grade` for the reward).

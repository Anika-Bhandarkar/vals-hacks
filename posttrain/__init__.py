"""Post-training scaffolding (SFT now; distillation and RL build on the same pieces).

    problems.py   training problem pool (Codeforces train split) + prompts
    decontam.py   keep eval problems out of training data
    sft_data.py   one JSONL format for every SFT source, conversions, quality filters
    sft_dataset.py  cookbook dataset builder: JSONL -> per-turn training examples
    evaluators.py in-training evals (agentic score, single-attempt pass@1) on held-out eval problems
"""

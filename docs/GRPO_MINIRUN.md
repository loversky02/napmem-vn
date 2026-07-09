# GRPO Mini-Run Prep

This repo does not run GPU training locally. It now exports a small JSONL seed
set and reward harness for a Qwen3-4B/9B GRPO navigation run.

## Export

```bash
../automem-vn/.venv/bin/python scripts/export_grpo_data.py --out data/grpo_seed.jsonl
```

Each row contains:

- question prompt with routing hint
- gold answer and answer mode
- support layer and provenance ids
- note to compare `F+C+U` against `F+C`

## Reward Harness

`napmem.grpo.reward_candidate(candidate, row, use_usage_bonus=True)` implements
the paper-style terminal reward for a candidate JSON answer:

```json
{"answer":"...", "tool_calls":[{"tool":"search_records"}]}
```

Use `use_usage_bonus=False` for the honest-finding ablation. On non-memory
questions, a correct no-tool candidate scores `0.0` under `F+C+U` but `1.0`
under `F+C`, matching the full live artifact.

## Suggested RunPod Smoke

1. Load `data/grpo_seed.jsonl`.
2. Train Qwen3-4B for a tiny number of GRPO steps with group size 4.
3. Evaluate against `scripts/run_synthetic.py --live ...` style metrics:
   accuracy, exact fail, memory call rate, unnecessary memory call rate, and
   `R+U` vs `F+C`.

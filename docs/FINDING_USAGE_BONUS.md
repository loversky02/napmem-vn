# Finding: Usage Bonus Can Penalize Correct Memory Skipping

NapMem's paper-style terminal reward includes a usage term (`U`) that rewards
calling at least one memory tool. That is useful as a training nudge when the
task truly requires memory, but it has a sharp edge: non-memory questions should
not call memory at all.

## Local Evidence

Scripted 40-case synthetic:

- `oracle`: accuracy 1.00, unnecessary memory calls 0.00, `R+U=0.80`, `F+C=1.00`, delta `-0.20`.
- `passive_topk`: accuracy 0.40, unnecessary memory calls 1.00, `R+U=0.10`, `F+C=-0.20`, delta `+0.30`.

Prompted 9router mixed batch:

```bash
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --insecure-ssl --live-timeout 35 --qids q_raw_nate,q_record_nate,q_non_memory_math,q_non_memory_capital --artifacts results
```

Result: accuracy 1.00, average calls 1.00, unnecessary memory calls 0.00,
provider error rate 0.00, `R+U=0.50`, `F+C=1.00`, delta `-0.50`.

Versioned artifact: `results/live_prompted_mixed.md/json`.

## Interpretation

The usage bonus can improve exploration, but it should be an ablation axis rather
than a default claim. In this local repro, `U` rewards memory-calling baselines
even when they call memory unnecessarily, and it penalizes correct policies that
skip memory for non-memory examples.

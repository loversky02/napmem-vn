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

Prompted mixed8 repeats the pattern with four memory and four non-memory
examples: accuracy 1.00, unnecessary memory calls 0.00, provider error rate 0.00,
`R+U=0.50`, `F+C=1.00`, delta `-0.50`.

Versioned artifact: `results/live_prompted_mixed8.md/json`.

Prompted mixed16 repeats it again with eight memory and eight non-memory
examples: accuracy 1.00, unnecessary memory calls 0.00, provider error rate 0.00,
`R+U=0.50`, `F+C=1.00`, delta `-0.50`.

Versioned artifact: `results/live_prompted_mixed16.md/json`.

Full 40-case prompted run: accuracy 0.97, unnecessary memory calls 0.00,
provider error rate 0.00, `R+U=0.76`, `F+C=0.95`, delta `-0.19`.

Versioned artifact: `results/live_prompted_full40.md/json`.

## Reward-signal evidence on the GRPO seed (offline, $0)

The sharpest evidence is on the actual training seed, before any GPU. We score
four policy archetypes over the 40 seed rows under both reward variants
(`scripts/grpo_reward_smoke.py`, `results/grpo_reward_smoke.md`):

| archetype | R(F+C+U) | R(F+C) | F+C+U non-memory |
|---|---:|---:|---:|
| oracle (skip memory when not needed) | 0.80 | 1.00 | 0.00 |
| tool_spam (always call a tool) | **1.00** | 1.00 | 1.00 |
| no_tool (never call a tool) | 0.00 | 1.00 | 0.00 |

Under F+C+U, `tool_spam` (1.00) **outscores the ideal `oracle` policy** (0.80):
the U term pays for the unnecessary tool call on non-memory questions, so the
reward literally prefers a memory-spamming policy. This predicts that GRPO trained
with F+C+U will have a higher unnecessary memory-call rate than F+C — the money
plot in `docs/GRPO_MINIRUN.md`.

## Confirmed on a trained model (RunPod, GRPO)

The offline prediction holds — dramatically — after actually training. Same base
(`Qwen/Qwen2.5-3B-Instruct` + LoRA), 100 GRPO steps, neutral prompt (no skip hint,
so the tool decision is reward-driven), once with F+C+U and once with F+C
(`results/grpo_money_plot.md`, A40, ~$0.5):

| checkpoint | accuracy | memory-call rate | unnecessary (non-memory) |
|---|---:|---:|---:|
| base | 0.85 | 0.15 | 0.00 |
| F+C+U | 0.88 | **0.97** | **0.88** |
| F+C   | 0.88 | **0.17** | **0.00** |

The usage term trains a **memory-spamming** policy — it calls memory on 97% of
questions and 88% of *non-memory* questions unnecessarily — while the F+C ablation
stays at base behaviour, at equal accuracy. U buys tool-calling, not capability.
(Sampled eval n=8 agrees: 0.78 vs 0.00 unnecessary. An earlier pass with a "needs
no memory" hint in the prompt masked the effect on the non-memory subset — the hint
pinned the behaviour; removing it let the reward drive the policy.)

## Interpretation

The usage bonus can improve exploration, but it should be an ablation axis rather
than a default claim. In this local repro, `U` rewards memory-calling baselines
even when they call memory unnecessarily, penalizes correct policies that skip
memory for non-memory examples, and — once trained with GRPO — measurably inflates
how often the policy calls memory.

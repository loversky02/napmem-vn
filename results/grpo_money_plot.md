# GRPO Money Plot — F+C+U vs F+C (RunPod, real run)

Trained the same base model (`Qwen/Qwen2.5-3B-Instruct`, LoRA r=16) twice with
GRPO — once with the paper reward **F+C+U**, once with the **F+C** ablation — then
evaluated on the 40-row seed. Single A40, ~$0.5.

Setup: single-turn, evidence-in-prompt so a correct answer is reachable, and a
**neutral** instruction (no hint about when to skip memory) so the tool-call
decision is driven purely by the reward. This isolates the usage term U — including
whether it teaches *unnecessary* calls on non-memory questions.

## Result (100 steps, lr 1e-5)

| checkpoint | accuracy | memory recall | memory-call rate | unnecessary (non-memory) |
|---|---:|---:|---:|---:|
| base (no training) | 0.85 | 0.84 | 0.15 | 0.00 |
| **F+C+U** | 0.88 | 0.88 | **0.97** | **0.88** |
| **F+C** | 0.88 | 0.88 | **0.17** | **0.00** |

Sampled eval (n=8 per question, temperature 0.8) confirms it is not a greedy
artifact:

| checkpoint | accuracy | memory-call rate | unnecessary (non-memory) |
|---|---:|---:|---:|
| F+C+U | 0.88 | 0.94 | **0.78** |
| F+C   | 0.88 | 0.17 | **0.00** |

**The usage bonus U trains a memory-spamming policy: it calls a memory tool on 97%
of questions — 88% of the non-memory questions unnecessarily — while the F+C
ablation stays at base behaviour (17% / 0%), at equal (slightly higher) accuracy.**
This is textbook reward hacking, and it confirms the offline reward-signal analysis
(`results/grpo_reward_smoke.md`: tool_spam 1.00 > oracle 0.80 under F+C+U) in a real
trained model.

## Notes

- Accuracy and memory recall are unchanged (0.88 for both) — U buys tool-calling,
  not capability. The cost is 0.88 unnecessary memory calls per non-memory question.
- The effect needs a real gradient and a reward-driven (not instructed) decision: a
  first pass at **30 steps / lr 1e-6** with a "needs no memory" hint in the prompt
  gave a **zero** delta (too small an update, and the hint pinned the behaviour).
  Removing the hint and training 100 steps at lr 1e-5 surfaced the full effect.
- Reproduce: `MODEL=Qwen/Qwen2.5-3B-Instruct STEPS=100 LR=1e-5 NS=8 bash scripts/runpod_bootstrap.sh`
  (see `docs/GRPO_MINIRUN.md`).

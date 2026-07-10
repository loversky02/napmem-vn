# GRPO Money Plot — F+C+U vs F+C (RunPod, real run)

Trained the same base model (`Qwen/Qwen2.5-3B-Instruct`, LoRA r=16) twice with
GRPO — once with the paper reward **F+C+U**, once with the **F+C** ablation — then
evaluated greedily on the 40-row seed. Single RTX 3090, ~$0.55 total.

Setup: single-turn, evidence-in-prompt (so a correct answer is reachable and the
only free behaviour is whether the policy emits `tool_calls`). This isolates the
usage term U exactly as the offline reward smoke assumes.

## Result (100 steps, lr 1e-5)

| checkpoint | accuracy | memory recall | memory-call rate | unnecessary (non-memory) |
|---|---:|---:|---:|---:|
| F+C+U | 0.85 | 0.84 | **0.80** | 0.00 |
| F+C   | 0.85 | 0.84 | **0.65** | 0.00 |
| **delta** | 0.00 | 0.00 | **+0.15** | 0.00 |

**The usage bonus U trains the policy to call memory tools 15 points more often
(0.80 vs 0.65) at identical task accuracy (0.85).** This confirms, in a real
trained model, the reward-signal distortion the offline smoke predicted
(`results/grpo_reward_smoke.md`: tool_spam 1.00 > oracle 0.80 under F+C+U).

## Honest caveats

- The gap shows on the **overall** memory-call rate, not the non-memory subset
  (which stayed at 0.00 for both). The seed prompt explicitly flags non-memory
  questions as needing no memory; that instruction is a strong prior this small
  run did not overturn. Surfacing the effect on the non-memory subset would need
  more steps / a larger model / sampled (non-greedy) eval.
- A first pass at **30 steps, lr 1e-6** produced a **zero** delta on every metric
  (`results/grpo_money_plot.json`) — that update was too small to move the greedy
  policy. The effect only appeared after 100 steps at lr 1e-5.
- Accuracy and memory recall are unchanged, so U does not buy capability here — it
  only changes tool-use propensity, which is exactly the concern.

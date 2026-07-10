# NapMem-VN Program

## Context

Paper: **From Passive Retrieval to Active Memory Navigation: Learning to Use Memory as a Structured Action Space** ([arXiv:2607.05794](https://arxiv.org/abs/2607.05794)), submitted July 7, 2026.

Core move: build a four-layer memory pyramid and train the agent with GRPO to choose which memory granularity to inspect before answering.

## Method Notes

| Component | Paper detail | Local repro decision |
|---|---|---|
| L1 raw conversations | append-only message store with role, timestamp, id | `Message` JSONL with exact ids |
| L2 records | fact/event/instruction/preference, source ids, update metadata | `MemoryRecord` with provenance validation |
| L3 topic tracks | Markdown topic narratives, max 20 tracks | markdown files under `files/` |
| L4 profile | compact stable user profile, length budget | `profile.md` |
| Search | hybrid keyword + vector RRF, k=60, return top-5 | keyword + RRF placeholder; embedding slot later |
| Tools | search/get records, search/get conversations, read file | implemented in `MemoryTools` |
| Inference | max 4 tool-calling turns, stop when evidence sufficient | TODO scripted/prompted navigator |
| RL | GRPO, Qwen3.5-9B, group 4, lr 1e-6, 5 epochs | `scripts/train_grpo.py` (TRL GRPO, Qwen3-4B, group 4); reward fn unit-tested, launch-ready |
| Reward | binary format/correctness/usage, range -1..1 | implemented with U ablation |

## Do All Plan

| Track | Goal | First artifact |
|---|---|---|
| A. Independent repro | runnable NapMem pyramid/tools/reward + baseline navigator | this repo |
| B. AutoMem-VN READ half | combine learned WRITE with learned READ — **DONE**: flat top-k 0.60@0.8 vs scripted active 1.00@8 vs prompted (9router) 1.00@1.4 calls; gap above the record layer, prompted closes it efficiently | `../automem-vn/experiments/run_prompted_bridge.py` → `../automem-vn/results/napmem_bridge_prompted.md` |
| C. super-agent axis | route `memory-granularity` with model/depth/skill-plan | add memory axis policy and mock benchmark |
| D. HOLA story | compare text/tool memory vs exact cache memory | write cross-memory note and shared figure |
| E. Honest finding | test whether U usage bonus causes tool spam | reward ablation: `F+C+U` vs `F+C` |
| F. GPU money plot | **DONE** (RunPod A40, ~$0.5): F+C+U trains a memory-spamming policy (0.97 memory-call, 0.88 unnecessary on non-memory) vs F+C 0.17/0.00 at equal accuracy 0.88 — textbook reward hacking, confirming the offline smoke in a trained model | `results/grpo_money_plot.md` (Qwen2.5-3B-Instruct LoRA) |

## Current Offline Smoke

Command:

```bash
../automem-vn/.venv/bin/python scripts/run_synthetic.py
```

Result:

| strategy | acc | memory acc | evidence hit | exact fail | avg tool calls | unnecessary call | note |
|---|---:|---:|---:|---:|---:|---:|---|
| no_memory | 0.20 | 0.00 | 0.20 | 0.68 | 0.00 | 0.00 | non-memory control is clean but misses memory tasks |
| passive_topk | 0.40 | 0.25 | 0.40 | 0.47 | 1.00 | 1.00 | records-only retrieval misses upper/raw evidence |
| records_only | 0.40 | 0.25 | 0.40 | 0.47 | 2.00 | 1.00 | exact record fetch does not fix missing granularity |
| upper_first | 0.60 | 0.50 | 0.60 | 0.40 | 9.00 | 1.00 | profile/topic helps but over-calls |
| drilldown | 0.70 | 0.62 | 0.70 | 0.20 | 4.12 | 1.00 | active multi-level navigation improves exactness but still over-calls |
| oracle | 1.00 | 1.00 | 1.00 | 0.00 | 0.80 | 0.00 | upper bound with known supporting ids |

Reward wrinkle: the oracle's mean paper reward `F+C+U` is lower than its `F+C`
reward because correct non-memory answers without a memory call receive 0 instead
of 1. This supports making the U term an explicit ablation rather than copying it
uncritically.

Reward ablation on 40 cases:

| strategy | acc | exact fail | calls | unnecessary | R+U | F+C | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_memory | 0.20 | 0.68 | 0.00 | 0.00 | -0.80 | -0.60 | -0.20 |
| passive_topk | 0.40 | 0.47 | 1.00 | 1.00 | 0.10 | -0.20 | +0.30 |
| records_only | 0.40 | 0.47 | 2.00 | 1.00 | 0.10 | -0.20 | +0.30 |
| upper_first | 0.60 | 0.40 | 9.00 | 1.00 | 0.40 | 0.20 | +0.20 |
| drilldown | 0.70 | 0.20 | 4.12 | 1.00 | 0.55 | 0.40 | +0.15 |
| oracle | 1.00 | 0.00 | 0.80 | 0.00 | 0.80 | 1.00 | -0.20 |

Interpretation: the U term rewards every strategy that calls memory, including
ones with 100% unnecessary calls on non-memory examples. The oracle is penalized
because it correctly skips memory when memory is unnecessary.

Versioned artifact: `results/offline_synthetic.md` and
`results/offline_synthetic.json`. Standalone finding note:
`docs/FINDING_USAGE_BONUS.md`.

## Current 9router Prompted Smoke

Command:

```bash
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --limit 8 --insecure-ssl --live-timeout 35
```

Result using the existing workspace 9router endpoint/model after exact-evidence
discipline and per-example backend error handling:

| qid | ok | calls | quote ok | answer |
|---|---:|---:|---:|---|
| q_raw_nate | 1 | 2 | 1 | Tampa beach |
| q_record_nate | 1 | 2 | 1 | Nate visited Tampa beach on a turtle trip for peace and relaxation. |
| q_topic_state | 1 | 1 | 1 | Florida |
| q_profile_nate | 1 | 1 | 1 | peaceful beach trips |
| q_raw_allergy | 1 | 2 | 1 | almonds |
| q_record_mira | 1 | 2 | 1 | throat irritation |
| q_topic_mira | 0 | 1 | 1 | Mira avoids almonds in snack planning |
| q_profile_mira | 1 | 1 | 1 | almonds |

Summary: accuracy 0.88, average calls 1.50, unnecessary memory calls 0.00,
exact fail 0.00, quote fail 0.00, provider error rate 0.00. The live runner now
also reports `R+U`, `F+C`, and `delta` so prompted trajectories can be folded
into the reward-usage ablation.

Mixed prompted U-term check:

```bash
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --insecure-ssl --live-timeout 35 --qids q_raw_nate,q_record_nate,q_non_memory_math,q_non_memory_capital
```

Result: accuracy 1.00, average calls 1.00, unnecessary calls 0.00, provider
error rate 0.00, `R+U=0.50`, `F+C=1.00`, `delta=-0.50`. The prompted navigator
correctly skips memory for non-memory examples, but the paper-style usage term
penalizes those correct skips. This is now an observed generated-trajectory
finding, not just a scripted-policy artifact.

Good sign: first-tool routing and exact layer discipline repair the previous
record/profile failures without causing non-memory tool spam. Remaining issue:
semantic topic answers can be stricter than humans would score, which is useful
pressure for the next evaluator pass. Full 40-case live can stall on provider
requests, so the runner now has `--live-timeout` and prints per-example errors.

## Immediate Next

1. Run full 40-case prompted smoke with `--live-timeout` and collect provider error rate. (done: `results/live_prompted_full40.md`)
2. Run a larger prompted mixed batch and measure whether any model starts calling memory on non-memory examples to avoid the U penalty. (done: mixed8/mixed16)
3. Use the scripted + prompted U-term ablations as the first paper-forge honest finding. (done: `docs/FINDING_USAGE_BONUS.md`)
4. AutoMem bridge READ DONE: passive-vs-active comparison ($0,
   `../automem-vn/results/napmem_bridge_compare.md`, flat 0.60 vs active 1.00) +
   prompted navigation (9router, `../automem-vn/results/napmem_bridge_prompted.md`,
   1.00 recall @ 1.4 calls vs scripted 8).
5. GRPO money plot DONE (RunPod A40, ~$0.5, `results/grpo_money_plot.md`): with a
   neutral (unhinted) prompt so the tool decision is reward-driven, F+C+U trains a
   **memory-spamming** policy — 0.97 memory-call rate, **0.88 unnecessary** calls on
   non-memory questions — vs F+C's 0.17/0.00 at equal accuracy 0.88. Sampled eval
   (n=8) agrees (0.78 vs 0.00). Textbook reward hacking, confirming the offline smoke
   (tool_spam 1.00 > oracle 0.80) in a trained model. Every milestone A–F done.

## Open Risks

- Paper code is not released, so exact extraction/reconciliation prompts are approximated from appendix templates.
- LongMemEval and PersonaMem-v2 availability/licensing may shape what can be redistributed.
- The U reward term may inflate memory-call rate; this is a target finding, not a problem to hide.

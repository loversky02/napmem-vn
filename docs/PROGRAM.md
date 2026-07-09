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
| RL | GRPO, Qwen3.5-9B, group 4, lr 1e-6, 5 epochs | TODO RunPod Qwen3-4B/9B small run |
| Reward | binary format/correctness/usage, range -1..1 | implemented with U ablation |

## Do All Plan

| Track | Goal | First artifact |
|---|---|---|
| A. Independent repro | runnable NapMem pyramid/tools/reward + baseline navigator | this repo |
| B. AutoMem-VN READ half | combine learned WRITE with learned READ | import `MemoryPyramid` into `automem-vn` as a NapMem bridge |
| C. super-agent axis | route `memory-granularity` with model/depth/skill-plan | add memory axis policy and mock benchmark |
| D. HOLA story | compare text/tool memory vs exact cache memory | write cross-memory note and shared figure |
| E. Honest finding | test whether U usage bonus causes tool spam | reward ablation: `F+C+U` vs `F+C` |
| F. GPU money plot | no-RL prompted navigation vs GRPO navigation | Qwen3-4B/9B LoRA on RunPod |

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

## Current 9router Prompted Smoke

Command:

```bash
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --limit 6 --insecure-ssl
```

Result using the existing workspace 9router endpoint/model after exact-evidence
discipline:

| qid | ok | calls | quote ok | answer |
|---|---:|---:|---:|---|
| q_raw_nate | 1 | 1 | 1 | Tampa beach |
| q_record_nate | 0 | 1 | 0 | Nate traveled to Tampa beach during a turtle trip |
| q_topic_state | 1 | 1 | 1 | Florida |
| q_profile_nate | 0 | 1 | 0 | turtle trip |
| q_raw_allergy | 1 | 1 | 1 | almonds |
| q_record_mira | 1 | 2 | 1 | throat irritation |

Summary: accuracy 0.67, average calls 1.17, unnecessary memory calls 0.00,
exact fail 0.40, quote fail 0.40.

Good sign: non-memory tool spam remains controlled and raw/record exact answers
work when the model chooses the right layer. New failure: the live prompted model
overuses topic files for record/profile-shaped questions, so the next prompt/policy
needs stronger first-tool routing by question type.

## Immediate Next

1. Improve prompted first-tool routing: exact record/profile questions should start with records/profile, not topic files.
2. Run full 40-case prompted smoke after routing prompt is stable.
3. Use the 40-case reward ablation as the first paper-forge honest finding.
4. Connect AutoMem bridge output to prompted READ evaluation over real AutoMem traces.

## Open Risks

- Paper code is not released, so exact extraction/reconciliation prompts are approximated from appendix templates.
- LongMemEval and PersonaMem-v2 availability/licensing may shape what can be redistributed.
- The U reward term may inflate memory-call rate; this is a target finding, not a problem to hide.

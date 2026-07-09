# NapMem-VN

Mac-first reproduction lane for **From Passive Retrieval to Active Memory Navigation: Learning to Use Memory as a Structured Action Space** ([arXiv:2607.05794](https://arxiv.org/abs/2607.05794)).

NapMem turns long-term user memory from passive top-k retrieval into an agent action space. This repo starts with the $0 substrate: a four-layer memory pyramid, the five paper tools, and the reward rubric needed to test navigation and reward-hacking ablations before any GPU GRPO run.

## What Is Implemented

| Layer | Local representation |
|---|---|
| L1 raw conversations | structured messages with ids, role, timestamp, session |
| L2 memory records | typed fact/event/instruction/preference records with provenance |
| L3 topic tracks | Markdown files with evidence links |
| L4 user profile | `profile.md` |

Five tools:

- `search_records(query)`
- `search_conversations(query)`
- `get_records(record_ids)`
- `get_conversation(message_ids)`
- `read_file(name)`

Reward:

- paper `F + C + U` terminal reward
- `use_usage_bonus=False` ablation to test whether the usage term teaches useful navigation or tool spam

## Quickstart

```bash
../automem-vn/.venv/bin/python -m pytest -q
../automem-vn/.venv/bin/python scripts/run_synthetic.py
../automem-vn/.venv/bin/python scripts/run_synthetic.py --ablation
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --limit 8 --insecure-ssl --live-timeout 35
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --insecure-ssl --live-timeout 35 --qids q_raw_nate,q_record_nate,q_non_memory_math,q_non_memory_capital
```

Current 40-case synthetic result:

| strategy | acc | memory acc | exact fail | calls |
|---|---:|---:|---:|---:|
| no_memory | 0.20 | 0.00 | 0.68 | 0.00 |
| passive_topk | 0.40 | 0.25 | 0.47 | 1.00 |
| records_only | 0.40 | 0.25 | 0.47 | 2.00 |
| upper_first | 0.60 | 0.50 | 0.40 | 9.00 |
| drilldown | 0.70 | 0.62 | 0.20 | 4.12 |
| oracle | 1.00 | 1.00 | 0.00 | 0.80 |

This is intentionally tiny: it verifies that active navigation can recover evidence
from raw messages, records, topic tracks, and profile before we spend tokens or GPU.
It also includes non-memory controls, where unnecessary memory calls should be zero.

Current 9router prompted smoke using the existing workspace endpoint on the first
8 examples:

| split | result |
|---|---:|
| accuracy | 0.88 |
| avg calls | 1.50 |
| unnecessary memory calls | 0.00 |
| exact fail | 0.00 |
| quote fail | 0.00 |
| provider error rate | 0.00 |

The first live smoke failed by generalizing exact evidence. After scaling to 40,
the next limited smoke showed a routing/layer failure. The current prompt fixes
that with explicit first-tool routing and exact layer discipline: record-shaped
questions answer from record text, raw-shaped questions from raw text, profile
questions from `profile.md`, and topic questions from topic files. The live
runner also reports backend errors per example and summarizes `R+U` vs `F+C`
reward so prompted trajectories can be used in the U-term ablation.

On a mixed prompted batch with two memory and two non-memory questions, the
navigator answered all four correctly and skipped memory for both non-memory
items, but the paper-style `R+U` score dropped to 0.50 while `F+C` stayed 1.00.
That is the cleanest local evidence so far that U must be ablated.

## Portfolio Role

This is the independent NapMem repro. The same substrate should also be used as:

- AutoMem-VN READ half: pair active navigation with AutoMem's learned WRITE/consolidate skill.
- super-agent axis: add `memory-granularity` beside `model × depth × skill-plan`.
- HOLA comparison: external textual memory/tool navigation versus exact cache-style memory.

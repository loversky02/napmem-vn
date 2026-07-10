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
../automem-vn/.venv/bin/python scripts/run_synthetic.py --ablation --artifacts results
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --limit 8 --insecure-ssl --live-timeout 35
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --insecure-ssl --live-timeout 35 --qids q_raw_nate,q_record_nate,q_non_memory_math,q_non_memory_capital
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --insecure-ssl --live-timeout 35 --qids q_raw_nate,q_record_nate,q_topic_state,q_profile_nate,q_non_memory_math,q_non_memory_capital,q_non_memory_water,q_non_memory_week --artifacts results --live-artifact-stem live_prompted_mixed8
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --insecure-ssl --live-timeout 35 --qids q_raw_nate,q_record_nate,q_topic_state,q_profile_nate,q_raw_allergy,q_record_mira,q_topic_linh,q_profile_style,q_non_memory_math,q_non_memory_capital,q_non_memory_water,q_non_memory_week,q_non_memory_color,q_non_memory_planet,q_non_memory_square,q_non_memory_author --artifacts results --live-artifact-stem live_prompted_mixed16
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --insecure-ssl --live-timeout 35 --artifacts results --live-artifact-stem live_prompted_full40
../automem-vn/.venv/bin/python scripts/export_grpo_data.py --out data/grpo_seed.jsonl
../automem-vn/.venv/bin/python scripts/import_locomo_plus.py --limit 16 --out data/locomo_plus_subset.jsonl --report results/locomo_plus_subset.md --insecure-ssl
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

The larger mixed8 prompted batch repeats the same pattern: accuracy 1.00,
unnecessary memory calls 0.00, `R+U=0.50`, `F+C=1.00`.

The mixed16 batch repeats it again across 8 memory and 8 non-memory examples:
accuracy 1.00, unnecessary memory calls 0.00, provider error rate 0.00,
`R+U=0.50`, `F+C=1.00`.

The full 40-case live artifact keeps the finding at paper-repro scale for this
synthetic suite: accuracy 0.97, unnecessary memory calls 0.00, provider error
rate 0.00, `R+U=0.76`, `F+C=0.95`.

Versioned artifacts:

- [`results/offline_synthetic.md`](results/offline_synthetic.md)
- [`results/offline_synthetic.json`](results/offline_synthetic.json)
- [`results/live_prompted_mixed.md`](results/live_prompted_mixed.md)
- [`results/live_prompted_mixed.json`](results/live_prompted_mixed.json)
- [`results/live_prompted_mixed8.md`](results/live_prompted_mixed8.md)
- [`results/live_prompted_mixed8.json`](results/live_prompted_mixed8.json)
- [`results/live_prompted_mixed16.md`](results/live_prompted_mixed16.md)
- [`results/live_prompted_mixed16.json`](results/live_prompted_mixed16.json)
- [`results/live_prompted_full40.md`](results/live_prompted_full40.md)
- [`results/live_prompted_full40.json`](results/live_prompted_full40.json)
- [`data/grpo_seed.jsonl`](data/grpo_seed.jsonl)
- [`docs/GRPO_MINIRUN.md`](docs/GRPO_MINIRUN.md)
- [`data/locomo_plus_subset.jsonl`](data/locomo_plus_subset.jsonl)
- [`results/locomo_plus_subset.md`](results/locomo_plus_subset.md)
- [`docs/BENCHMARK_SUBSETS.md`](docs/BENCHMARK_SUBSETS.md)
- [`docs/FINDING_USAGE_BONUS.md`](docs/FINDING_USAGE_BONUS.md)

## Portfolio Role

This is the independent NapMem repro. The same substrate should also be used as:

- **AutoMem-VN READ half (Track B — done):** AutoMem's learned WRITE trace is
  bridged into this pyramid and read back by NapMem navigation. Three read policies
  over the same bridged trace: passive top-k over records (Mem0-style) **0.60** recall
  @ 0.8 calls; scripted active_nav (read every layer) **1.00** @ 8 calls; the prompted
  navigator (9router) **1.00** @ **1.4** calls — same recall, ~1/6 the calls, and 0
  calls on the non-memory question. The recall gap is entirely evidence that sits
  above the record layer (a consolidated topic track and the profile index); the
  learned policy is what makes the hierarchy efficient. See
  [`../automem-vn/results/napmem_bridge_compare.md`](../automem-vn/results/napmem_bridge_compare.md)
  ($0) and
  [`../automem-vn/results/napmem_bridge_prompted.md`](../automem-vn/results/napmem_bridge_prompted.md)
  (9router). Next: GRPO to train the policy.
- super-agent axis: add `memory-granularity` beside `model × depth × skill-plan`.
- HOLA comparison: external textual memory/tool navigation versus exact cache-style memory.

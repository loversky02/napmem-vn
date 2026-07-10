# NapMem-VN

Mac-first, $0-first reproduction of **From Passive Retrieval to Active Memory
Navigation: Learning to Use Memory as a Structured Action Space**
([arXiv:2607.05794](https://arxiv.org/abs/2607.05794), no official code).

NapMem turns long-term user memory from passive top-k retrieval into an **agent
action space**: a four-layer memory pyramid the agent learns to *navigate* with
tools, trained by GRPO. This repo reproduces the substrate and the navigation
policy, and — the headline — runs an **honest audit of the paper's reward**.

Everything runs offline on Apple Silicon; only the final GRPO run needs a GPU.

## Headline results

**1. Active navigation beats flat retrieval — and a learned policy makes it cheap.**
AutoMem's learned WRITE trace is bridged into the NapMem pyramid (see
[Portfolio role](#portfolio-role)); three read policies over the *same* memory:

| read policy | recall | avg tool calls |
|---|---:|---:|
| passive top-k over records (Mem0-style) | 0.60 | 0.8 |
| scripted active_nav (read every layer) | 1.00 | 8.0 |
| prompted navigator (9router) | 1.00 | **1.4** |

The recall gap is entirely evidence *above* the record layer (a consolidated topic
track + the profile index) that flat retrieval cannot reach; the learned policy
recovers it at ~1/6 the calls.

**2. The paper's usage-bonus reward trains a memory-spamming policy (reward hacking).**
GRPO on `Qwen2.5-3B-Instruct` + LoRA, paper reward **F+C+U** vs the **F+C** ablation
(RunPod A40, ~$0.5, [`results/grpo_money_plot.md`](results/grpo_money_plot.md)):

| checkpoint | accuracy | memory-call rate | unnecessary (non-memory) |
|---|---:|---:|---:|
| base | 0.85 | 0.15 | 0.00 |
| **F+C+U** | 0.88 | **0.97** | **0.88** |
| **F+C** | 0.88 | 0.17 | 0.00 |

U trains the policy to call memory on 97% of questions — 88% of *non-memory*
questions unnecessarily — at no accuracy gain (sampled eval n=8 agrees: 0.78 vs
0.00). This was predicted first, offline and free, by the reward smoke
([`results/grpo_reward_smoke.md`](results/grpo_reward_smoke.md): a tool-spamming
policy scores 1.00 under F+C+U vs the ideal policy's 0.80).

## Substrate

| Layer | Local representation |
|---|---|
| L1 raw conversations | structured messages with ids, role, timestamp, session |
| L2 memory records | typed fact/event/instruction/preference records with provenance |
| L3 topic tracks | Markdown files with evidence links |
| L4 user profile | `profile.md` |

Five paper tools: `search_records`, `search_conversations`, `get_records`,
`get_conversation`, `read_file`. Terminal reward `F + C + U` (format, correctness,
usage) with a `use_usage_bonus=False` ablation to test whether the usage term
teaches useful navigation or tool spam.

## Quickstart

```bash
# offline, $0 (shared venv from ../automem-vn)
../automem-vn/.venv/bin/python -m pytest -q                              # 27 tests
../automem-vn/.venv/bin/python scripts/run_synthetic.py --ablation       # strategy + U-term ablation
../automem-vn/.venv/bin/python scripts/grpo_reward_smoke.py --artifacts results   # reward-signal audit
../automem-vn/.venv/bin/python scripts/export_grpo_data.py --out data/grpo_seed.jsonl

# prompted navigation over 9router (OpenAI-compatible endpoint via .env)
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live --insecure-ssl --artifacts results

# GPU money plot (RunPod, see docs/GRPO_MINIRUN.md) — end to end on a fresh pod:
# MODEL=Qwen/Qwen2.5-3B-Instruct STEPS=100 LR=1e-5 NS=8 bash scripts/runpod_bootstrap.sh
```

## Results in detail

**Synthetic navigation ($0).** A balanced 40-case suite with evidence at every
layer plus non-memory controls, comparing scripted read strategies. Active
multi-level navigation (`drilldown`) recovers exact evidence that flat retrieval
misses (0.70 vs 0.40 accuracy) and the `oracle` upper bound hits 1.00 at 0.80
calls. Full table: [`results/offline_synthetic.md`](results/offline_synthetic.md).

**Prompted navigation (9router).** With first-tool routing and exact-layer
discipline, the prompted navigator reaches 0.88–1.00 accuracy at ~1.5 tool calls
and makes **zero** unnecessary memory calls on non-memory questions
(`live_prompted_*` artifacts).

**The usage term U is misspecified** ([`docs/FINDING_USAGE_BONUS.md`](docs/FINDING_USAGE_BONUS.md)).
Offline over the GRPO seed, a tool-spamming policy scores **1.00** under F+C+U
while the ideal policy (which correctly skips memory on non-memory questions)
scores **0.80** — the reward literally prefers spam. Training confirms it: see the
[headline money plot](#headline-results).

## Artifacts

- Navigation: [`results/offline_synthetic.md`](results/offline_synthetic.md) · live prompted [`mixed8`](results/live_prompted_mixed8.md) / [`mixed16`](results/live_prompted_mixed16.md) / [`full40`](results/live_prompted_full40.md)
- Reward audit: [`results/grpo_reward_smoke.md`](results/grpo_reward_smoke.md) · [`docs/FINDING_USAGE_BONUS.md`](docs/FINDING_USAGE_BONUS.md)
- GRPO money plot: [`results/grpo_money_plot.md`](results/grpo_money_plot.md) · [`scripts/train_grpo.py`](scripts/train_grpo.py) · [`scripts/eval_grpo.py`](scripts/eval_grpo.py) · [`docs/GRPO_MINIRUN.md`](docs/GRPO_MINIRUN.md) · [`data/grpo_seed.jsonl`](data/grpo_seed.jsonl)
- AutoMem→NapMem bridge (in the [automem-vn](https://github.com/loversky02/automem-vn) repo): [`napmem_bridge_compare.md`](https://github.com/loversky02/automem-vn/blob/main/results/napmem_bridge_compare.md) · [`napmem_bridge_prompted.md`](https://github.com/loversky02/automem-vn/blob/main/results/napmem_bridge_prompted.md)
- Benchmark subsets: [`docs/BENCHMARK_SUBSETS.md`](docs/BENCHMARK_SUBSETS.md) · [`results/locomo_plus_subset.md`](results/locomo_plus_subset.md)

## Portfolio role

The independent NapMem repro, paired with sibling projects around one memory theme:

- **AutoMem-VN READ half (Track B — done):** [AutoMem-VN](https://github.com/loversky02/automem-vn)
  learns the memory **WRITE**; NapMem learns the **READ**. AutoMem's WRITE trace is bridged
  into this pyramid and read back by the three policies in headline result #1 —
  passive top-k (0.60) < active nav (1.00 @ 8 calls) < prompted (1.00 @ 1.4 calls).
- **super-agent axis:** `memory-granularity` as a fourth routing axis beside
  `model × depth × skill-plan`.
- **HOLA comparison:** external textual memory/tool navigation vs exact cache-style
  memory ([HOLA](https://github.com/loversky02/hola)).

---

$0 / Apple-Silicon substrate; only the GRPO money plot needs a GPU (~$0.5 on a
rented A40). Not affiliated with the paper's authors.

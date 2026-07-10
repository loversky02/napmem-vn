# GRPO Mini-Run — the money plot

Train the NapMem navigation policy with GRPO and show what the reward does. We
train the *same* base model twice — once with the paper reward **F+C+U** and once
with the **F+C** ablation — and compare memory-call rate. Everything up to the GPU
boundary is built and verified offline; only `train_grpo.py` needs a pod.

## Result (done — RunPod, ~$0.5, `results/grpo_money_plot.md`)

`Qwen/Qwen2.5-3B-Instruct` + LoRA, 100 GRPO steps, lr 1e-5, single-turn with
evidence-in-prompt and a **neutral** instruction (no skip hint), so the tool-call
decision is reward-driven:

| checkpoint | accuracy | memory-call rate | unnecessary (non-memory) |
|---|---:|---:|---:|
| base | 0.85 | 0.15 | 0.00 |
| **F+C+U** | 0.88 | **0.97** | **0.88** |
| **F+C** | 0.88 | **0.17** | **0.00** |

**U trains a memory-spamming policy — it calls memory on 97% of questions, 88% of
non-memory questions unnecessarily — while F+C stays at base behaviour, at equal
accuracy.** Textbook reward hacking, confirming the offline prediction in a real
trained model (sampled eval n=8 agrees: 0.78 vs 0.00 unnecessary). Full write-up:
`results/grpo_money_plot.md`.

## 0. Offline reward-signal check (done, $0)

Before spending GPU, verify the training signal:

```bash
../automem-vn/.venv/bin/python scripts/grpo_reward_smoke.py --artifacts results
```

Result (`results/grpo_reward_smoke.md`) over the real 40-row seed:

| archetype | R(F+C+U) | R(F+C) | F+C+U memory | F+C+U non-memory |
|---|---:|---:|---:|---:|
| oracle (skip memory when not needed) | 0.80 | 1.00 | 1.00 | 0.00 |
| tool_spam (always call a tool) | **1.00** | 1.00 | 1.00 | 1.00 |
| no_tool (never call a tool) | 0.00 | 1.00 | 0.00 | 0.00 |
| wrong | -0.60 | -1.00 | -0.50 | -1.00 |

**Prediction:** under F+C+U, `tool_spam` (1.00) outscores the ideal `oracle`
(0.80) because the U term pays for the unnecessary tool call on non-memory
questions. So GRPO trained with F+C+U should end up with a **higher unnecessary
memory-call rate** than F+C. The GPU run below tests that prediction.

## 1. Export the seed (done)

```bash
../automem-vn/.venv/bin/python scripts/export_grpo_data.py --out data/grpo_seed.jsonl
```

40 rows, balanced across raw / record / topic / profile / non_memory (8 each),
32 memory + 8 non-memory. Columns: `prompt`, `gold_answer`, `answer_mode`,
`support_layer`, `requires_memory`, `route_hint`, `support`.

## 2. RunPod env (Qwen3-4B first)

Cheapest useful pod: a single A5000/A6000. Pin versions — this is the combination
that avoids the vllm/torch/cuda mismatch seen on prior runs:

```bash
pip install "torch==2.5.1" "transformers==4.51.*" "trl==0.17.*" "datasets>=2.19" "accelerate>=0.34"
# vllm optional (GRPO generation); if used, match its torch/cuda to the wheel above.
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Launch each run detached so an SSH drop does not kill it, and so it is easy to
reap:

```bash
setsid ../automem-vn/.venv/bin/python scripts/train_grpo.py \
    --model Qwen/Qwen3-4B --out runs/fcu --max-steps 60 --group-size 4 \
    > runs/fcu.log 2>&1 &

setsid python scripts/train_grpo.py \
    --model Qwen/Qwen3-4B --out runs/fc --max-steps 60 --group-size 4 --no-usage-bonus \
    > runs/fc.log 2>&1 &
```

Reap a run by pattern without matching the `pkill` itself:

```bash
pkill -f "[t]rain_grpo.py"
```

`train_grpo.py` imports torch/trl/datasets lazily, so it is importable and
`py_compile`-clean on the Mac; the reward function (`napmem.grpo.make_grpo_reward_fn`)
is unit-tested offline.

## 3. Evaluate both checkpoints (the plot)

Point the prompted navigator at each checkpoint's OpenAI-compatible server (vLLM
`--served-model-name`, or route through 9router) and run the existing eval:

```bash
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live \
    --insecure-ssl --artifacts results --live-artifact-stem grpo_fcu
../automem-vn/.venv/bin/python scripts/run_synthetic.py --live \
    --insecure-ssl --artifacts results --live-artifact-stem grpo_fc
```

Money plot = **unnecessary memory-call rate: F+C+U vs F+C** (plus accuracy and
`R+U` vs `F+C`). The offline smoke predicts F+C+U > F+C on unnecessary calls; the
bridged prompted result (`../automem-vn/results/napmem_bridge_prompted.md`) is the
no-RL efficient reference at 1.4 calls.

## 4. Teardown

Save `runs/*/` + `results/grpo_*` off the pod, then **terminate the pod** — it
bills by the hour.

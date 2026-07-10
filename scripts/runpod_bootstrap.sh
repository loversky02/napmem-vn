#!/usr/bin/env bash
# Drive the whole GRPO money plot on a fresh RunPod GPU pod, end to end.
#   MODEL=Qwen/Qwen2.5-3B-Instruct STEPS=100 LR=1e-5 bash scripts/runpod_bootstrap.sh
# Stages echo markers (SMOKE_OK / ALL_DONE) so a poller can track progress.
# Always writes to a log; drive it under tmux (or a kept-alive SSH) so it survives.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
STEPS="${STEPS:-100}"
LR="${LR:-1e-5}"
NS="${NS:-8}"          # samples per question for the probabilistic eval
PY="${PY:-python}"
echo "MODEL=$MODEL STEPS=$STEPS LR=$LR NS=$NS"

echo "=== [1/6] install pinned deps ==="
pip install -q "transformers==4.51.*" "trl==0.17.*" "peft>=0.11,<0.15" "datasets>=2.19" "accelerate>=0.34" \
  || { echo "PINNED_INSTALL_FAILED, trying loose"; pip install -q transformers trl peft datasets accelerate; }
$PY -c "import torch,trl,peft,transformers as t; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'trl',trl.__version__,'tf',t.__version__)" || { echo "IMPORT_FAILED"; exit 1; }

echo "=== [2/6] offline reward smoke (\$0 sanity) ==="
$PY scripts/grpo_reward_smoke.py --artifacts results || { echo "REWARD_SMOKE_FAILED"; exit 1; }

echo "=== [3/6] GPU smoke: 3 GRPO steps (catches OOM / API drift; expect reward_std>0) ==="
$PY scripts/train_grpo.py --model "$MODEL" --out runs/smoke --max-steps 3 --group-size 4 --max-completion-length 96 --lr "$LR" || { echo "GPU_SMOKE_FAILED"; exit 1; }
echo "SMOKE_OK"

echo "=== [4/6] train FCU (F+C+U) and FC (F+C ablation) ==="
$PY scripts/train_grpo.py --model "$MODEL" --out runs/fcu --max-steps "$STEPS" --group-size 4 --max-completion-length 96 --lr "$LR" || { echo "TRAIN_FCU_FAILED"; exit 1; }
$PY scripts/train_grpo.py --model "$MODEL" --out runs/fc  --max-steps "$STEPS" --group-size 4 --max-completion-length 96 --lr "$LR" --no-usage-bonus || { echo "TRAIN_FC_FAILED"; exit 1; }

echo "=== [5/6] eval base + both checkpoints (greedy, then sampled) ==="
$PY scripts/eval_grpo.py --base "$MODEL" --label base --out results/grpo_eval_base.json || echo "EVAL_BASE_FAILED"
$PY scripts/eval_grpo.py --base "$MODEL" --adapter runs/fcu --label fcu --out results/grpo_eval_fcu.json || echo "EVAL_FCU_FAILED"
$PY scripts/eval_grpo.py --base "$MODEL" --adapter runs/fc  --label fc  --out results/grpo_eval_fc.json  || echo "EVAL_FC_FAILED"
$PY scripts/eval_grpo.py --base "$MODEL" --adapter runs/fcu --label fcu_sampled --n-samples "$NS" --out results/grpo_eval_fcu_sampled.json || echo "EVAL_FCU_S_FAILED"
$PY scripts/eval_grpo.py --base "$MODEL" --adapter runs/fc  --label fc_sampled  --n-samples "$NS" --out results/grpo_eval_fc_sampled.json  || echo "EVAL_FC_S_FAILED"

echo "=== [6/6] money plot ==="
$PY - <<'PY'
import json, os
def m(p):
    try: return json.load(open(p))["metrics"]
    except Exception: return {}
g = {k: m(f"results/grpo_eval_{k}.json") for k in ("base","fcu","fc")}
s = {k: m(f"results/grpo_eval_{k}_sampled.json") for k in ("fcu","fc")}
print("== greedy ==")
print("ckpt   acc   mem_recall  mem_call  unnecessary")
for k in ("base","fcu","fc"):
    x=g[k]
    if x: print(f"{k:<5}  {x['accuracy']:.2f}  {x['memory_recall']:.2f}       {x['memory_call_rate']:.2f}      {x['unnecessary_memory_call_rate']:.2f}")
print("== sampled (n) ==")
for k in ("fcu","fc"):
    x=s[k]
    if x: print(f"{k:<5}  {x['accuracy']:.2f}  {x['memory_recall']:.2f}       {x['memory_call_rate']:.2f}      {x['unnecessary_memory_call_rate']:.2f}")
out={"model":os.environ.get("MODEL"),"steps":os.environ.get("STEPS"),"lr":os.environ.get("LR"),"greedy":g,"sampled":s}
if g["fcu"] and g["fc"]:
    out["greedy_delta_mem_call"]=round(g["fcu"]["memory_call_rate"]-g["fc"]["memory_call_rate"],3)
    out["greedy_delta_unnecessary"]=round(g["fcu"]["unnecessary_memory_call_rate"]-g["fc"]["unnecessary_memory_call_rate"],3)
if s["fcu"] and s["fc"]:
    out["sampled_delta_mem_call"]=round(s["fcu"]["memory_call_rate"]-s["fc"]["memory_call_rate"],3)
    out["sampled_delta_unnecessary"]=round(s["fcu"]["unnecessary_memory_call_rate"]-s["fc"]["unnecessary_memory_call_rate"],3)
    print("\nsampled delta unnecessary (fcu-fc):", out["sampled_delta_unnecessary"], "| sampled delta mem_call:", out["sampled_delta_mem_call"])
json.dump(out, open("results/grpo_money_plot.json","w"), indent=2)
print("wrote results/grpo_money_plot.json")
PY
echo "ALL_DONE"
